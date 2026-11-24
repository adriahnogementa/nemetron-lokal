#!/usr/bin/env python3
"""
Aufruf
  python build_de_dataset.py --fuzzy --threshold 0.9 --out data_de

Benutzen
  from datasets import load_dataset
  dataset = load_dataset("json", data_files={
      "train": "data_de/train.jsonl",
      "validation": "data_de/validation.jsonl",
      "test": "data_de/test.jsonl",
  })
"""
import argparse
import os
import re
import sys

import pandas as pd
from datasets import load_dataset
from similarity_report import write_similarity_report

V3_REPO = "nvidia/Nemotron-Safety-Guard-Dataset-v3"
V35_REPO = "nvidia/Nemotron-3.5-Content-Safety-Dataset"

V35_TO_V3_COLUMNS = {
    "row_id": "id",
    "input_label": "prompt_label",
    "input_label_source": "prompt_label_source",
}

GERMAN_LANGUAGE_CODES = {"de", "german", "deutsch", "de-de"}
VALID_LABELS = {"safe", "unsafe"}

SPLIT_NAME_ALIASES = {"valid": "validation", "val": "validation", "dev": "validation"}

COMPARE_COLUMN = "_normalized_prompt"


# ---------- Protokoll ----------

FILTER_LOG = []


def log_step(dataset, split, step, before, after):
    """Hält fest, wie viele Zeilen ein Schritt entfernt hat, und gibt es aus."""
    removed = before - after
    share = removed / before if before else 0.0
    FILTER_LOG.append({"dataset": dataset, "split": split, "step": step,
                       "before": before, "removed": removed, "after": after,
                       "removed_pct": round(share * 100, 1)})
    print(f"[{dataset:4s} | {split:10s}] {step:38s} "
          f"{before:7d} -> {after:7d}   (-{removed:6d}, {share:6.1%})")


def write_filter_log(output_folder):
    log = pd.DataFrame(FILTER_LOG)
    path = os.path.join(output_folder, "filter_log.csv")
    log.to_csv(path, index=False)
    print(f"\nProtokoll gespeichert: {path}")


# ---------- Laden und Filtern ----------

def normalize_text(text):
    lowercase = str(text).lower()
    without_punctuation = re.sub(r"\W+", " ", lowercase)
    return without_punctuation.strip()


def load_german_splits(repo_name, label_column, dataset_tag):
    full_dataset = load_dataset(repo_name)
    german_splits = {}

    for split_name, split_data in full_dataset.items():
        unified_split_name = SPLIT_NAME_ALIASES.get(split_name, split_name)
        table = split_data.to_pandas()

        for required_column in ("language", "prompt", label_column):
            if required_column not in table.columns:
                sys.exit(f"{repo_name}: Spalte '{required_column}' fehlt. "
                         f"Vorhanden: {list(table.columns)}")

        n = len(table)
        is_german = table["language"].astype(str).str.lower().isin(GERMAN_LANGUAGE_CODES)
        table = table[is_german]
        log_step(dataset_tag, unified_split_name, "nur Deutsch", n, len(table))

        n = len(table)
        has_valid_label = table[label_column].astype(str).str.lower().isin(VALID_LABELS)
        table = table[has_valid_label].copy()
        log_step(dataset_tag, unified_split_name, "gültiges Label (safe/unsafe)", n, len(table))

        n = len(table)
        table[COMPARE_COLUMN] = table["prompt"].map(normalize_text)
        has_real_text = ~table[COMPARE_COLUMN].isin(["", "redacted"])
        table = table[has_real_text]
        log_step(dataset_tag, unified_split_name, "ohne leere/REDACTED-Prompts", n, len(table))

        german_splits[unified_split_name] = table.reset_index(drop=True)

    return german_splits


def find_similar_texts(texts_to_check, reference_texts, similarity_threshold, batch_size=200):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    vectorizer.fit(list(reference_texts) + list(texts_to_check))
    reference_vectors = vectorizer.transform(reference_texts)
    check_vectors = vectorizer.transform(texts_to_check)

    highest_similarity = np.zeros(check_vectors.shape[0])
    best_match_position = np.zeros(check_vectors.shape[0], dtype=int)
    for start in range(0, check_vectors.shape[0], batch_size):
        end = start + batch_size
        similarities = check_vectors[start:end] @ reference_vectors.T
        highest_similarity[start:end] = similarities.max(axis=1).toarray().ravel()
        best_match_position[start:end] = np.asarray(similarities.argmax(axis=1)).ravel()

    is_similar = highest_similarity >= similarity_threshold
    most_similar_text = [reference_texts[position] for position in best_match_position]
    return is_similar, highest_similarity, most_similar_text


def remove_overlaps(table, forbidden_texts, split_name, use_fuzzy, similarity_threshold,
                    output_folder, dataset_tag="v3"):
    n = len(table)
    table = table.drop_duplicates(COMPARE_COLUMN)
    log_step(dataset_tag, split_name, "Duplikate innerhalb des Splits", n, len(table))

    n = len(table)
    is_exact_overlap = table[COMPARE_COLUMN].isin(forbidden_texts)
    table = table[~is_exact_overlap]
    log_step(dataset_tag, split_name, "exakte Überschneidung", n, len(table))

    if use_fuzzy and len(table) > 0:
        is_similar, highest_similarity, most_similar_text = find_similar_texts(
            table[COMPARE_COLUMN].tolist(), list(forbidden_texts), similarity_threshold)

        write_similarity_report(table[COMPARE_COLUMN].tolist(), highest_similarity,
                                most_similar_text, similarity_threshold,
                                output_folder, split_name, show_from=0.7)

        print(f"\n  Mögliche Schwellenwerte für {split_name}:")
        for t in [0.90, 0.85, 0.80, 0.75]:
            removed = highest_similarity >= t
            print(f"    ab {t:.2f}: {removed.sum():6d} entfernt ({removed.mean():.1%})")
        print()

        n = len(table)
        table = table[~is_similar]
        log_step(dataset_tag, split_name, f"Beinahe-Duplikate (TF-IDF >= {similarity_threshold})",
                 n, len(table))

    return table.reset_index(drop=True)


# ---------- Hauptprogramm ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="../data_in", help="Zielordner")
    parser.add_argument("--fuzzy", action="store_true",
                        help="auch Beinahe-Duplikate entfernen (dauert länger)")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Ähnlichkeitsgrenze für --fuzzy")
    args = parser.parse_args()

    print("=== Laden ===")
    v3_splits = load_german_splits(V3_REPO, label_column="prompt_label", dataset_tag="v3")
    v35_splits = load_german_splits(V35_REPO, label_column="input_label", dataset_tag="v3.5")

    print("\n=== Train ===")
    n = len(v35_splits["train"])
    train = v35_splits["train"].drop_duplicates(COMPARE_COLUMN).reset_index(drop=True)
    log_step("v3.5", "train", "Duplikate innerhalb des Splits", n, len(train))
    train = train.rename(columns=V35_TO_V3_COLUMNS)

    # Alle Prompts, die irgendwo im Training vorkommen (v3 und 3.5)
    all_training_texts = set(train[COMPARE_COLUMN]) | set(v3_splits["train"][COMPARE_COLUMN])
    print(f"Referenz für Überschneidungen: {len(all_training_texts)} eindeutige Trainings-Prompts")

    os.makedirs(args.out, exist_ok=True)

    print("\n=== Test ===")
    test = remove_overlaps(v3_splits["test"], all_training_texts, "test",
                           args.fuzzy, args.threshold, args.out)

    print("\n=== Validation ===")
    texts_not_allowed_in_validation = all_training_texts | set(test[COMPARE_COLUMN])
    validation = remove_overlaps(v3_splits["validation"], texts_not_allowed_in_validation,
                                 "validation", args.fuzzy, args.threshold, args.out)

    print("\n=== Speichern ===")
    sources = {"train": "v3.5", "validation": "v3", "test": "v3"}
    final_splits = {"train": train, "validation": validation, "test": test}
    for split_name, table in final_splits.items():
        output_file = os.path.join(args.out, f"{split_name}.jsonl")
        table = table.drop(columns=COMPARE_COLUMN)
        table["source"] = sources[split_name]
        table.to_json(output_file, orient="records", lines=True, force_ascii=False)
        print(f"{split_name:10s}: {len(table):6d} Zeilen -> {output_file}")

    write_filter_log(args.out)


if __name__ == "__main__":
    main()