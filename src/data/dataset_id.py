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

from datasets import load_dataset

V3_REPO = "nvidia/Nemotron-Safety-Guard-Dataset-v3"
V35_REPO = "nvidia/Nemotron-3.5-Content-Safety-Dataset"

GERMAN_LANGUAGE_CODES = {"de", "german", "deutsch", "de-de"}
VALID_LABELS = {"safe", "unsafe"}

SPLIT_NAME_ALIASES = {"valid": "validation", "val": "validation", "dev": "validation"}

COMPARE_COLUMN = "_normalized_prompt"

def normalize_text(text):
    lowercase = str(text).lower()
    without_punctuation = re.sub(r"\W+", " ", lowercase)
    return without_punctuation.strip()


def load_german_splits(repo_name, label_column):
    
    full_dataset = load_dataset(repo_name)
    german_splits = {}

    for split_name, split_data in full_dataset.items():
        table = split_data.to_pandas()

        for required_column in ("language", "prompt", label_column):
            if required_column not in table.columns:
                sys.exit(f"{repo_name}: Spalte '{required_column}' fehlt. "
                         f"Vorhanden: {list(table.columns)}")

        is_german = table["language"].astype(str).str.lower().isin(GERMAN_LANGUAGE_CODES)
        table = table[is_german]

        has_valid_label = table[label_column].astype(str).str.lower().isin(VALID_LABELS)
        table = table[has_valid_label].copy()

        table[COMPARE_COLUMN] = table["prompt"].map(normalize_text)
        has_real_text = ~table[COMPARE_COLUMN].isin(["", "redacted"])
        table = table[has_real_text]

        unified_split_name = SPLIT_NAME_ALIASES.get(split_name, split_name)
        german_splits[unified_split_name] = table.reset_index(drop=True)

    return german_splits


def find_similar_texts(texts_to_check, reference_texts, similarity_threshold, batch_size=200):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    reference_vectors = vectorizer.fit_transform(reference_texts)
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
                    output_folder):
    table = table.drop_duplicates(COMPARE_COLUMN)

    is_exact_overlap = table[COMPARE_COLUMN].isin(forbidden_texts)
    table = table[~is_exact_overlap]

    if use_fuzzy and len(table) > 0:
        is_similar, highest_similarity, most_similar_text = find_similar_texts(
            table[COMPARE_COLUMN].tolist(), list(forbidden_texts), similarity_threshold)

        # Entfernte Zeilen zum Nachschauen speichern, die ähnlichsten zuerst
        removed_rows = table.copy()
        removed_rows["similarity"] = highest_similarity.round(3)
        removed_rows["most_similar_training_text"] = most_similar_text
        removed_rows = removed_rows[is_similar].sort_values("similarity", ascending=False)
        report_file = os.path.join(output_folder, f"removed_similar_{split_name}.csv")
        removed_rows[["similarity", "prompt", COMPARE_COLUMN, "most_similar_training_text",
                      "prompt_label"]].to_csv(report_file, index=False, sep=";",
                                              encoding="utf-8-sig")

        table = table[~is_similar]

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

    v3_splits = load_german_splits(V3_REPO, label_column="prompt_label")
    v35_splits = load_german_splits(V35_REPO, label_column="input_label")

    train = v3_splits["train"].drop_duplicates(COMPARE_COLUMN).reset_index(drop=True)

    # Alle Prompts, die irgendwo im Training vorkommen (v3 und 3.5)
    all_training_texts = set(train[COMPARE_COLUMN]) | set(v35_splits["train"][COMPARE_COLUMN])

    os.makedirs(args.out, exist_ok=True)

    # Test: nichts aus dem Training
    test = remove_overlaps(v3_splits["test"], all_training_texts, "test",
                           args.fuzzy, args.threshold, args.out)

    # Validation: nichts aus dem Training und nichts aus dem Test
    texts_not_allowed_in_validation = all_training_texts | set(test[COMPARE_COLUMN])
    validation = remove_overlaps(v3_splits["validation"], texts_not_allowed_in_validation,
                                 "validation", args.fuzzy, args.threshold, args.out)

    # Speichern: eine lesbare JSONL-Datei pro Split (eine Zeile = ein Eintrag)
    final_splits = {"train": train, "validation": validation, "test": test}
    for split_name, table in final_splits.items():
        output_file = os.path.join(args.out, f"{split_name}.jsonl")
        table = table.drop(columns=COMPARE_COLUMN)
        table["source"] = "v3"
        table.to_json(output_file, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()