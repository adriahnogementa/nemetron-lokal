import os
import logging
import pandas as pd
from datasets import load_dataset


EXPGUARD_TEST_DS = "6rightjade/expguardmix"
POLYGUARD_DS = "Virtue-AI-HUB/PolyGuard"
OUTPUT_FILE_EXPGUARD = "../data_ood/expguard_finance_test.jsonl"
OUTPUT_FILE_POLYGUARD = "../data_ood/polyguard_finance_test.jsonl"
LOG_FILE = "../data_ood/prepare_ood_data.log"

TEXT_COLUMNS = ["prompt", "response", "scenario", "category"]


# ------------------------------------------------------------------ Logging

def setup_logging(log_file=LOG_FILE):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )


log = logging.getLogger("ood_prep")


def log_step(name, before, after):
    """Loggt, wie viele Zeilen ein Schritt entfernt hat."""
    removed = before - after
    pct = (removed / before * 100) if before else 0.0
    log.info(f"[{name}] {before} -> {after} Zeilen (entfernt: {removed}, {pct:.2f} %)")


def log_distribution(df, column, title):
    """Loggt die Verteilung einer Spalte (z. B. safe/unsafe)."""
    if column not in df.columns:
        return
    counts = df[column].value_counts(dropna=False)
    log.info(f"{title} ({column}):")
    for value, n in counts.items():
        log.info(f"    {value}: {n} ({n / len(df) * 100:.1f} %)")


def clean_text(s):
    if isinstance(s, str):
        return s.replace("\u2028", "\n").replace("\u2029", "\n\n").strip()
    return s


def clean_columns(df, columns=TEXT_COLUMNS):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(clean_text)
    return df


# ---------------------------------------------------------------- PolyGuard

def load_datasets_from_polyguard(dataset_name=POLYGUARD_DS):
    log.info(f"Lade PolyGuard ({dataset_name}, Config 'finance_input') ...")
    finance_split_dataset = load_dataset(dataset_name, "finance_input")

    frames = []
    for split_name, split in finance_split_dataset.items():  # z. B. "FINRA_unsafe"
        source, label = split_name.rsplit("_", 1)
        df = split.to_pandas()

        if label == "unsafe":
            df["prompt"] = df["rephrased instance"]
            prompt_col = "rephrased instance"
        else:
            df["prompt"] = df["original instance"]
            prompt_col = "original instance"
        df["policy_source"] = source
        df["prompt_label"] = label
        frames.append(df)

        log.info(f"    Split {split_name:<15} {len(df):>5} Zeilen (Prompt aus '{prompt_col}')")

    table = pd.concat(frames, ignore_index=True)
    log.info(f"PolyGuard geladen: {len(table)} Zeilen aus {len(frames)} Splits")
    return table


def prepare_polyguard_finance(table):
    log.info("Bereite PolyGuard auf ...")
    n_start = len(table)

    out = pd.DataFrame({
        "id": [f"polyguard_{i}" for i in range(len(table))],
        "prompt": table["prompt"].values,
        "response": None,
        "prompt_label": table["prompt_label"].values,
        "response_label": None,
        "category": table["category name"].values,
        "taxonomy": "polyguard",
        "domain": "finance",
        "scenario": table["rule"].values,
        "policy_source": table["policy_source"].values,
        "language": "en",
        "source": "polyguard",
    })

    out = clean_columns(out)

    # Leere Prompts / Platzhalter
    is_na = out["prompt"].isna()
    is_placeholder = out["prompt"].astype(str).str.strip().isin(["", "Not applicable"])
    bad = is_na | is_placeholder
    log.info(f"[Leere/Platzhalter-Prompts] NaN: {is_na.sum()}, leer/'Not applicable': {(is_placeholder & ~is_na).sum()}")
    if bad.any():
        log.error(f"{bad.sum()} leere/Platzhalter-Prompts gefunden, betroffene Splits:")
        for (src, lab), n in out[bad].groupby(["policy_source", "prompt_label"]).size().items():
            log.error(f"    {src}_{lab}: {n}")
    assert not bad.any(), f"{bad.sum()} leere/Platzhalter-Prompts gefunden"

    # Duplikate
    before = len(out)
    dup_mask = out.duplicated("prompt", keep="first")
    if dup_mask.any():
        log.info("Duplikate nach Quelle/Label (entfernt werden die späteren Vorkommen):")
        for (src, lab), n in out[dup_mask].groupby(["policy_source", "prompt_label"]).size().items():
            log.info(f"    {src}_{lab}: {n}")
        # Duplikate, deren Label sich widersprechen (gleicher Prompt einmal safe, einmal unsafe)
        conflicting = out[out["prompt"].duplicated(keep=False)].groupby("prompt")["prompt_label"].nunique()
        n_conflict = int((conflicting > 1).sum())
        if n_conflict:
            log.warning(f"{n_conflict} Prompts kommen mit widersprüchlichen Labels vor (safe und unsafe)")
    out = out.drop_duplicates("prompt").reset_index(drop=True)
    log_step("Duplikate (prompt)", before, len(out))

    out["id"] = [f"polyguard_{i}" for i in range(len(out))]

    log_step("PolyGuard gesamt", n_start, len(out))
    log_distribution(out, "prompt_label", "Label-Verteilung PolyGuard")
    log_distribution(out, "policy_source", "Policy-Quellen PolyGuard")
    return out


# ----------------------------------------------------------------- ExpGuard

def load_dataset_splits_from_expguard(dataset_name=EXPGUARD_TEST_DS):
    log.info(f"Lade ExpGuard ({dataset_name}, Config 'expguardtest', Split 'test') ...")
    test_split_dataset = load_dataset(dataset_name, "expguardtest", split="test")
    table = test_split_dataset.to_pandas()
    log.info(f"ExpGuard geladen: {len(table)} Zeilen")
    log_distribution(table, "domain", "Domänen im ExpGuardTest")

    is_finance = table["domain"].astype(str).str.lower().str.strip() == "finance"
    finance = table[is_finance].copy()
    log_step("Filter domain == finance", len(table), len(finance))
    return finance


def prepare_expguard_finance(table):
    log.info("Bereite ExpGuard auf ...")
    n_start = len(table)

    out = pd.DataFrame({
        "id": [f"expguard_{i}" for i in range(len(table))],
        "prompt": table["prompt"].values,
        "response": table["response"].values,
        "prompt_label": table["prompt_label"].astype(str).str.lower().str.strip().values,
        "response_label": table["response_label"].str.lower().str.strip().values,
        "category": table["prompt_category"].values,
        "taxonomy": "expguard",
        "domain": table["domain"].values,
        "scenario": table["scenario"].values,
        "policy_source": None,
        "language": "en",
        "source": "expguard",
    })

    out = clean_columns(out)

    # Label-Filter
    before = len(out)
    invalid = ~out["prompt_label"].isin(["safe", "unsafe"])
    if invalid.any():
        log.info("Entfernte Werte in prompt_label:")
        for value, n in out.loc[invalid, "prompt_label"].value_counts(dropna=False).items():
            log.info(f"    '{value}': {n}")
    out = out[~invalid]
    log_step("Filter prompt_label in {safe, unsafe}", before, len(out))

    # Duplikate
    before = len(out)
    n_dup_prompt_only = int(out.duplicated("prompt").sum())
    out = out.drop_duplicates(["prompt", "response"]).reset_index(drop=True)
    log_step("Duplikate (prompt + response)", before, len(out))
    n_prompt_dups_left = int(out.duplicated("prompt").sum())
    log.info(
        f"Hinweis: {n_dup_prompt_only} Zeilen hatten vorher einen doppelten Prompt; "
        f"{n_prompt_dups_left} davon bleiben, weil sich die Response unterscheidet"
    )

    log_step("ExpGuard gesamt", n_start, len(out))
    log_distribution(out, "prompt_label", "Label-Verteilung ExpGuard")
    return out


# --------------------------------------------------------------------- Main

def save_jsonl(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    log.info(f"Gespeichert: {path}  ({len(df)} Zeilen)")


def main():
    setup_logging()
    log.info("=" * 60)
    log.info("Start der OOD-Datenaufbereitung")
    log.info("=" * 60)

    ood_expg = prepare_expguard_finance(load_dataset_splits_from_expguard())
    save_jsonl(ood_expg, OUTPUT_FILE_EXPGUARD)

    log.info("-" * 60)

    ood_poly = prepare_polyguard_finance(load_datasets_from_polyguard())
    save_jsonl(ood_poly, OUTPUT_FILE_POLYGUARD)

    log.info("=" * 60)
    log.info(f"Fertig. ExpGuard: {len(ood_expg)} Zeilen, PolyGuard: {len(ood_poly)} Zeilen")
    log.info(f"Log-Datei: {LOG_FILE}")


if __name__ == "__main__":
    main()