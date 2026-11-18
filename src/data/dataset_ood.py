import os
import pandas as pd
from datasets import load_dataset


EXPGUARD_TEST_DS = "6rightjade/expguardmix"
POLYGUARD_DS = "Virtue-AI-HUB/PolyGuard"
OUTPUT_FILE_EXPGUARD = "../data_ood/expguard_finance_test.jsonl"
OUTPUT_FILE_POLYGUARD = "../data_ood/polyguard_finance_test.jsonl"

TEXT_COLUMNS = ["prompt", "response", "scenario", "category"]


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
    finance_split_dataset = load_dataset(dataset_name, "finance_input")

    frames = []
    for split_name, split in finance_split_dataset.items():  # z. B. "FINRA_unsafe"
        source, label = split_name.rsplit("_", 1)
        df = split.to_pandas()

        if label == "unsafe":
            df["prompt"] = df["rephrased instance"]
        else:
            df["prompt"] = df["original instance"]
        df["policy_source"] = source
        df["prompt_label"] = label
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def prepare_polyguard_finance(table):
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

    bad = out["prompt"].isna() | out["prompt"].astype(str).str.strip().isin(["", "Not applicable"])
    assert not bad.any(), f"{bad.sum()} leere/Platzhalter-Prompts gefunden"

    out = out.drop_duplicates("prompt").reset_index(drop=True)
    out["id"] = [f"polyguard_{i}" for i in range(len(out))]
    return out


# ----------------------------------------------------------------- ExpGuard

def load_dataset_splits_from_expguard(dataset_name=EXPGUARD_TEST_DS):
    test_split_dataset = load_dataset(dataset_name, "expguardtest", split="test")
    table = test_split_dataset.to_pandas()

    is_finance = table["domain"].astype(str).str.lower().str.strip() == "finance"
    return table[is_finance].copy()


def prepare_expguard_finance(table):
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
    out = out[out["prompt_label"].isin(["safe", "unsafe"])]
    out = out.drop_duplicates(["prompt", "response"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------- Main

def save_jsonl(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def main():
    ood_expg = prepare_expguard_finance(load_dataset_splits_from_expguard())
    save_jsonl(ood_expg, OUTPUT_FILE_EXPGUARD)

    ood_poly = prepare_polyguard_finance(load_datasets_from_polyguard())
    save_jsonl(ood_poly, OUTPUT_FILE_POLYGUARD)

    print(f"Gespeichert: {OUTPUT_FILE_EXPGUARD}  ({len(ood_expg)} Zeilen)")
    print(f"Gespeichert: {OUTPUT_FILE_POLYGUARD}  ({len(ood_poly)} Zeilen)")

if __name__ == "__main__":
    main()