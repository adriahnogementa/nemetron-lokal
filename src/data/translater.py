import deepl
import os
import pandas as pd
import argparse
import sys
import numpy as np
from dotenv import load_dotenv

EXPGUARD = "../../data_ood/expguard_finance_test.jsonl"
POLYGUARD = "../data_ood/polyguard_finance_test.jsonl"
OUT_DIR = "../data_ood_translated/"

def translate_in_batches(prompts, client, target_lang="DE" , batch_size=50):
    texts = prompts.astype(str).to_list()
    translated = []

    for i in range(0,len(texts), batch_size):
        batch = texts[i:i+batch_size]
        result = client.translate_text(batch, target_lang=target_lang)
        translated.extend(r.text for r in result)

    return pd.Series(translated, index=prompts.index)

def reduce_dataset(dataset):
    N = 1000
    SEED = 42

    pro_label = N // 2
    anteile = dataset[dataset["prompt_label"] == "unsafe"]["policy_source"].value_counts(normalize=True)
    roh = anteile * pro_label
    quoten = np.floor(roh).astype(int)

    teile = []
    for (quelle, label), g in dataset.groupby(["policy_source", "prompt_label"]):
        ein_pro_regel = g.groupby("scenario").sample(n=1, random_state=SEED)
        teile.append(ein_pro_regel.sample(n=quoten[quelle], random_state=SEED))

    return pd.concat(teile).sample(frac=1, random_state=SEED).reset_index(drop=True)

def save(df, name):
    df.to_json(
                os.path.join(OUT_DIR, name),
                orient="records", lines=True, force_ascii=False,
            )

def main(dateset):
    load_dotenv()
    client = deepl.DeepLClient(os.getenv("DEEPL_API_KEY"))

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_json(dateset, lines=True)

    if dataset is POLYGUARD:
        df = reduce_dataset(df)

    df["prompt_before_translation"] = df["prompt"]
    df["prompt"] = translate_in_batches(df["prompt_before_translation"], client)

    if dataset is POLYGUARD:
        save(df, "polyguard_finance_test_de.jsonl")
    else:
        save(df, "expguard_finance_test_de.jsonl")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_ood", help="Welcher OOD Dataset? polyguard oder expguard")
    args = parser.parse_args()
    dataset_name = args.dataset_ood

    if dataset_name.upper() == "POLYGUARD":
        dataset = POLYGUARD
    elif dataset_name.upper() == "EXPGUARD":
        dataset = EXPGUARD
    else:
        print("Ungültiges Dataset.")
        sys.exit(1)

    main(dataset)