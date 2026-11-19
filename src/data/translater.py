import deepl
import os
import pandas as pd
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

def main():
    load_dotenv()
    client = deepl.DeepLClient(os.getenv("DEEPL_API_KEY"))

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_json(POLYGUARD, lines=True)
    df["prompt_before_translation"] = df["prompt"]
    df["prompt"] = translate_in_batches(df["prompt_before_translation"], client)

    df.to_json(
        os.path.join(OUT_DIR, "polyguard_finance_test_de.jsonl"),
        orient="records", lines=True, force_ascii=False,
    )


if __name__ == "__main__":
    main()




