import yaml
from model import load_model
from data import load_data
from methods.msp import proceed_with_msp
from methods.verb import proceed_with_verb
from pathlib import Path
from pprint import pprint


def load_policy(path):
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def main():
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)


    processor, model = load_model(config["model"]["id"])
    method_cfg = config["methods"]["verb"]

    policy = load_policy(method_cfg.get("custom_policy"))
    thinking = method_cfg["enable_thinking"]
    gen_kwargs = method_cfg["generation"]

    for prompt in load_data(): 

        result = proceed_with_verb(model, processor, prompt=prompt, thinking=thinking, policy=policy, **gen_kwargs)
        pprint(result)



if __name__ == "__main__":
   main()
   



