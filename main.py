import torch
import yaml
from model import load_model, build_inputs, generate
from data import load_data
from pathlib import Path

def load_policy(path):
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def main():
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prompt = load_data()[0]

    processor, model = load_model(config["model"]["id"])

    for method_name, method_cfg in config["methods"].items():
        policy = load_policy(method_cfg.get("custom_policy"))
        thinking = method_cfg["enable_thinking"]
        gen_kwargs = method_cfg["generation"]

        inputs = build_inputs(processor, model, prompt, thinking=thinking, policy=policy)
        out, generated = generate(model, inputs, **gen_kwargs)
        processor.decode(generated[0], skip_special_tokens=True)




if __name__ == "__main__":
   main()
   



