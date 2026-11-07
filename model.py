import torch
import yaml
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

#Modell laden, auswechselbar über config.yaml
def load_model():
    MODEL_ID = config["model"]["id"]

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    
    return processor, model
