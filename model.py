import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

#Model in die GPU laden
def load_model(model_id):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).to(device).eval()

    return processor, model

#Inputs werden so vorbereitet, dass Sie das Model verstehen würde -> <user>.. etc
def build_inputs(processor, model, prompt, thinking, policy):

    content = [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    
    inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                request_categories="/categories",
                enable_thinking=thinking,
                custom_policy=policy
            ).to(model.device)
    return inputs

#Die generierten Inputs werden dem Model zur Inferenz gegeben. Das generated gibt zwei Dimensionen aus -> durchs sampeln
def generate(model, inputs, **gen_kwargs):
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_logits=True,
            **gen_kwargs,
        )
    generated = out.sequences[:, input_len:]
    return out, generated 
