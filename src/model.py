import torch
import logging
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

logger = logging.getLogger(__name__)

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
                #request_categories="/categories",
                enable_thinking=thinking,
                custom_policy=policy
            ).to(model.device)

    logger.debug("Prompt:\n%s", processor.decode(inputs["input_ids"][0]))

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

#Eine Liste mit Tokens wird an das Model zurückgegeben und mit einem Suffix ergänzt. Danach wird das Model die Ausgabe ergänzen
#Wird für verbalized benötigt, weil aus Erfahrung das Model (Nemetron 3.5 CS) die Custom Policy ignoriert.
def continue_with(processor, model, inputs, generated_tokens, suffix, max_new_tokens=4):
    suffix_ids = processor.tokenizer.encode(suffix, add_special_tokens=False)

    prefix = torch.tensor(
        [inputs["input_ids"][0].tolist() + generated_tokens + suffix_ids],
        device=model.device,
    )
    attention_mask = torch.ones_like(prefix)

    with torch.inference_mode():
        out = model.generate(
            input_ids=prefix,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = out[0][prefix.shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True)