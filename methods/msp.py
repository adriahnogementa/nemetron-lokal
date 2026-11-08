from model import build_inputs, generate
from tokens import find_decision_pos, decision_probabilities, UNSAFE
from . import MethodResult, MethodError, Label


def proceed_with_msp(model, processor, prompt, thinking, policy, **gen_kwargs):
    inputs = build_inputs(processor=processor, model=model, prompt=prompt, thinking=thinking, policy=policy)
    out, generated = generate(model=model, inputs=inputs, **gen_kwargs)
    outputs = processor.batch_decode(generated, skip_special_tokens=True)
    token_list = generated[0].tolist()
    pos_decision_token = find_decision_pos(token_list=token_list, thinking_mode=thinking)


    if (pos_decision_token is None):
        return MethodResult(
            outputs=outputs,
            error=MethodError.DECISION_NOT_FOUND
        )   
        
    out_logits = out.logits[pos_decision_token][0]
    p_safe, p_unsafe= decision_probabilities(out_logits)

    return MethodResult(
        outputs=outputs,
        prediction= Label.UNSAFE if token_list[pos_decision_token] == UNSAFE else Label.SAFE,
        confidence=max(p_safe, p_unsafe),
        metadata={
            "p_safe": p_safe,
            "p_unsafe": p_unsafe,
            "decision_position": pos_decision_token,
            }    
        )
