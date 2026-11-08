from model import build_inputs, generate
from tokens import find_decision_pos, decision_probabilities, find_after, UNSAFE, THINK_END_PATTERN
from . import MethodResult, MethodError, Label
import re

def find_confidence_in_answer(answer):
      match = re.search(r"Confidence:\s*(\d+)", answer)
      if match:
        return int(match.group(1))
        
      return None

def is_confidence_valid(confidence):
        return 0 <= confidence <= 100

def proceed_with_verb(model, processor, prompt, thinking, policy, **gen_kwargs):
    inputs = build_inputs(processor=processor, model=model, prompt=prompt, thinking=thinking, policy=policy)
    out, generated = generate(model=model, inputs=inputs, **gen_kwargs)
    outputs = processor.batch_decode(generated, skip_special_tokens=True)
    token_list = generated[0].tolist()
    pos_decision_token = find_decision_pos(token_list=token_list, thinking_mode=thinking)

    if thinking is False:
        answer_text = outputs[0]
    else:
        start = find_after(token_list,THINK_END_PATTERN)
        if start is None:
            return MethodResult(
                        outputs=outputs,
                        error=MethodError.THINK_END_NOT_FOUND,
                         )   
        answer_tokens = token_list[start:]
        answer_text = processor.decode(answer_tokens, skip_special_tokens=True)

    if (pos_decision_token is None):
            return MethodResult(
                outputs=outputs,
                error=MethodError.DECISION_NOT_FOUND
            )   
            
    out_logits = out.logits[pos_decision_token][0]
    p_safe, p_unsafe= decision_probabilities(out_logits)


    verb_confidence = find_confidence_in_answer(answer_text)

    if verb_confidence is None:
         return MethodResult(
                outputs=outputs,
                error=MethodError.VERB_CONF_NOT_FOUND,
                metadata={
                                    "p_safe": p_safe,
                                    "p_unsafe": p_unsafe,
                                    "decision_position": pos_decision_token,
                                    }
         )

    if(is_confidence_valid(verb_confidence)):
         verb_confidence = verb_confidence/100
    else:
         return MethodResult(
                outputs=outputs,
                error=MethodError.VERB_CONF_IS_NOT_VALID,
                metadata={
                    "p_safe": p_safe,
                    "p_unsafe": p_unsafe,
                    "decision_position": pos_decision_token,
                    "raw_confidence": verb_confidence
                    }    
         )
    
    return MethodResult(
        outputs=outputs,
        prediction= Label.UNSAFE if token_list[pos_decision_token] == UNSAFE else Label.SAFE,
        confidence=verb_confidence,
        metadata={
            "p_safe": p_safe,
            "p_unsafe": p_unsafe,
            "decision_position": pos_decision_token,
            }    
        )