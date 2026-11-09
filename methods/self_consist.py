from collections import Counter
from model import build_inputs, generate, continue_with
from tokens import find_decision_pos, decision_probabilities, find_after, UNSAFE, THINK_END_PATTERN
from . import MethodResult, MethodError, Label
import re



def proceed_with_self_cons(model, processor, prompt, thinking, policy, **gen_kwargs):
    inputs = build_inputs(processor=processor, model=model, prompt=prompt, thinking=thinking, policy=policy)
    out, generated = generate(model=model, inputs=inputs, **gen_kwargs)
    outputs = processor.batch_decode(generated, skip_special_tokens=True)

    samples = []

    for sample_index, sample in enumerate(generated):
        token_list = sample.tolist()
        pos_decision_token = find_decision_pos(token_list=token_list, thinking_mode=thinking)

        if pos_decision_token is None:
            samples.append({"error": MethodError.DECISION_NOT_FOUND,
                            "sample_index": sample_index})
            continue

        out_logits = out.logits[pos_decision_token][sample_index]
        p_safe, p_unsafe= decision_probabilities(out_logits)

        prediction = Label.UNSAFE if token_list[pos_decision_token] == UNSAFE else Label.SAFE

        samples.append({
            "sample_index": sample_index,
            "prediction": prediction,
            "p_safe": p_safe,
            "p_unsafe": p_unsafe,
            "decision_position": pos_decision_token
        })

    valid_samples = [sample for sample in samples if "error" not in sample]

    if not valid_samples:
        return MethodResult(outputs=outputs, 
                            error=MethodError.ALL_SAMPLES_FAILED, 
                            metadata=
                            {
                                "samples": samples
                                }
                            )

    counts = Counter(sample["prediction"] for sample in valid_samples)
    majority_prediction, majority_count = counts.most_common(1)[0]
    majority_confidence = majority_count / len(valid_samples)

    if majority_confidence == 0.5:
        majority_prediction = Label.UNSAFE
    
    p_unsafe_summed = sum(sample["p_unsafe"] for sample in valid_samples)
    p_unsafe_mean = p_unsafe_summed/len(valid_samples)
    soft_vote_confidence = max(p_unsafe_mean, 1-p_unsafe_mean)

    if p_unsafe_mean >= 0.5:
        soft_vote_prediction = Label.UNSAFE
    else:
        soft_vote_prediction = Label.SAFE

    weighted_sum_all_predictions = 0.0
    weighted_sum_unsafe_prediction = 0.0

    for sample in valid_samples:
        if sample["prediction"] == Label.SAFE:
            weighted_sum_all_predictions += sample["p_safe"]
        else:
            weighted_sum_all_predictions += sample["p_unsafe"]
            weighted_sum_unsafe_prediction += sample["p_unsafe"]

    confidence_unsafe_weighted_sum = weighted_sum_unsafe_prediction / weighted_sum_all_predictions

    if confidence_unsafe_weighted_sum >= 0.5:
        weighted_sum_confidence = confidence_unsafe_weighted_sum
        weighted_sum_prediction = Label.UNSAFE
    else:
        weighted_sum_confidence = 1 - confidence_unsafe_weighted_sum
        weighted_sum_prediction = Label.SAFE

    return MethodResult(
        outputs=outputs,
        metadata={
            "aggregations": {
                "majority": {
                    "prediction": majority_prediction,
                    "confidence": majority_confidence,
                },
                "avg_conf": {
                    "prediction": weighted_sum_prediction,
                    "confidence": weighted_sum_confidence,
                },
                "soft_vote": {
                    "prediction": soft_vote_prediction,
                    "confidence": soft_vote_confidence,
                },
            },
            "n_samples": len(samples),
            "n_valid": len(valid_samples),
            "n_safe": counts[Label.SAFE],
            "n_unsafe": counts[Label.UNSAFE],
            "samples": samples,
        },
    )