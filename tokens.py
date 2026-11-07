import torch

#Gilt nur für Nemetron, falls andere Guardrailmodelle ausprobiert werden sollen muss das angepasst werden
USER = 2887
SAFETY = 16661
COLON = 236787
THINK_END_OPEN = 954     # '</'
THINK = 36345            # 'think'
THINK_END_CLOSE = 236813 # '>'
SAFE = 6338              # ' safe'
UNSAFE = 25793           # ' unsafe'

THINK_END_PATTERN = [THINK_END_OPEN, THINK, THINK_END_CLOSE]
USER_SAFETY_PATTERN = [USER, SAFETY, COLON]

#Sucht nach einem bestimmten Pattern, was in der Ausgabe vorkommen muss und gibt die Position
def find_after(token_list, pattern, start=0):
    for i in range(start ,len(token_list) - len(pattern) + 1):
        if token_list[i:i + len(pattern)] == pattern:
            return i + len(pattern)
    return None

#Für die Tokens Safe/Unsafe wird an dem Logit die W'keiten ausgerechnet
def decision_probabilities(logits):
    two_logits = logits[[SAFE, UNSAFE]].float()
    p_safe, p_unsafe = torch.softmax(two_logits, dim=-1).tolist()
    return p_safe, p_unsafe

#Sucht nach dem User Safety Pattern die Position
def find_decision_pos(token_list, thinking_mode):
    start = 0

    if thinking_mode:
        start = find_after(token_list,THINK_END_PATTERN)
        if start is None:
            return None
        
    found_token_pos = find_after(token_list, USER_SAFETY_PATTERN, start)

    return found_token_pos

