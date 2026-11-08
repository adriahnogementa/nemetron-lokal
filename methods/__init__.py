from dataclasses import dataclass, field
from enum import StrEnum

class Label(StrEnum):
    SAFE = "safe"
    UNSAFE = "unsafe"

class MethodError(StrEnum):
    DECISION_NOT_FOUND = "decision_not_found"
    THINK_END_NOT_FOUND = "think_end_not_found"
    ALL_SAMPLES_FAILED = "all_samples_failed"

@dataclass
class MethodResult:
    outputs: list[str]
    prediction: Label | None = None
    confidence: float | None = None
    error: MethodError | None = None
    metadata: dict = field(default_factory=dict)


