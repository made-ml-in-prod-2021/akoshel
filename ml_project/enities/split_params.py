from dataclasses import dataclass

@dataclass
class SplitParams:
    validation_size: float = 0.2
    random_state: int = 17