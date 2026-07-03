from dataclasses import dataclass
from typing import Any


@dataclass
class JaxTrainState:
    """Placeholder container for JAX training state."""

    params: Any
    optimizer_state: Any
    epoch: int
    rng: Any
