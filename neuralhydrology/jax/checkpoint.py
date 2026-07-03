from pathlib import Path
from typing import Any, Dict


def save_checkpoint(state: Dict[str, Any], run_dir: Path, epoch: int):
    """Placeholder for saving JAX training state."""
    raise NotImplementedError("JAX checkpoint saving is a placeholder and has not been implemented yet.")


def load_checkpoint(run_dir: Path, epoch: int = None) -> Dict[str, Any]:
    """Placeholder for loading JAX training state."""
    raise NotImplementedError("JAX checkpoint loading is a placeholder and has not been implemented yet.")
