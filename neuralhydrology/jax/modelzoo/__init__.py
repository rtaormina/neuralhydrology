from typing import Any

from neuralhydrology.jax.config import validate_jax_config
from neuralhydrology.utils.config import Config


def get_model(cfg: Config) -> Any:
    """Get a JAX model object.

    Placeholder: the JAX model implementations have not been added yet.
    """
    validate_jax_config(cfg)
    raise NotImplementedError("JAX model creation is a placeholder and has not been implemented yet.")
