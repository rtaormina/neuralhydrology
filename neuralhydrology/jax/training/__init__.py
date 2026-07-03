from typing import Any

from neuralhydrology.jax.config import validate_jax_config
from neuralhydrology.utils.config import Config


def get_loss_obj(cfg: Config) -> Any:
    """Get a JAX loss object.

    Placeholder: the JAX loss implementations have not been added yet.
    """
    validate_jax_config(cfg)
    raise NotImplementedError("JAX loss creation is a placeholder and has not been implemented yet.")


def get_optimizer(cfg: Config) -> Any:
    """Get a JAX optimizer object.

    Placeholder: the JAX optimizer implementations have not been added yet.
    """
    validate_jax_config(cfg)
    raise NotImplementedError("JAX optimizer creation is a placeholder and has not been implemented yet.")
