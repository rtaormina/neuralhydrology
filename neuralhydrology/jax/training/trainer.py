from neuralhydrology.jax.config import validate_jax_config
from neuralhydrology.utils.config import Config


class JaxTrainer(object):
    """Placeholder trainer for the experimental JAX backend."""

    def __init__(self, cfg: Config):
        super(JaxTrainer, self).__init__()
        validate_jax_config(cfg)
        self.cfg = cfg

    def initialize_training(self):
        """Initialize JAX training.

        Placeholder: the actual JAX training implementation has not been added yet.
        """
        raise NotImplementedError("JAX training is a placeholder and has not been implemented yet.")

    def train_and_validate(self):
        """Train and validate a JAX model.

        Placeholder: the actual JAX training implementation has not been added yet.
        """
        raise NotImplementedError("JAX training is a placeholder and has not been implemented yet.")
