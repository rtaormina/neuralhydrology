from pathlib import Path
from typing import Union

from neuralhydrology.jax.config import validate_jax_config
from neuralhydrology.utils.config import Config


class JaxRegressionTester(object):
    """Placeholder tester for JAX regression models."""

    def __init__(self, cfg: Config, run_dir: Path, period: str = "test", init_model: bool = True):
        super(JaxRegressionTester, self).__init__()
        validate_jax_config(cfg)
        self.cfg = cfg
        self.run_dir = run_dir
        self.period = period
        self.init_model = init_model

    def evaluate(self,
                 epoch: int = None,
                 save_results: bool = True,
                 save_all_output: bool = False,
                 metrics: Union[list, dict] = None):
        """Evaluate a JAX regression model.

        Placeholder: the actual JAX evaluation implementation has not been added yet.
        """
        raise NotImplementedError("JAX evaluation is a placeholder and has not been implemented yet.")
