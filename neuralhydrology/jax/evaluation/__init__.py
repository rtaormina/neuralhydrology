from pathlib import Path

from neuralhydrology.jax.config import validate_jax_config
from neuralhydrology.jax.evaluation.tester import JaxRegressionTester
from neuralhydrology.utils.config import Config


def get_tester(cfg: Config, run_dir: Path, period: str, init_model: bool) -> JaxRegressionTester:
    """Get a JAX tester object."""
    validate_jax_config(cfg)
    if cfg.head.lower() == "regression":
        Tester = JaxRegressionTester
    else:
        raise NotImplementedError(f"No JAX evaluation method implemented for {cfg.head} head")

    return Tester(cfg=cfg, run_dir=run_dir, period=period, init_model=init_model)
