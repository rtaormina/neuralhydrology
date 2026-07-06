"""Tests for the experimental JAX backend routing placeholders."""
import pytest

from neuralhydrology.jax.config import validate_jax_config
from neuralhydrology.jax.training.trainer import JaxTrainer
from neuralhydrology.utils.config import Config


def _get_supported_jax_config() -> Config:
    return Config({
        "backend": "jax",
        "model": "cudalstm",
        "head": "regression",
        "loss": "mse",
        "optimizer": "adam",
        "use_frequencies": ["1D"],
        "dev_mode": True
    })


def test_backend_defaults_to_torch():
    cfg = Config({"dev_mode": True})

    assert cfg.backend == "torch"


def test_backend_accepts_jax_case_insensitive():
    cfg = Config({"backend": "JAX", "dev_mode": True})

    assert cfg.backend == "jax"


def test_backend_rejects_unknown_backend():
    cfg = Config({"backend": "numpy", "dev_mode": True})

    with pytest.raises(ValueError, match="'backend' must be either"):
        _ = cfg.backend


def test_jax_trainer_placeholder_for_supported_initial_config():
    trainer = JaxTrainer(_get_supported_jax_config())

    with pytest.raises(NotImplementedError, match="JAX training is a placeholder"):
        trainer.initialize_training()


def test_jax_config_rejects_unsupported_model():
    cfg = _get_supported_jax_config()
    cfg.update_config({"model": "mtslstm"}, dev_mode=True)

    with pytest.raises(NotImplementedError, match="model 'mtslstm' is not supported"):
        validate_jax_config(cfg)
