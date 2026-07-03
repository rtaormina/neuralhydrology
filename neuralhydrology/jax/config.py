from neuralhydrology.utils.config import Config


SUPPORTED_JAX_MODELS = ["cudalstm"]
SUPPORTED_JAX_HEADS = ["regression"]
SUPPORTED_JAX_LOSSES = ["mse", "nse", "rmse"]
SUPPORTED_JAX_OPTIMIZERS = ["adam", "adamw"]


def validate_jax_config(cfg: Config):
    """Validate the initial feature subset planned for the JAX backend."""
    errors = []

    if cfg.model.lower() not in SUPPORTED_JAX_MODELS:
        errors.append(f"model '{cfg.model}' is not supported")

    if cfg.head.lower() not in SUPPORTED_JAX_HEADS:
        errors.append(f"head '{cfg.head}' is not supported")

    if cfg.loss.lower() not in SUPPORTED_JAX_LOSSES:
        errors.append(f"loss '{cfg.loss}' is not supported")

    if cfg.optimizer.lower() not in SUPPORTED_JAX_OPTIMIZERS:
        errors.append(f"optimizer '{cfg.optimizer}' is not supported")

    if len(cfg.use_frequencies) > 1:
        errors.append("multi-frequency runs are not supported")

    if cfg.mc_dropout:
        errors.append("MC dropout is not supported")

    if cfg.regularization:
        errors.append("regularization terms are not supported")

    if cfg.autoregressive_inputs:
        errors.append("autoregressive inputs are not supported")

    if cfg.mass_inputs:
        errors.append("mass inputs are not supported")

    if cfg.nan_handling_method is not None:
        errors.append("nan handling methods are not supported")

    if cfg.forecast_inputs or cfg.hindcast_inputs:
        errors.append("forecast and hindcast inputs are not supported")

    if errors:
        raise NotImplementedError("The experimental JAX backend does not yet support: " + "; ".join(errors) + ".")
