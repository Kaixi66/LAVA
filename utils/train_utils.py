import torch
import logging

logger = logging.getLogger(__name__)


def count_parameters(model, model_name="Model"):
    """
    Compute and print the number of model parameters (in millions)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_m = total_params / 1e6
    trainable_m = trainable_params / 1e6

    logger.info(f"[{model_name}] Stats:")
    logger.info(f"  - Total Parameters: {total_m:.2f}M")
    logger.info(f"  - Trainable Parameters: {trainable_m:.2f}M")

    return total_m, trainable_m
