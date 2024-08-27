import torch
import torch.nn.utils.prune as prune

from model.adaptive_ensemble import AdaptiveEnsemble
from utilites.logger import get_logger

logging = get_logger()

import os

from train import evaluate_model

from config import (DEVICE, MODEL_PATH, NC, ORIGINAL_MODEL_NAME,
                    PRUNED_MODEL_NAME, PRUNING_RATIOS)
from utilites.dataloader import get_data_loaders


def prune_model():
    _, val_loader, _ = get_data_loaders()
    model = AdaptiveEnsemble(num_classes=NC, num_clusters=NC).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, ORIGINAL_MODEL_NAME)))

    original_accuracy = evaluate_model(model, val_loader)
    logging.info(f"Original model accuracy: {original_accuracy:.2f}%")

    best_pruned_model = None
    best_pruning_ratio = None
    best_pruned_accuracy = 0

    for ratio in PRUNING_RATIOS:
        pruned_model = prune_layers(model, ratio)
        pruned_accuracy = evaluate_model(pruned_model, val_loader)
        logging.info(f"Pruned model accuracy (ratio {ratio}): {pruned_accuracy:.2f}%")

        if (
            pruned_accuracy > best_pruned_accuracy
            and pruned_accuracy >= original_accuracy - 1
        ):
            best_pruned_model = pruned_model
            best_pruning_ratio = ratio
            best_pruned_accuracy = pruned_accuracy

    if best_pruned_model is not None:
        logging.info(f"Best pruning ratio: {best_pruning_ratio}")
        logging.info(f"Best pruned model accuracy: {best_pruned_accuracy:.2f}%")
        save_model(best_pruned_model, str(PRUNED_MODEL_NAME))
    else:
        logging.info("No pruned model met the accuracy criteria.")


def prune_layers(model, ratio):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=ratio)
            prune.remove(module, "weight")
    return model


def save_model(model, filename):
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, filename))
