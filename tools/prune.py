import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from model.adaptive_ensemble import AdaptiveEnsemble
from utilites.logger import get_logger
from pathlib import Path

logging = get_logger()

import os

from tools.train import evaluate

from config import (
    DEVICE,
    MODEL_PATH,
    NC,
    ORIGINAL_MODEL_NAME,
    PRUNED_MODEL_NAME,
    PRUNING_RATIOS,
)
from utilites.dataloader import get_data_loaders


def prune_model():
    _, val_loader, _ = get_data_loaders()
    model = AdaptiveEnsemble(num_classes=NC, num_clusters=NC).to(DEVICE)
    model.load_state_dict(torch.load(ORIGINAL_MODEL_NAME))
    classification_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        _, original_accuracy = evaluate(
            model, val_loader, classification_criterion, DEVICE
        )
    logging.info(f"Original model accuracy: {original_accuracy:.2f}%")
    original_params = get_model_size(model)
    original_file_size = save_model(model, ORIGINAL_MODEL_NAME)
    logging.info(f"Original model parameters: {original_params}")
    logging.info(f"Original model file size: {original_file_size / 1024:.2f} KB")

    best_pruned_model = None
    best_pruning_ratio = None
    best_pruned_accuracy = 0

    for ratio in PRUNING_RATIOS:
        pruned_model = prune_layers(model, ratio)
        _, pruned_accuracy = evaluate(
            pruned_model, val_loader, classification_criterion, DEVICE
        )
        logging.info(f"Pruned model accuracy (ratio {ratio}): {pruned_accuracy:.2f}%")
        ACCURACY_TOLERANCE = 70.0  # 1.0% tolerance
        if pruned_accuracy > best_pruned_accuracy and pruned_accuracy >= (
            original_accuracy - ACCURACY_TOLERANCE
        ):
            best_pruned_model = pruned_model
            best_pruning_ratio = ratio
            best_pruned_accuracy = pruned_accuracy

    if best_pruned_model is not None:
        logging.info(f"Best pruning ratio: {best_pruning_ratio}")
        logging.info(f"Best pruned model accuracy: {best_pruned_accuracy:.2f}%")
        pruned_params = get_model_size(best_pruned_model)
        pruned_file_size = save_model(
            best_pruned_model, str(best_pruning_ratio) + "_" + PRUNED_MODEL_NAME
        )
        logging.info(f"Pruned model parameters: {pruned_params}")
        logging.info.infoint(
            f"Pruned model file size: {pruned_file_size / 1024:.2f} KB"
        )
        logging.info(
            f"Parameter reduction: {(original_params - pruned_params) / original_params * 100:.2f}%"
        )
        logging.info(
            f"File size reduction: {(original_file_size - pruned_file_size) / original_file_size * 100:.2f}%"
        )
    else:
        logging.info("No pruned model met the accuracy criteria.")
    return best_pruning_ratio


def prune_layers(model, ratio):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=ratio)
            prune.remove(module, "weight")
    return model


def save_model(model, filename):
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    filepath = os.path.join(MODEL_PATH, filename)
    torch.save(model.state_dict(), filepath)
    return os.path.getsize(filepath)


def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
