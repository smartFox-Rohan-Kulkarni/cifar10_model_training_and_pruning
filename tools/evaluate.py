from model.adaptive_ensemble import AdaptiveEnsemble
import torch
import numpy as np

from config import DEVICE, MODEL_PATH, NC, ORIGINAL_MODEL_NAME, PRUNED_MODEL_NAME
import os

from utilites.dataloader import get_data_loaders

from utilites.logger import get_logger

logging = get_logger()

def evaluate_models():
    _, _, test_loader = get_data_loaders()

    original_model = AdaptiveEnsemble(num_classes=NC, num_clusters=NC).to(DEVICE)
    original_model.load_state_dict(
        torch.load(os.path.join(MODEL_PATH, ORIGINAL_MODEL_NAME))
    )

    pruned_model = AdaptiveEnsemble(num_classes=NC, num_clusters=NC).to(DEVICE)
    pruned_model.load_state_dict(
        torch.load(os.path.join(MODEL_PATH, PRUNED_MODEL_NAME))
    )

    logging.info("Evaluating original model:")
    original_accuracy, original_p50, original_p90 = evaluate_model(
        original_model, test_loader
    )

    logging.info("\nEvaluating pruned model:")
    pruned_accuracy, pruned_p50, pruned_p90 = evaluate_model(pruned_model, test_loader)

    logging.info("\nResults summary:")
    logging.info(
        f"Original model - Accuracy: {original_accuracy:.2f}%, P50: {original_p50:.4f}, P90: {original_p90:.4f}"
    )
    logging.info(
        f"Pruned model - Accuracy: {pruned_accuracy:.2f}%, P50: {pruned_p50:.4f}, P90: {pruned_p90:.4f}"
    )


def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    latencies = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            outputs = model(inputs)
            end_time.record()

            torch.cuda.synchronize()
            latency = start_time.elapsed_time(end_time)
            latencies.append(latency)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)

    logging.info(f"Accuracy: {accuracy:.2f}%")
    logging.info(f"P50 latency: {p50:.4f} ms")
    logging.info(f"P90 latency: {p90:.4f} ms")

    return accuracy, p50, p90
