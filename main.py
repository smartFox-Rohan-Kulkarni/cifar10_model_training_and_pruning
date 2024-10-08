from tools.evaluate import evaluate_models
from tools.prune import prune_model
from tools.train import train_model
from utilites.logger import get_logger

logging = get_logger()


def main():
    logging.info("Starting model training")
    train_model()

    logging.info("\nStarting model pruning")
    best_pruning_ratio = prune_model()

    logging.info("\nEvaluating models")
    evaluate_models(best_pruning_ratio)


if __name__ == "__main__":
    main()
