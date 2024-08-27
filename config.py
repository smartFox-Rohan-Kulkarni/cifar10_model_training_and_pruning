import torch

RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NC = 10
DATA_PATH = "./data"
MODEL_PATH = "./models"
ORIGINAL_MODEL_NAME = "original_model.pth"
PRUNED_MODEL_NAME = "pruned_model.pth"

PRUNING_RATIOS = [0.5, 0.7]
