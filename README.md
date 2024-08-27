# CIFAR-10 Model Training and Pruning

This project implements a Convolutional Neural Network (CNN) for the CIFAR-10 dataset, including model training, pruning, and evaluation.

## Requirements

- Python 3.10+
- PyTorch
- torchvision 0.10.0
- numpy 1.21.0
- matplotlib 3.4.2
- torchviz 0.0.2

Install the required packages using: pip install -r extras/requirements.txt

## Project Structure

- `config.py`: Contains configuration parameters for the project.
- `dataloader.py`: Handles data loading and preprocessing.
- `model dir`: Defines the CNN architecture.
- `train.py`: Implements the model training process.
- `prune.py`: Implements model pruning techniques.
- `evaluate.py`: Evaluates the original and pruned models.
- `main.py`: Main script to run the entire pipeline.

## Usage

To run the entire pipeline (training, pruning, and evaluation), execute: python main.py

## Results

The results, including model accuracies, pruning ratios, and latency metrics, will be displayed in the console output.

## Saved Models

- The original trained model will be saved as `original_model.pth`.
- The pruned model will be saved as `pruned_model.pth`.

Both models will be stored in the `./models` directory.