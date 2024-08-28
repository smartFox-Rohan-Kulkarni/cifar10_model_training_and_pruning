# pip install torch torchvision

# pip install tqdm torchviz

# pip freeze | grep tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from tqdm import tqdm
import logging
import torchviz
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch.jit as jit


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Define the EfficientNet-based feature extractor
class EfficientNetExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetExtractor, self).__init__()
        self.efficient_net = efficientnet_b0(pretrained=True)

        # Remove the last few layers to adapt to smaller input size
        self.feature_extractor = nn.Sequential(
            self.efficient_net.features[0],  # Conv2d
            self.efficient_net.features[1],  # BatchNorm2d
            self.efficient_net.features[2],  # SiLU
            self.efficient_net.features[3],  # MBConv block
            self.efficient_net.features[4],  # MBConv block
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


# Define the Contrastive Clustering Loss
class ContrastiveClusteringLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveClusteringLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        contrast_feature = torch.cat(
            [features, features], dim=0
        )  # Duplicate features for contrastive learning
        anchor_feature = contrast_feature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask to remove self-comparisons
        mask = torch.eye(batch_size * 2, device=features.device)
        exp_logits = torch.exp(logits) * (1 - mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean of log-likelihood over positive samples
        loss = -torch.sum(mask * log_prob) / batch_size

        return loss


# Define the adaptive ensemble model with clustering head
class AdaptiveEnsemble(nn.Module):
    def __init__(self, num_classes=10, num_clusters=10):
        super(AdaptiveEnsemble, self).__init__()
        self.feature_extractor1 = EfficientNetExtractor()
        self.feature_extractor2 = EfficientNetExtractor()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Adjust the combined_features size based on the output of EfficientNetExtractor
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = self.feature_extractor1(dummy_input)
            combined_features = dummy_output.shape[1] * 2

        self.fc = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        self.clustering_head = nn.Linear(combined_features, num_clusters)

    def forward(self, x):
        features1 = self.feature_extractor1(x)
        features2 = self.feature_extractor2(x)

        # Apply adaptive pooling
        features1 = self.adaptive_pool(features1).flatten(1)
        features2 = self.adaptive_pool(features2).flatten(1)

        # Combine features from both feature extractors
        combined_features = torch.cat([features1, features2], dim=1)
        classification_output = self.fc(combined_features)
        clustering_output = self.clustering_head(combined_features)

        return classification_output, clustering_output


batch_size = 64
dataloader_workers = 2
train_val_split_factor = 0.8  # 80%
nc = 10
_lr = 0.001
train_epochs = 30
pretrain_epochs = 10


# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdaptiveEnsemble(num_classes=nc, num_clusters=nc).to(device)
classification_criterion = nn.CrossEntropyLoss()
contrastive_criterion = ContrastiveClusteringLoss()
optimizer = optim.Adam(model.parameters(), lr=_lr)

# Data loading and augmentation
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter()], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# Load the full training set
full_trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)

# Split the full training set into train and validation sets
train_size = int(train_val_split_factor * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = random_split(full_trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers
)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=dataloader_workers
)


# # Training and testing functions
def pretrain(model, dataloader, contrastive_criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Pretraining", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        _, clustering_output = model(inputs)
        loss = contrastive_criterion(clustering_output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return running_loss / len(dataloader.dataset)


def train(model, dataloader, classification_criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs, _ = model(inputs)
        loss = classification_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, classification_criterion, device):
    model.eval()
    correct = 0
    total = 0
    eval_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = classification_criterion(outputs, labels)
            eval_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    accuracy = 100 * correct / total
    return eval_loss / len(dataloader.dataset), accuracy


# Generate model diagram
def generate_model_diagram(model):
    x = torch.randn(1, 3, 32, 32).to(device)  # Use 32x32 input size
    try:
        y = model(x)
        dot = torchviz.make_dot(y[0], params=dict(model.named_parameters()))
        dot.render("model_diagram.png", format="png")
        logging.info("Model diagram generated and saved as 'model_diagram.png'")
    except Exception as e:
        logging.error(f"Failed to generate model diagram: {str(e)}")


# Generate the model diagram
generate_model_diagram(model)
# img = mpimg.imread("model_diagram.png")

# plt.imshow(img)
# plt.show()

# Pretraining the model with contrastive loss
for epoch in range(pretrain_epochs):
    pretrain_loss = pretrain(
        model, trainloader, contrastive_criterion, optimizer, device
    )
    print(
        f"Pretrain Epoch {epoch+1}/{pretrain_epochs}, Pretrain Loss: {pretrain_loss:.4f}"
    )

# Fine-tuning the model with classification loss
best_val_accuracy = 0
for epoch in range(train_epochs):
    logging.info(f"epoch: {epoch}")
    train_loss = train(model, trainloader, classification_criterion, optimizer, device)
    val_loss, val_accuracy = evaluate(
        model, valloader, classification_criterion, device
    )

    print(
        f"Finetune Epoch {epoch+1}/{train_epochs}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Val Loss: {val_loss:.4f}, "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")
        print(
            f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%"
        )

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load("best_model.pth"))
test_loss, test_accuracy = evaluate(model, testloader, classification_criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "jit_model.pt")

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdaptiveEnsemble(num_classes=nc, num_clusters=nc).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()


def get_prediction(image):
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))[0]
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]


# Sample predictions
num_samples = 10
dataiter = iter(testloader)
images, labels = next(dataiter)

logging.info("Sample predictions:")
for i in range(min(num_samples, len(images))):
    image, label = images[i], labels[i]
    prediction = get_prediction(image)
    logging.info(f"Sample {i+1}: True: {classes[label]}, Predicted: {prediction}")

# Evaluate on the entire test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
logging.info(f"Accuracy on the test set: {accuracy:.2f}%")

# Confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)[0]
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_predictions)
logging.info("Confusion Matrix:")
logging.info(np.array2string(cm, separator=", "))

# Per-class accuracy
class_correct = list(0.0 for i in range(nc))
class_total = list(0.0 for i in range(nc))
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)[0]
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(nc):
    class_accuracy = 100 * class_correct[i] / class_total[i]
    logging.info(f"Accuracy of {classes[i]}: {class_accuracy:.2f}%")


batch_size = 64
dataloader_workers = 2
train_val_split_factor = 0.8  # 80%
nc = 10
_lr = 0.001
train_epochs = 30
pretrain_epochs = 10
PRUNING_RATIOS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.989, 0.99]
PRUNED_MODEL_NAME = "pruned_model.pth"
# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdaptiveEnsemble(num_classes=nc, num_clusters=nc).to(device)
classification_criterion = nn.CrossEntropyLoss()
contrastive_criterion = ContrastiveClusteringLoss()
optimizer = optim.Adam(model.parameters(), lr=_lr)

import torch
import os
import torch.nn.utils.prune as prune


def prune_model():
    val_loader = valloader
    model = AdaptiveEnsemble(num_classes=nc, num_clusters=nc).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    classification_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        _, original_accuracy = evaluate(
            model, val_loader, classification_criterion, device
        )
    logging.info(f"Original model accuracy: {original_accuracy:.2f}%")
    original_params = get_model_size(model)
    original_file_size = save_model(model, "best_model.pth")
    logging.info(f"Original model parameters: {original_params}")
    logging.info(f"Original model file size: {original_file_size / 1024:.2f} KB")

    best_pruned_model = None
    best_pruning_ratio = None
    best_pruned_accuracy = 0

    for ratio in PRUNING_RATIOS:
        pruned_model = prune_layers(model, ratio)
        _, pruned_accuracy = evaluate(
            pruned_model, val_loader, classification_criterion, device
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
        print(f"Best pruning ratio: {best_pruning_ratio}")
        print(f"Best pruned model accuracy: {best_pruned_accuracy:.2f}%")
        pruned_params = get_model_size(best_pruned_model)
        pruned_file_size = save_model(
            best_pruned_model, str(best_pruning_ratio) + "_" + PRUNED_MODEL_NAME
        )
        print(f"Pruned model parameters: {pruned_params}")
        print(f"Pruned model file size: {pruned_file_size / 1024:.2f} KB")
        print(
            f"Parameter reduction: {(original_params - pruned_params) / original_params * 100:.2f}%"
        )
        logging.info(
            f"File size reduction: {(original_file_size - pruned_file_size) / original_file_size * 100:.2f}%"
        )
    else:
        print("No pruned model met the accuracy criteria.")


def prune_layers(model, ratio):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=ratio)
            prune.remove(module, "weight")
    return model


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    return os.path.getsize(filename)


def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


prune_model()
