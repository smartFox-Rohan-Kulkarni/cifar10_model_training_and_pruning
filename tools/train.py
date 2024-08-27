from model.adaptive_ensemble import AdaptiveEnsemble
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import DEVICE, LEARNING_RATE, NC, NUM_EPOCHS, MODEL_PATH, ORIGINAL_MODEL_NAME
import os

from utilites.dataloader import get_data_loaders
from utilites.logger import get_logger

logging = get_logger()

def evaluate_model(model, dataloader, classification_criterion, device):
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


def train_tool(model, dataloader, classification_criterion, optimizer, device):
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

def train_model():
    
    model = AdaptiveEnsemble(num_classes=NC, num_clusters=NC).to(DEVICE)
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainloader, valloader, _ = get_data_loaders()
    best_val_accuracy = 0
    for epoch in range(NUM_EPOCHS):
        logging.info(f"epoch: {epoch}")
        train_loss = train_tool(model, trainloader, classification_criterion, optimizer, DEVICE)
        val_loss, val_accuracy = evaluate(model, valloader, classification_criterion, DEVICE)
        
        logging.info(
                    f'Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}, '
                    f'Val Accuracy: {val_accuracy:.2f}%')
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, ORIGINAL_MODEL_NAME)
            logging.info(f'New best model saved with validation accuracy: {best_val_accuracy:.2f}%')


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

def save_model(model, filename):
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, filename))
