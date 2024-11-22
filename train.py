import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils.metrics import compute_metrics
from utils.visualization import plot_metrics
from models.losses import compute_loss

def train_one_epoch(model, data_loader, optimizer, device):
    """
    Train the model for one epoch.
    Args:
        model (torch.nn.Module): The model to train.
        data_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device for training ('cpu' or 'cuda').
    Returns:
        float: Average loss over the epoch.
    """
    model.train()
    total_loss = 0
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = compute_loss(outputs, labels, inputs)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    """
    Evaluate the model on a validation set.
    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for validation data.
        device (str): Device for evaluation ('cpu' or 'cuda').
    Returns:
        float: Accuracy score.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs['logits'], 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

def train_model(model, train_loader, val_loader, config):
    """
    Full training loop.
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        config (Config): Training configuration.
    """
    device = config.DEVICE
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    best_val_acc = 0.0
    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}]: Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{config.MODEL_SAVE_DIR}/best_model.pth")
            print("Saved best model!")
    
    print("Training complete.")

