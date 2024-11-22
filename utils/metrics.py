import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Computes accuracy, precision, recall, and F1-score.
    Args:
        predictions (torch.Tensor): Predicted labels (shape: [B]).
        labels (torch.Tensor): True labels (shape: [B]).
    Returns:
        dict: Dictionary containing the metrics.
    """
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='binary'),
        'recall': recall_score(labels, predictions, average='binary'),
        'f1_score': f1_score(labels, predictions, average='binary'),
    }

def evaluate(model, data_loader, device) -> tuple:
    """
    Evaluates the model on the validation set.
    Args:
        model (nn.Module): The deepfake detection model.
        data_loader (DataLoader): Validation DataLoader.
        device (torch.device): Device to run the model on.
    Returns:
        tuple: Validation loss and metrics.
    """
    model.eval()
    val_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(frames)
            logits = outputs['logits']
            loss = compute_loss(outputs, labels, frames)
            val_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    metrics = compute_metrics(torch.tensor(all_predictions), torch.tensor(all_labels))
    return val_loss / len(data_loader), metrics
