import torch
import torch.nn.functional as F

def compute_loss(outputs: dict, labels: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """
    Computes the loss for deepfake detection.
    Args:
        outputs (dict): Dictionary containing 'logits' and optional other keys.
        labels (torch.Tensor): Ground truth labels.
        frames (torch.Tensor): Input frames (optional for certain loss functions).
    Returns:
        torch.Tensor: Computed loss.
    """
    logits = outputs['logits']
    classification_loss = F.cross_entropy(logits, labels)
    
    # Add other auxiliary loss terms if needed (e.g., regularization)
    total_loss = classification_loss  # Can include additional terms
    
    return total_loss
