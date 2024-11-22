import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_frames(frames, title=None):
    """
    Visualizes a sequence of video frames.
    Args:
        frames (torch.Tensor or np.ndarray): Frames with shape (T, H, W, C).
        title (str, optional): Title for the visualization.
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    
    num_frames = frames.shape[0]
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
    
    for i in range(num_frames):
        axes[i].imshow(frames[i].astype(np.uint8))
        axes[i].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    plt.show()

def plot_metrics(metrics: dict, epochs: int):
    """
    Plots training/validation metrics.
    Args:
        metrics (dict): Dictionary with keys as metric names and values as lists.
        epochs (int): Number of epochs to plot.
    """
    plt.figure(figsize=(10, 5))
    for key, values in metrics.items():
        plt.plot(range(1, epochs + 1), values, label=key)
    
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('Training and Validation Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()
