import torch
from torchvision.transforms import Normalize

def preprocess_frames(frames, num_frames=32):
    """
    Preprocesses frames for inference.
    Args:
        frames (np.ndarray): Raw frames (shape: [T, H, W, C]).
        num_frames (int): Number of frames to sample.
    Returns:
        torch.Tensor: Preprocessed frames (shape: [1, C, T, H, W]).
    """
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T, C, H, W)
    frames = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frames)
    frames = frames[:num_frames]
    frames = frames.unsqueeze(0)  # Add batch dimension
    return frames

def infer(model, video_path, num_frames=32, device='cuda'):
    """
    Performs inference on a single video.
    Args:
        model (nn.Module): Trained model.
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample.
        device (str): Device to run inference on.
    Returns:
        dict: Inference result containing logits and predicted label.
    """
    from data.dataset import DeepfakeDataset  # To use frame extraction
    dataset = DeepfakeDataset([video_path], [0], num_frames, training=False)
    frames = dataset[0]['frames']
    frames = preprocess_frames(frames.numpy(), num_frames=num_frames).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(frames)
        logits = outputs['logits']
        prediction = torch.argmax(logits, dim=1).item()
    
    return {
        'logits': logits.cpu().numpy(),
        'prediction': prediction
    }
