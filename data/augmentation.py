import random
import torchvision.transforms as transforms

def get_augmentations(training: bool = True):
    """
    Returns a set of transformations for data augmentation.
    Args:
        training (bool): Whether to return training augmentations or not.
    Returns:
        torchvision.transforms.Compose: Transformations.
    """
    if training:
        augmentations = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
    else:
        augmentations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    
    return augmentations

def apply_augmentation(frames, augmentations):
    """
    Applies augmentations to a batch of frames.
    Args:
        frames (torch.Tensor): Frames to augment (shape: [T, H, W, C]).
        augmentations (transforms.Compose): Augmentations to apply.
    Returns:
        torch.Tensor: Augmented frames (shape: [T, C, H, W]).
    """
    frames_augmented = [augmentations(frame) for frame in frames]
    return torch.stack(frames_augmented)
