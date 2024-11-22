import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

class DeepfakeDataset(Dataset):
    def __init__(
        self,
        video_paths: list,
        labels: list,
        num_frames: int = 32,
        transform: Optional[A.Compose] = None,
        training: bool = True
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.training = training
        
        # Default augmentation pipeline
        self.transform = transform or A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(),
                A.ImageCompression(quality_lower=60, quality_upper=100),
            ], p=0.5),
            A.Normalize(),
        ])
        
    def __len__(self):
        return len(self.video_paths)
        
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess frames
        frames = self._load_frames(video_path)
        
        # Apply augmentations
        if self.transform and self.training:
            frames = [
                self.transform(image=frame)['image']
                for frame in frames
            ]
        
        # Stack frames
        frames = np.stack(frames)
        
        return {
            'frames': torch.from_numpy(frames).float(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
    def _load_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices for uniform sampling
        frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) < self.num_frames:
            # Handle videos shorter than expected
            frames.extend([frames[-1]] * (self.num_frames - len(frames)))
        
        return frames