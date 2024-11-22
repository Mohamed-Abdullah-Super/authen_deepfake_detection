import cv2
import numpy as np

def extract_frames(video_path: str, num_frames: int = 32) -> np.ndarray:
    """
    Extracts evenly spaced frames from a video.
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract.
    Returns:
        np.ndarray: Extracted frames (shape: [T, H, W, C]).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int)

    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if idx in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

def normalize_frames(frames: np.ndarray) -> np.ndarray:
    """
    Normalizes video frames to [0, 1] range.
    Args:
        frames (np.ndarray): Raw frames (shape: [T, H, W, C]).
    Returns:
        np.ndarray: Normalized frames.
    """
    return frames / 255.0

def preprocess_video(video_path: str, num_frames: int = 32) -> np.ndarray:
    """
    Combines frame extraction and normalization for preprocessing.
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to preprocess.
    Returns:
        np.ndarray: Preprocessed frames (shape: [T, H, W, C]).
    """
    frames = extract_frames(video_path, num_frames)
    frames = normalize_frames(frames)
    return frames
