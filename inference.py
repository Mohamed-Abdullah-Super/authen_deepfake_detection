import argparse
import os
import torch
from utils.inference import classify_videos
from models.model import DeepfakeDetectionModel  # Replace with your actual model name
from data.preprocessing import preprocess_video
from config import Config

def load_model(model_path, device):
    """
    Load the trained model.
    Args:
        model_path (str): Path to the saved model weights.
        device (str): Device to load the model onto ('cpu' or 'cuda').
    Returns:
        torch.nn.Module: Loaded model.
    """
    model = DeepfakeDetectionModel(num_classes=Config.NUM_CLASSES)  # Adjust as per your model architecture
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main(args):
    """
    Perform inference on input videos.
    Args:
        args: Command-line arguments.
    """
    device = Config.DEVICE
    model = load_model(args.model_path, device)

    # Get list of video files to process
    video_paths = [args.input] if os.path.isfile(args.input) else [
        os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(('.mp4', '.avi'))
    ]
    print(f"Found {len(video_paths)} video(s) for inference.")

    # Perform inference
    predictions = classify_videos(
        model=model,
        video_paths=video_paths,
        preprocess_fn=lambda video_path, num_frames: preprocess_video(video_path, num_frames, Config.INPUT_SIZE),
        device=device,
        num_frames=Config.NUM_FRAMES
    )

    # Save or display results
    for video, probs in predictions.items():
        print(f"Video: {video}, Predictions: {probs}")

    if args.output:
        with open(args.output, "w") as f:
            for video, probs in predictions.items():
                f.write(f"{video},{probs.tolist()}\n")
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input video file or directory containing videos.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model weights.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save prediction results as a CSV file.")
    args = parser.parse_args()

    main(args)
