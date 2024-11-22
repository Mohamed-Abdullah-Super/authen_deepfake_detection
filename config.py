import os

class Config:
    # Paths
    DATA_DIR = "/path/to/data"
    OUTPUT_DIR = "/path/to/output"
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    
    # Model Parameters
    NUM_CLASSES = 2
    INPUT_SIZE = 224  # Height and width of frames
    NUM_FRAMES = 32  # Number of frames per video
    
    # Training Parameters
    BATCH_SIZE = 16
    EPOCHS = 25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Augmentation Settings
    AUGMENTATION = {
        "use_color_jitter": True,
        "use_random_crop": True,
        "use_horizontal_flip": True
    }
    
    # Logging
    PRINT_FREQ = 10
