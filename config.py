# config.py

# ================================
# General Project Settings
# ================================

# Dataset path
DATA_DIR = r"D:\Telegram Downloads\Graduation Project\Latest from ALL\Facial_Expression files\KDEF_Cropped_YOLO11n_Split + M_B_Split"

# Hugging Face model checkpoint for training
MODEL_CHECKPOINT = "motheecreator/vit-Facial-Expression-Recognition"

# Where to save training checkpoints
OUTPUT_DIR = r"D:\Telegram Downloads\Graduation Project\Latest from ALL\Facial_Expression files\vit-Facial-Expression-Recognition-KDEF-M_B-checkpoints"

# Best model directory after saving
BEST_MODEL_DIR = r"D:\Telegram Downloads\Graduation Project\Latest from ALL\Facial_Expression files\model\best_kdef_M-B_model"

# ================================
# Training Hyperparameters
# ================================
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 20
WARMUP_STEPS = 300
LOGGING_STEPS = 25
EVAL_STEPS = 100