# Suppress TensorFlow startup logs (INFO & WARNING)
# -------------------------------------------------
# These logs often display environment info, library loading, and optimization hints.
# They are NOT required for running the model but may help with debugging hardware/driver issues.
# If you want to see these logs again for troubleshooting, delete or comment the lines below.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = remove INFO, 2 = remove INFO & WARNING, 3 = remove all logs
# -------------------------------------------------
import warnings
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomRotation, RandomAdjustSharpness, RandomHorizontalFlip, ColorJitter


import tensorflow as tf


from config import DATA_DIR, MODEL_CHECKPOINT

warnings.filterwarnings("ignore")


def load_and_preprocess_data(data_dir=DATA_DIR, model_checkpoint=MODEL_CHECKPOINT):
    # Load dataset
    MB_dataset = load_dataset("imagefolder", data_dir=data_dir)
    
    dataset = DatasetDict({
        "train": MB_dataset["train"],
        "validation": MB_dataset["validation"],
        "test": MB_dataset["test"]
    })

    # Label mappings
    labels = dataset["train"].features["label"].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    print("Label mappings:", label2id)

    # Image processor
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    image_mean, image_std = image_processor.image_mean, image_processor.image_std
    size = image_processor.size["height"]

    # Transforms
    normalize = Normalize(mean=image_mean, std=image_std)
    train_tf = Compose([
        Resize((size, size)),
        RandomRotation(15),
        RandomAdjustSharpness(2),
        RandomHorizontalFlip(0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ToTensor(),
        normalize
    ])
    val_tf = Compose([
        Resize((size, size)),
        ToTensor(),
        normalize
    ])

    def preprocess_train(examples):
        examples["pixel_values"] = [train_tf(img.convert("RGB")) for img in examples["image"]]
        return examples

    def preprocess_val(examples):
        examples["pixel_values"] = [val_tf(img.convert("RGB")) for img in examples["image"]]
        return examples

    dataset["train"].set_transform(preprocess_train)
    dataset["validation"].set_transform(preprocess_val)
    dataset["test"].set_transform(preprocess_val)

    return dataset, label2id, id2label, image_processor


if __name__ == "__main__":
    dataset, label2id, id2label, image_processor = load_and_preprocess_data()
    print("Dataset prepared successfully.")