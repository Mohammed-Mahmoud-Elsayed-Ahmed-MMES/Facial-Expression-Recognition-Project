import numpy as np
import matplotlib
# Use non-interactive backend to suppress Qt warnings
matplotlib.use('Agg')  # Set before importing pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from transformers import AutoModelForImageClassification, AutoImageProcessor, Trainer
import torch
from PIL import Image
import io
from datasets import Dataset

from data_preprocessing import load_and_preprocess_data
from config import BEST_MODEL_DIR, DATA_DIR, MODEL_CHECKPOINT

def preprocess_dataset(dataset, processor):
    """
    Preprocess dataset to ensure it has 'pixel_values' and 'label' columns as tensors.
    Handles datasets with 'image', 'pixel_values', or both.
    Args:
        dataset: Dataset object (from datasets library)
        processor: AutoImageProcessor for preprocessing images
    Returns:
        Dataset with 'pixel_values' (single tensor) and 'label' columns
    """
    # Print dataset columns for minimal verification
    print("Dataset columns:", dataset.column_names)

    # If dataset has pixel_values and label, ensure pixel_values is a tensor
    if "pixel_values" in dataset.column_names and "label" in dataset.column_names:
        print("✅ Dataset has 'pixel_values' and 'label' — validating format.")
        if isinstance(dataset[0]["pixel_values"], (list, np.ndarray)):
            print("Converting list of pixel_values to single tensor...")
            pixel_values = torch.stack([torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x for x in dataset["pixel_values"]])
        else:
            pixel_values = dataset["pixel_values"]
        columns_to_remove = [col for col in dataset.column_names if col not in ["pixel_values", "label"]]
        dataset = dataset.remove_columns(columns_to_remove)
        return Dataset.from_dict({
            "pixel_values": pixel_values,
            "label": dataset["label"]
        })

    # If dataset has 'image' column, preprocess it
    if "image" not in dataset.column_names:
        raise KeyError("Dataset missing 'image' column and no valid 'pixel_values'. Ensure load_and_preprocess_data provides 'image' or correct 'pixel_values'.")

    def process_examples(examples):
        images = []
        for img_data in examples["image"]:
            if hasattr(img_data, 'convert'):
                img = img_data.convert("RGB")
            else:
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(img)
        processed = processor(images, return_tensors="pt")
        return {
            "pixel_values": processed["pixel_values"],
            "label": torch.tensor(examples["label"])
        }

    # Process in batches
    print("Preprocessing dataset from images...")
    batch_size = 32
    processed_data = {"pixel_values": [], "label": []}

    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        processed_batch = process_examples({
            "image": [batch[j]["image"] for j in range(len(batch))],
            "label": [batch[j]["label"] for j in range(len(batch))]
        })
        processed_data["pixel_values"].append(processed_batch["pixel_values"])
        processed_data["label"].append(processed_batch["label"])

    # Concatenate batches
    return Dataset.from_dict({
        "pixel_values": torch.cat(processed_data["pixel_values"], dim=0),
        "label": torch.cat(processed_data["label"], dim=0)
    })

def evaluate_model(model_path, dataset):
    """
    Evaluate the model on the test dataset.
    Args:
        model_path: Path to the saved model (BEST_MODEL_DIR)
        dataset: DatasetDict with train, validation, and test splits
    """
    try:
        # Load model and processor
        model = AutoModelForImageClassification.from_pretrained(model_path)
        try:
            processor = AutoImageProcessor.from_pretrained(model_path)
        except:
            print(f"Warning: No processor found in {model_path}. Falling back to {MODEL_CHECKPOINT}.")
            processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    except Exception as e:
        print(f"Error loading model or processor: {str(e)}")
        return

    # Get label mappings
    id2label = model.config.id2label
    labels = [id2label[i] for i in range(len(id2label))]

    # Preprocess test dataset
    print("\nPreprocessing test dataset...")
    try:
        test_dataset = preprocess_dataset(dataset["test"], processor)
    except Exception as e:
        print(f"Error preprocessing test dataset: {str(e)}")
        return

    # Verify processed dataset columns
    print("Processed test dataset columns:", test_dataset.column_names)

    def collate_fn(examples):
        pixel_values = [ex["pixel_values"] for ex in examples]
        pixel_values = [x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in pixel_values]
        return {
            "pixel_values": torch.stack(pixel_values),
            "labels": torch.tensor([ex["label"] for ex in examples])
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        tokenizer=processor,
        data_collator=collate_fn
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    try:
        test_results = trainer.evaluate(test_dataset)
        print("Test metrics:", test_results)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return

    # Get predictions
    test_outputs = trainer.predict(test_dataset)
    y_true = test_outputs.label_ids
    y_pred = np.argmax(test_outputs.predictions, axis=1)

    # Compute and print metrics
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    # Plot and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")  # Save plot instead of showing
    plt.close()  # Close to avoid display issues

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    try:
        dataset, _, _, _ = load_and_preprocess_data(DATA_DIR)
        evaluate_model(BEST_MODEL_DIR, dataset)
    except Exception as e:
        print(f"Error loading dataset or running evaluation: {str(e)}")