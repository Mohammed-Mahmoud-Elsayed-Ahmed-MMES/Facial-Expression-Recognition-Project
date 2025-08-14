import torch
import numpy as np
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

from data_preprocessing import load_and_preprocess_data
from config import (
    MODEL_CHECKPOINT, 
    OUTPUT_DIR, 
    LEARNING_RATE, 
    TRAIN_BATCH_SIZE, 
    EVAL_BATCH_SIZE, 
    NUM_EPOCHS, 
    WARMUP_STEPS, 
    LOGGING_STEPS, 
    EVAL_STEPS
)

def train_model(dataset, label2id, id2label, model_checkpoint=MODEL_CHECKPOINT):
    """
    Train the model on the provided dataset.
    """
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=EVAL_STEPS,  # Save checkpoint every 100 steps
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=2,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        report_to="none",
        save_total_limit=5  # Limit to 5 checkpoints to save disk space
    )

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return {"accuracy": (predictions == eval_pred.label_ids).mean()}

    def collate_fn(examples):
        return {
            "pixel_values": torch.stack([ex["pixel_values"] for ex in examples]),
            "labels": torch.tensor([ex["label"] for ex in examples])
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=collate_fn
    )

    print("\nStarting training...")
    trainer.train()
    print("\nTraining completed.")
    print(f"The trainer identified this as the best checkpoint: {trainer.state.best_model_checkpoint}")
    print("Run save_model.py to analyze metrics and save the best model.")

    return trainer

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    dataset, label2id, id2label, _ = load_and_preprocess_data()
    
    print("\nInitializing model training...")
    train_model(dataset, label2id, id2label)