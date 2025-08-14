import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoModelForImageClassification, AutoImageProcessor, Trainer
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import OUTPUT_DIR, BEST_MODEL_DIR, MODEL_CHECKPOINT, DATA_DIR

class CheckpointEvaluator:
    def __init__(self, checkpoint_dir=OUTPUT_DIR, model_checkpoint=MODEL_CHECKPOINT):
        self.checkpoint_dir = checkpoint_dir
        self.model_checkpoint = model_checkpoint
        self.results = []
        
    def get_all_checkpoints(self):
        """Get all checkpoint directories sorted by step number"""
        checkpoints = []
        for item in os.listdir(self.checkpoint_dir):
            if os.path.isdir(os.path.join(self.checkpoint_dir, item)) and item.startswith('checkpoint-'):
                try:
                    step_num = int(item.split('-')[1])
                    checkpoints.append((item, step_num))
                except ValueError:
                    continue
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x[1])
        return [checkpoint[0] for checkpoint in checkpoints]
    
    def prepare_dataset_for_evaluation(self, dataset, processor):
        """Prepare dataset for evaluation without transforms"""
        from datasets import Dataset
        from PIL import Image
        import io
        
        def process_examples(examples):
            # Process images using the processor
            images = []
            for img_data in examples["image"]:
                if hasattr(img_data, 'convert'):
                    # PIL Image
                    img = img_data.convert("RGB")
                else:
                    # Handle other image formats
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                images.append(img)
            
            # Use processor to get pixel_values
            processed = processor(images, return_tensors="pt")
            
            return {
                "pixel_values": processed["pixel_values"],
                "labels": torch.tensor(examples["label"])
            }
        
        # Process in batches to avoid memory issues
        processed_data = {"pixel_values": [], "labels": []}
        batch_size = 32
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            batch_dict = {
                "image": [batch[j]["image"] for j in range(len(batch))],
                "label": [batch[j]["label"] for j in range(len(batch))]
            }
            processed_batch = process_examples(batch_dict)
            
            processed_data["pixel_values"].append(processed_batch["pixel_values"])
            processed_data["labels"].append(processed_batch["labels"])
        
        # Concatenate all batches
        all_pixel_values = torch.cat(processed_data["pixel_values"], dim=0)
        all_labels = torch.cat(processed_data["labels"], dim=0)
        
        return all_pixel_values, all_labels
    
    def evaluate_single_checkpoint(self, checkpoint_name, val_dataset, test_dataset):
        """Evaluate a single checkpoint on validation and test sets"""
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        step_num = int(checkpoint_name.split('-')[1])
        
        print(f"\n{'='*60}")
        print(f"Evaluating {checkpoint_name} (Step {step_num})")
        print(f"{'='*60}")
        
        try:
            # Load model and processor
            model = AutoModelForImageClassification.from_pretrained(checkpoint_path)
            
            try:
                processor = AutoImageProcessor.from_pretrained(checkpoint_path)
            except:
                processor = AutoImageProcessor.from_pretrained(self.model_checkpoint)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            
            # Prepare datasets
            print("Processing validation dataset...")
            val_pixel_values, val_labels = self.prepare_dataset_for_evaluation(val_dataset, processor)
            
            print("Processing test dataset...")
            test_pixel_values, test_labels = self.prepare_dataset_for_evaluation(test_dataset, processor)
            
            # Evaluate on validation set
            print("Evaluating on validation set...")
            val_predictions, val_loss = self.evaluate_dataset(
                model, val_pixel_values, val_labels, device
            )
            val_accuracy = accuracy_score(val_labels.cpu(), val_predictions)
            val_f1_macro = f1_score(val_labels.cpu(), val_predictions, average='macro')
            val_f1_weighted = f1_score(val_labels.cpu(), val_predictions, average='weighted')
            
            # Evaluate on test set
            print("Evaluating on test set...")
            test_predictions, test_loss = self.evaluate_dataset(
                model, test_pixel_values, test_labels, device
            )
            test_accuracy = accuracy_score(test_labels.cpu(), test_predictions)
            test_f1_macro = f1_score(test_labels.cpu(), test_predictions, average='macro')
            test_f1_weighted = f1_score(test_labels.cpu(), test_predictions, average='weighted')
            
            # Store results
            result = {
                'checkpoint': checkpoint_name,
                'step': step_num,
                'val_accuracy': val_accuracy,
                'val_f1_macro': val_f1_macro,
                'val_f1_weighted': val_f1_weighted,
                'val_loss': val_loss,
                'test_accuracy': test_accuracy,
                'test_f1_macro': test_f1_macro,
                'test_f1_weighted': test_f1_weighted,
                'test_loss': test_loss,
                'status': 'success'
            }
            
            print(f"✓ Validation - Accuracy: {val_accuracy:.4f}, F1-Macro: {val_f1_macro:.4f}, Loss: {val_loss:.4f}")
            print(f"✓ Test - Accuracy: {test_accuracy:.4f}, F1-Macro: {test_f1_macro:.4f}, Loss: {test_loss:.4f}")
            
            return result
            
        except Exception as e:
            print(f"Error evaluating {checkpoint_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'checkpoint': checkpoint_name,
                'step': step_num,
                'val_accuracy': 0,
                'val_f1_macro': 0,
                'val_f1_weighted': 0,
                'val_loss': float('inf'),
                'test_accuracy': 0,
                'test_f1_macro': 0,
                'test_f1_weighted': 0,
                'test_loss': float('inf'),
                'status': 'failed',
                'error': str(e)
            }
    
    def evaluate_dataset(self, model, pixel_values, labels, device, batch_size=32):
        """Evaluate model on a dataset"""
        model.eval()
        all_predictions = []
        total_loss = 0.0
        num_batches = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for i in range(0, len(pixel_values), batch_size):
                batch_pixels = pixel_values[i:i+batch_size].to(device)
                batch_labels = labels[i:i+batch_size].to(device)
                
                outputs = model(pixel_values=batch_pixels, labels=batch_labels)
                loss = outputs.loss
                logits = outputs.logits
                
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return np.array(all_predictions), avg_loss
    
    def evaluate_all_checkpoints(self, val_dataset, test_dataset):
        """Evaluate all checkpoints and return results"""
        checkpoints = self.get_all_checkpoints()
        
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {self.checkpoint_dir}")
        
        print(f"Found {len(checkpoints)} checkpoints to evaluate:")
        for checkpoint in checkpoints:
            print(f"  - {checkpoint}")
        
        self.results = []
        for checkpoint in checkpoints:
            result = self.evaluate_single_checkpoint(checkpoint, val_dataset, test_dataset)
            self.results.append(result)
        
        return self.results
    
    def analyze_results(self):
        """Analyze and display results"""
        if not self.results:
            raise ValueError("No results to analyze. Run evaluate_all_checkpoints first.")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.results)
        df_success = df[df['status'] == 'success'].copy()
        
        if df_success.empty:
            print("No successful evaluations found!")
            return None
        
        print(f"\n{'='*80}")
        print("CHECKPOINT EVALUATION RESULTS")
        print(f"{'='*80}")
        
        # Sort by validation accuracy (descending)
        df_success = df_success.sort_values('val_accuracy', ascending=False)
        
        print("\nSUMMARY TABLE (Sorted by Validation Accuracy)")
        print("-" * 120)
        print(f"{'Checkpoint':<15} {'Step':<6} {'Val Acc':<8} {'Val F1':<8} {'Test Acc':<9} {'Test F1':<8} {'Val Loss':<9} {'Test Loss':<9}")
        print("-" * 120)
        
        for _, row in df_success.iterrows():
            print(f"{row['checkpoint']:<15} {row['step']:<6} {row['val_accuracy']:<8.4f} {row['val_f1_macro']:<8.4f} "
                  f"{row['test_accuracy']:<9.4f} {row['test_f1_macro']:<8.4f} {row['val_loss']:<9.4f} {row['test_loss']:<9.4f}")
        
        # Best checkpoints analysis
        best_val_acc = df_success.iloc[0]
        best_test_acc = df_success.loc[df_success['test_accuracy'].idxmax()]
        best_val_f1 = df_success.loc[df_success['val_f1_macro'].idxmax()]
        
        print(f"\nBEST CHECKPOINTS ANALYSIS")
        print("-" * 60)
        print(f"Best Validation Accuracy: {best_val_acc['checkpoint']} ({best_val_acc['val_accuracy']:.4f})")
        print(f"Best Test Accuracy:       {best_test_acc['checkpoint']} ({best_test_acc['test_accuracy']:.4f})")
        print(f"Best Validation F1:       {best_val_f1['checkpoint']} ({best_val_f1['val_f1_macro']:.4f})")
        
        # Recommendation
        print(f"\nRECOMMENDATION")
        print("-" * 60)
        print(f"Recommended checkpoint: {best_val_acc['checkpoint']}")
        print(f"Reason: Highest validation accuracy ({best_val_acc['val_accuracy']:.4f})")
        print(f"Test performance: {best_val_acc['test_accuracy']:.4f} accuracy, {best_val_acc['test_f1_macro']:.4f} F1-macro")
        
        return best_val_acc['checkpoint']
    
    def plot_results(self):
        """Plot evaluation results"""
        if not self.results:
            raise ValueError("No results to plot. Run evaluate_all_checkpoints first.")
        
        df = pd.DataFrame(self.results)
        df_success = df[df['status'] == 'success'].copy()
        
        if df_success.empty:
            print("No successful evaluations to plot!")
            return
        
        df_success = df_success.sort_values('step')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Checkpoint Evaluation Results', fontsize=16, fontweight='bold')
        
        steps = df_success['step']
        
        # Accuracy comparison
        ax1.plot(steps, df_success['val_accuracy'], 'o-', label='Validation', color='blue', linewidth=2)
        ax1.plot(steps, df_success['test_accuracy'], 's-', label='Test', color='red', linewidth=2)
        ax1.set_title('Accuracy Comparison')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1-Score comparison
        ax2.plot(steps, df_success['val_f1_macro'], 'o-', label='Validation', color='green', linewidth=2)
        ax2.plot(steps, df_success['test_f1_macro'], 's-', label='Test', color='orange', linewidth=2)
        ax2.set_title('F1-Score (Macro) Comparison')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('F1-Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Loss comparison
        ax3.plot(steps, df_success['val_loss'], 'o-', label='Validation', color='purple', linewidth=2)
        ax3.plot(steps, df_success['test_loss'], 's-', label='Test', color='brown', linewidth=2)
        ax3.set_title('Loss Comparison')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance summary
        metrics = ['Validation Accuracy', 'Test Accuracy', 'Validation F1', 'Test F1']
        best_values = [
            df_success['val_accuracy'].max(),
            df_success['test_accuracy'].max(),
            df_success['val_f1_macro'].max(),
            df_success['test_f1_macro'].max()
        ]
        
        bars = ax4.bar(metrics, best_values, color=['skyblue', 'lightcoral', 'lightgreen', 'wheat'])
        ax4.set_title('Best Performance Summary')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, best_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def save_best_model(self, best_checkpoint_name, save_dir=BEST_MODEL_DIR):
        """Save the best model to the specified directory"""
        best_checkpoint_path = os.path.join(self.checkpoint_dir, best_checkpoint_name)
        
        print(f"\n{'='*60}")
        print("SAVING BEST MODEL")
        print(f"{'='*60}")
        print(f"Source: {best_checkpoint_path}")
        print(f"Destination: {save_dir}")
        
        try:
            # Create save directory
            os.makedirs(save_dir, exist_ok=True)
            
            # Load and save model
            model = AutoModelForImageClassification.from_pretrained(best_checkpoint_path)
            model.save_pretrained(save_dir)
            
            # Load and save processor
            try:
                processor = AutoImageProcessor.from_pretrained(best_checkpoint_path)
            except:
                processor = AutoImageProcessor.from_pretrained(self.model_checkpoint)
            processor.save_pretrained(save_dir)
            
            # Save metadata
            metadata = {
                'best_checkpoint': best_checkpoint_name,
                'source_path': best_checkpoint_path,
                'saved_at': datetime.now().isoformat(),
                'evaluation_results': [r for r in self.results if r['checkpoint'] == best_checkpoint_name][0]
            }
            
            with open(os.path.join(save_dir, 'model_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Successfully saved best model: {best_checkpoint_name}")
            print(f"Model directory: {save_dir}")
            print(f"Metadata saved to: {os.path.join(save_dir, 'model_metadata.json')}")
            
            return save_dir
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return None
    
    def save_evaluation_report(self, output_file="checkpoint_evaluation_report.json"):
        """Save detailed evaluation report"""
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'checkpoint_directory': self.checkpoint_dir,
            'total_checkpoints': len(self.results),
            'successful_evaluations': len([r for r in self.results if r['status'] == 'success']),
            'failed_evaluations': len([r for r in self.results if r['status'] == 'failed']),
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to: {output_file}")


def main():
    """Main evaluation pipeline"""
    print("Starting Checkpoint Evaluation Pipeline")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = CheckpointEvaluator()
    
    # Load dataset using your existing preprocessing function
    print("Loading and preprocessing dataset...")
    from datasets import load_dataset, DatasetDict
    
    # Load raw dataset without transforms for evaluation
    MB_dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
    dataset = DatasetDict({
        "train": MB_dataset["train"],
        "validation": MB_dataset["validation"],
        "test": MB_dataset["test"]
    })
    
    # Get labels
    labels = dataset["train"].features["label"].names
    label2id = {label: i for i, label in enumerate(labels)}
    
    print(f"Dataset loaded successfully")
    print(f"   - Train samples: {len(dataset['train'])}")
    print(f"   - Validation samples: {len(dataset['validation'])}")
    print(f"   - Test samples: {len(dataset['test'])}")
    print(f"   - Classes: {list(label2id.keys())}")
    
    # Evaluate all checkpoints
    print(f"\nEvaluating all checkpoints...")
    evaluator.evaluate_all_checkpoints(dataset["validation"], dataset["test"])
    
    # Analyze results
    print(f"\nAnalyzing results...")
    best_checkpoint = evaluator.analyze_results()
    
    if best_checkpoint:
        # Plot results
        print(f"\nGenerating plots...")
        evaluator.plot_results()
        
        # Save best model
        print(f"\nSaving best model...")
        saved_path = evaluator.save_best_model(best_checkpoint)
        
        # Save evaluation report
        print(f"\nSaving evaluation report...")
        evaluator.save_evaluation_report()
        
        if saved_path:
            print(f"\nEVALUATION COMPLETE!")
            print(f"Best checkpoint: {best_checkpoint}")
            print(f"Model saved to: {saved_path}")
            print(f"Ready for inference and real-time usage!")
    else:
        print(f"\nNo successful evaluations found. Please check your checkpoints.")


if __name__ == "__main__":
    main()