# Facial Expression Recognition with Vision Transformer

This project implements a **Facial Expression Recognition** system using a fine-tuned Vision Transformer (ViT) model to classify seven emotions: **anger**, **disgust**, **fear**, **happy**, **neutral**, **sad**, and **surprise**. The model is trained on the KDEF dataset and a custom webcam-quality dataset, with face cropping performed using YOLOv11n-face. The pipeline supports data preprocessing, model training, checkpoint evaluation, test set evaluation, and real-time webcam-based detection.

![Project Workflow](images/workflow_diagram.png)

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Real-Time Detection Demo](#real-time-detection-demo)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The project fine-tunes a pre-trained Vision Transformer model (`motheecreator/vit-Facial-Expression-Recognition`) on two datasets:
- **KDEF**: Karolinska Directed Emotional Faces, a standard dataset for facial expression recognition.
- **Custom Webcam-Quality Dataset**: A dataset created to mimic real-world webcam conditions, enhancing model robustness for real-time detection.

Faces in both datasets are cropped using **YOLOv11n-face** for consistent input. The pipeline includes:
- **Data Preprocessing**: Applies augmentations (rotation, flip, sharpness, color jitter) and normalization.
- **Model Training**: Fine-tunes the ViT model for 20 epochs.
- **Checkpoint Evaluation**: Evaluates checkpoints to select the best model based on validation accuracy.
- **Test Set Evaluation**: Generates metrics and a confusion matrix for the best model.
- **Real-Time Detection**: Uses MediaPipe for face detection and the ViT model for expression classification via webcam.

The best model achieves **95.01% accuracy** and **94.99% macro F1** on the test set.

## Datasets
- **Karolinska Directed Emotional Faces(KDEF) Dataset**:
  - **Description**: A standard dataset with posed facial expressions.
  - **Size after applying YOLOv11n-face**: 3,423 train, 978 validation, 490 test images.
  - **Classes**: Anger, disgust, fear, happy, neutral, sad, surprise.
- **Custom Webcam-Quality Dataset**:
  - **Description**: A dataset created to simulate webcam conditions, improving real-time performance by me.
  - **Size YOLOv11n-face**: 354 train, 102  validation, 51 test images.
  - **Details**: Faces cropped using YOLOv11n-face to match KDEF preprocessing.
  - **Note**: Combined with KDEF in the dataset folder for unified training.
- **Preprocessing**: Images are cropped with YOLOv11n-face, resized to 224x224, and augmented (train) or normalized (validation/test).

**Note**: The datasets are not included due to size and licensing. Update `DATA_DIR` in `config.py` to point to your dataset path (`KDEF_Cropped_YOLO11n_Split + M_B_Split`).

## Features
- Fine-tuned ViT model for seven emotion classes.
- YOLOv11n-face for accurate face cropping.
- Data augmentation for robust training.
- Comprehensive checkpoint evaluation (accuracy, F1, loss).
- Real-time detection with MediaPipe and OpenCV.
- Visualizations: Confusion matrix and checkpoint performance plots.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/facial-expression-recognition.git
   cd facial-expression-recognition
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Paths**:
   - Edit `config.py`:
     - `DATA_DIR`: Path to your dataset (`KDEF_Cropped_YOLO11n_Split + M_B_Split`).
     - `OUTPUT_DIR`: Directory for training checkpoints.
     - `BEST_MODEL_DIR`: Directory for the best model.
     - `MODEL_CHECKPOINT`: `motheecreator/vit-Facial-Expression-Recognition`.

5. **Hardware**:
   - GPU (NVIDIA CUDA-enabled) recommended.
   - 8GB+ RAM for preprocessing.
   - Webcam for real-time detection.

6. **YOLOv11n-face Setup**:
   - Ensure YOLOv11n-face weights are available for preprocessing (not included in this repo).
   - Update dataset preprocessing if using a different face detection model.

## Usage
Run the pipeline scripts in order:

1. **Preprocess Data**:
   ```bash
   python data_preprocessing.py
   ```
   Loads and preprocesses the combined KDEF and custom dataset.

2. **Train Model**:
   ```bash
   python model_training.py
   ```
   Fine-tunes the ViT model for 20 epochs, saving checkpoints every 100 steps.

3. **Evaluate Checkpoints**:
   ```bash
   python model_saving.py
   ```
   Evaluates checkpoints, saves the best model to `BEST_MODEL_DIR`, and generates plots (`checkpoint_plots.png`).

   ![Checkpoint Plots](images/checkpoint_plots.png)

4. **Evaluate Best Model**:
   ```bash
   python model_evaluation.py
   ```
   Evaluates the best model on the test set, producing metrics and a confusion matrix (`confusion_matrix.png`).

   ![Confusion Matrix](images/confusion_matrix.png)

5. **Real-Time Detection**:
   ```bash
   python real_time_detection.py
   ```
   Runs webcam-based detection. Press `q` to exit.

   **Demo Video**:
   ```markdown
   <video controls>
     <source src="videos/realtime_detection_demo.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   ```

   To record a demo:
   - Use OBS Studio to capture `real_time_detection.py` output.
   - Save as `videos/realtime_detection_demo.mp4`.
   - Push to GitHub.

## Results
The best model (selected by validation accuracy) achieves:
- **Accuracy**: 95.01%
- **Macro F1**: 94.99%
- **Weighted F1**: 95.00%

**Classification Report**:
```
              precision    recall  f1-score   support
anger         0.9740    0.9615    0.9677        78
disgust       0.9733    0.9481    0.9605        77
fear          0.9067    0.9189    0.9128        74
happy         0.9733    0.9733    0.9733        75
neutral       0.9412    1.0000    0.9697        80
sad           0.9324    0.9324    0.9324        74
surprise      0.9500    0.9157    0.9325        83
accuracy                          0.9501       541
macro avg     0.9501    0.9500    0.9499       541
weighted avg  0.9504    0.9501    0.9500       541
```

**Visualizations**:
- **Confusion Matrix**: Saved as `confusion_matrix.png` by `model_evaluation.py`.
- **Checkpoint Plots**: Accuracy, F1, and loss trends saved as `checkpoint_plots.png` by `model_saving.py`.

## Real-Time Detection Demo
The `real_time_detection.py` script uses MediaPipe for face detection and the ViT model for classification, displaying:
- Bounding box around detected faces.
- Predicted emotion label.
- Top-3 probability scores.

**Screenshot**:
![Real-Time Detection](images/realtime_detection_screenshot.png)

**Video Demo**:
- Record `real_time_detection.py` using a webcam.
- Save to `videos/realtime_detection_demo.mp4`.
- Push to GitHub for rendering in the README.

## Contributing
1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push: `git push origin feature/your-feature`.
5. Open a Pull Request.

## License
[MIT License](LICENSE)
