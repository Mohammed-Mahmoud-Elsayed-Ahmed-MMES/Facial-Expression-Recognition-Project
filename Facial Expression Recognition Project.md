# Facial Expression Recognition Project

This project implements a facial expression recognition system using a Vision Transformer (ViT) model, fine-tuned on a custom dataset. It includes modules for data preprocessing, model training, and checkpoint evaluation.

## Project Structure

```
.
├── config.py
├── data_preprocessing.py
├── model_training.py
├── model_saving.py
├── requirements.txt
├── README.md
└── media/
    ├── video_demo.mp4
    └── image_example.png
```

## Files Overview

- `config.py`: Contains all the configuration settings and hyperparameters for the project, including dataset paths, model checkpoints, learning rates, batch sizes, and training epochs.
- `data_preprocessing.py`: Handles the loading and preprocessing of the image dataset. It uses Hugging Face `datasets` and `transformers` libraries to prepare the data for training, including transformations like resizing, normalization, random rotation, and horizontal flips.
- `model_training.py`: Implements the model training logic. It utilizes the `transformers.Trainer` API to fine-tune an `AutoModelForImageClassification` on the preprocessed dataset. It also defines the `compute_metrics` function for evaluating model performance during training.
- `model_saving.py`: Provides utilities for evaluating and saving the best model checkpoints. It includes a `CheckpointEvaluator` class to assess the performance of various checkpoints on validation and test sets, and to identify the best performing model based on accuracy and F1-score.
- `requirements.txt`: Lists all the Python dependencies required to run this project. You can install them using `pip install -r requirements.txt`.
- `README.md`: This file, providing an overview of the project, its structure, and instructions.
- `media/`: This directory is intended to store visual assets such as demonstration videos and example images.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Configure Project Settings**: Open `config.py` and update the `DATA_DIR`, `OUTPUT_DIR`, and `BEST_MODEL_DIR` paths according to your local setup.

2. **Run Data Preprocessing**: The `data_preprocessing.py` script will load and preprocess your dataset. You can run it directly to verify data preparation:
   ```bash
   python data_preprocessing.py
   ```

3. **Train the Model**: Start the model training process by running `model_training.py`:
   ```bash
   python model_training.py
   ```
   The training checkpoints will be saved in the directory specified by `OUTPUT_DIR` in `config.py`.

4. **Evaluate and Save Best Model**: After training, use `model_saving.py` to evaluate the checkpoints and save the best performing model:
   ```bash
   python model_saving.py
   ```

## Demo

### Video Demonstration

Here's a short video demonstrating the facial expression recognition in action:

[![Video Demo](media/video_thumbnail.png)](media/video_demo.mp4)

*(Click the image above to watch the video demo. Make sure to replace `video_thumbnail.png` with an actual thumbnail image for your video.)*

### Example Images

Below are some examples of facial expressions recognized by the model:

![Example 1](media/image_example.png)

*(Replace `image_example.png` with an actual image showing a recognized expression.)*

## Libraries Used (requirements.txt)

- `torch`
- `numpy`
- `huggingface-hub`
- `datasets`
- `transformers`
- `tensorflow`
- `torchvision`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `tqdm`
- `Pillow`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

