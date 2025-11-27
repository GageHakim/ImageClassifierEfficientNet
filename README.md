# Bacteria Image Classifier

This project is an image classifier for different types of bacteria, built using EfficientNet and PyTorch.

## Project Structure
```
.
├── EfficientNet-PyTorch
│   ├── efficientnet_pytorch
│   ├── train.py
│   └── predict.py
├── dataset
│   ├── bacteria_resized_224
│   ├── train
│   ├── val
│   └── test
├── prepare_dataset.py
├── requirements.txt
└── README.md
```

## Dataset

The dataset consists of images of the following bacteria:

*   B.subtilis
*   C.albicans
*   Contamination
*   E.coli
*   P.aeruginosa
*   S.aureus

The raw images are expected to be in the `dataset/bacteria_resized_224` directory, organized into subdirectories for each class.

## Model

The classifier uses the EfficientNet-B0 architecture, implemented in PyTorch using the [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) library.

## Getting Started

### Prerequisites

*   Python 3.x
*   PyTorch
*   TorchVision
*   NumPy
*   scikit-learn
*   Pillow

You can install all the dependencies using the `requirements.txt` file.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/GageHakim/ImageClassifierEfficientNet.git
    cd ImageClassifierEfficientNet
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Prepare the Dataset

Run the `prepare_dataset.py` script from the root directory to split the raw images into training, validation, and testing sets. The script will move the files from `dataset/bacteria_resized_224` to `dataset/train`, `dataset/val`, and `dataset/test`.

```bash
python prepare_dataset.py
```

### 2. Train the Model

To train the model, navigate to the `EfficientNet-PyTorch` directory and run the `train.py` script.

```bash
cd EfficientNet-PyTorch
python train.py
```

The script will train the model and save the best performing version to `best_model.pth`.

### 3. Evaluate the Model

To evaluate the trained model on the test set, run the `train.py` script with the `--test` flag.

```bash
cd EfficientNet-PyTorch # If you are not already in this directory
python train.py --test
```

This will load the `best_model.pth` and print a classification report with accuracy, precision, recall, and F1-score.

### 4. Predict on a New Image

To use the trained model to predict the class of a new image, use the `predict.py` script. Place your image in the `EfficientNet-PyTorch` directory or provide the full path to the image.

```bash
cd EfficientNet-PyTorch # If you are not already in this directory
python predict.py /path/to/your/image.jpg
```

The script will load the trained model and print the predicted class for the image.
