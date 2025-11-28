
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from efficientnet_pytorch import EfficientNet
import argparse
import torch.nn.functional as F


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    Generates and saves a confusion matrix heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to {save_path}")
    plt.close()


def get_test_loader(data_dir, batch_size):
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dir = os.path.join(data_dir, 'test')
    test_dataset = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_loader, test_dataset.classes

def main():
    parser = argparse.ArgumentParser(description='Evaluate an EfficientNet model.')
    parser.add_argument('--model-path', type=str, default='EfficientNet-PyTorch/best_model.pth',
                        help='Path to the trained model weights.')
    parser.add_argument('--data-dir', type=str, default='./dataset',
                        help='Path to the dataset directory.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation.')
    args = parser.parse_args()

    # --- Hardware Optimization ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # --- Data ---
    test_loader, class_names = get_test_loader(args.data_dir, args.batch_size)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {', '.join(class_names)}")

    # --- Load Model ---
    print("Loading EfficientNet model...")
    # Load the model architecture. 
    # 'num_classes' must match the model that was trained.
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    
    # Load the trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Evaluating on {len(test_loader.dataset)} images...")

    # --- Evaluation Loop ---
    all_preds = []
    all_labels = []
    total_time = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            
            # Timing
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.time()

            outputs = model(images)

            if device.type == 'cuda': torch.cuda.synchronize()
            end = time.time()

            total_time += (end - start)

            _, predicted = torch.max(outputs, 1)

            # Store results on CPU for metrics calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Metrics ---
    # 1. Accuracy
    acc = accuracy_score(all_labels, all_preds) * 100

    # 2. Timing
    avg_inference_time_ms = (total_time / len(test_loader.dataset)) * 1000
    fps = 1.0 / (total_time / len(test_loader.dataset))

    print("-" * 30)
    print(f"Global Accuracy: {acc:.2f}%")
    print(f"Inference Speed: {avg_inference_time_ms:.2f} ms/image ({fps:.1f} FPS)")
    print("-" * 30)

    # 3. Detailed Report (Precision, Recall, F1)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 4. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, save_path='efficientnet_confusion_matrix.png')


if __name__ == '__main__':
    main()
