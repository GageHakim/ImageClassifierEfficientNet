
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import argparse
import time
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score

# Note: This script requires scikit-learn for advanced metrics.
# Install with: pip install scikit-learn

def get_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset.classes

def get_test_loader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dir = os.path.join(data_dir, 'test')
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader, test_dataset.classes

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_model.pth')
                
    print(f'Best val Acc: {best_acc:4f}')

def test_model(model, test_loader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    end_time = time.time()
    
    num_images = len(test_loader.dataset)
    test_loss = running_loss / num_images
    test_acc = running_corrects.double() / num_images
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    class_names = test_loader.dataset.classes
    
    # Calculate AUC using One-vs-Rest strategy for multi-class
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan') # AUC is not defined for some cases (e.g., only one class present in labels)
        print("Warning: AUC could not be calculated.")

    print(f'Test completed in {end_time - start_time:.2f} seconds')
    print(f'Processed {num_images} images.')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Accuracy: {test_acc:.4f}')
    print(f'AUC (One-vs-Rest): {auc:.4f}')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))

def main():
    parser = argparse.ArgumentParser(description='Train or test the model.')
    parser.add_argument('--test', action='store_true', help='Test the model.')
    args = parser.parse_args()

    data_dir = '../dataset'
    batch_size = 32
    
    if args.test:
        test_loader, classes = get_test_loader(data_dir, batch_size)
        num_classes = len(classes)
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        model.load_state_dict(torch.load('best_model.pth'))
        criterion = nn.CrossEntropyLoss()
        test_model(model, test_loader, criterion)
    else:
        num_epochs = 10
        train_loader, val_loader, classes = get_data_loaders(data_dir, batch_size)
        num_classes = len(classes)
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

if __name__ == '__main__':
    main()
