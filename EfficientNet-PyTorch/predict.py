
import torch
from torchvision import transforms
from PIL import Image
import argparse
import os
from efficientnet_pytorch import EfficientNet

def predict(image_path, model, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        
    return class_names[preds[0]]

def main():
    parser = argparse.ArgumentParser(description='Predict the class of a single image.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    args = parser.parse_args()
    
    # These class names are based on the folder names in the training data
    class_names = ['B.subtilis', 'C.albicans', 'Contamination', 'E.coli', 'P.aeruginosa', 'S.aureus']
    
    num_classes = len(class_names)
    
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    model.load_state_dict(torch.load('best_model.pth'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    predicted_class = predict(args.image_path, model, class_names)
    
    print(f'The predicted class for the image is: {predicted_class}')

if __name__ == '__main__':
    main()
