import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet50_Weights

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image

#######################################
#           DATA LOADING              #
#######################################

def load_data(train_data_path, training_split=0.2, batch_size=128, num_workers=4):
    # Define image transformations
    transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])
    
    # Use ImageFolder to load images from directory structure
    dataset = datasets.ImageFolder(train_data_path, transform=transform)
    dataset_size = len(dataset)
    val_size = int(dataset_size * training_split)
    train_size = dataset_size - val_size

    # Split the dataset into training and validation subsets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, dataset

def compute_weights(train_subset, full_dataset):
    # Extract targets from the training subset (random_split returns a Subset)
    train_targets = [full_dataset.targets[i] for i in train_subset.indices]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_targets), y=train_targets)
    print("Class weights (per class):", class_weights)
    class_counts = np.bincount(train_targets)
    print("Class counts:", class_counts)
    # For binary classification with BCEWithLogitsLoss, compute pos_weight (ratio of negatives/positives)
    pos_weight = class_counts[0] / class_counts[1]
    pos_weight = torch.tensor([pos_weight], dtype=torch.float)
    return pos_weight

#######################################
#           MODEL DEFINITION          #
#######################################

class CustomResNet50(nn.Module):
    def __init__(self, extra_conv_layers=5):
        super(CustomResNet50, self).__init__()
        # Load a pretrained ResNet50 and remove the avgpool and fc layers
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base = nn.Sequential(*list(base_model.children())[:-2])  # output shape: (N, 2048, H, W)
        
        # Add extra convolutional layers.
        # The first extra layer takes 2048 channels and outputs 128 channels.
        # Subsequent layers keep 128 channels.
        conv_layers = []
        for i in range(extra_conv_layers):
            in_channels = 2048 if i == 0 else 128
            conv_layers.append(nn.Conv2d(in_channels, 128, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.BatchNorm2d(128))
        self.extra_convs = nn.Sequential(*conv_layers)
        
        # Global Average Pooling to collapse spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)  # output raw logits

    def forward(self, x):
        x = self.base(x)
        x = self.extra_convs(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # flatten to (N, 128)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Note: no sigmoid here, as we use BCEWithLogitsLoss
        return x

#######################################
#           TRAINING LOOP             #
#######################################

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)  # BCEWithLogitsLoss expects float labels of shape (N, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend((preds > 0.5).int().cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    return model

#######################################
#           EVALUATION                #
#######################################

def evaluate_model(model, val_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).int()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    print("Classification Report")
    print(classification_report(all_labels, all_preds))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["AI_GENERATED", "NON_AI_GENERATED"],
                yticklabels=["AI_GENERATED", "NON_AI_GENERATED"])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()
    return cm

#######################################
#         GRADCAM VISUALIZATION       #
#######################################

def gradcam_heatmap(image_path, output_name, model, target_layer, target_class=None, device='cpu'):
    # Read and pre-process image
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(device)
    
    # Use GradCAM from pytorch_grad_cam
    with GradCAM(model=model,
                 target_layers=[target_layer],) as cam:
        cam.batch_size = 32  # adjust for speed if needed
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=target_class,
                            aug_smooth=True,
                            eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        
        # Guided Backpropagation
        gb_model = GuidedBackpropReLUModel(model=model, device=device)
        gb = gb_model(input_tensor, target_category=target_class)
        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)
        
        # Save outputs
        output_dir = os.path.join(os.path.abspath(os.path.curdir), 'AI-Image-Detection-CNN/GradCam')
        os.makedirs(output_dir, exist_ok=True)
        
        cam_output_path = os.path.join(output_dir, f'{output_name}_cam.jpg')
        gb_output_path = os.path.join(output_dir, f'{output_name}_gb.jpg')
        cam_gb_output_path = os.path.join(output_dir, f'{output_name}_cam_gb.jpg')
        
        cv2.imwrite(cam_output_path, cam_image)
        cv2.imwrite(gb_output_path, gb)
        cv2.imwrite(cam_gb_output_path, cam_gb)

#######################################
#           MAIN EXECUTION            #
#######################################

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python pytorch_ai_art_classification.py <ai_art_classification_top_dir> <ModelOutputDir> <ModelName>")
        sys.exit(1)
        
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    current_dir = os.path.abspath(os.path.curdir)
    top_dir = sys.argv[1]
    model_output_dir = os.path.join(os.path.curdir, sys.argv[2])
    model_name = f"{sys.argv[3]}_model.pth"
    
    # Define paths
    train_data_path = os.path.join(current_dir, top_dir, 'train')
    test_data_path = os.path.join(current_dir, top_dir, 'test')
    print("Training Data Path:", train_data_path)
    print("Testing Data Path:", test_data_path)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, full_dataset = load_data(train_data_path, training_split=0.2)
    
    # Compute class weights from the training subset (used for BCEWithLogitsLoss)
    print("Computing class weights...")
    pos_weight = compute_weights(train_loader.dataset, full_dataset)
    
    # Create the model
    print("Creating model...")
    model = CustomResNet50(extra_conv_layers=5)
    model = model.to(device)
    
    # (Optional) Freeze the base layers
    for param in model.base.parameters():
        param.requires_grad = False
        
    # Set up optimizer and loss function
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # Train the model
    print("Training model...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer,
                        num_epochs=50, device=device, patience=5)
    
    # Save the model
    print("Saving model...")
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, model_name)
    torch.save(model.state_dict(), model_path)
    
    # Evaluate the model
    print("Evaluating model...")
    cm = evaluate_model(model, val_loader, device)
    
    # GradCAM Visualization:
    # Choose a target convolutional layer. For instance, here we pick the first conv layer
    # of the extra convolution block. This may need to be adjusted
    target_layer = model.extra_convs[0]
    
    gradcam_heatmap(os.path.join(current_dir, 'RawData/ai_art_classification/train/AI_GENERATED/0.jpg'),
                    'fake_img_1', model, target_layer, target_class=None, device=device)
    gradcam_heatmap(os.path.join(current_dir, 'RawData/ai_art_classification/train/NON_AI_GENERATED/3.jpg'),
                    'real_img_1', model, target_layer, target_class=None, device=device)
