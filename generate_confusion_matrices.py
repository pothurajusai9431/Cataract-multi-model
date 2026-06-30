import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from classification_model import DeepCNN, DeepANN, ResNet, VGG, AlexNet

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model configurations
classes = ["Cataract", "Normal"]
model_classes = {
    "DeepCNN": DeepCNN,
    "DeepANN": DeepANN,
    "ResNet": ResNet,
    "VGG": VGG,
    "AlexNet": AlexNet
}

# Image transform (same as in app.py)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load test dataset
test_dir = "Dataset/Test"
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Test dataset size: {len(test_dataset)}")
print(f"Classes: {classes}")

# Create plots directory if it doesn't exist
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Function to load a model
def load_model(model_name):
    model_path = os.path.join("models", f"catarct_or_normal{model_name}.pth")
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found at {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = model_classes[model_name](num_classes=len(classes))
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model = model_classes[model_name](num_classes=len(classes))
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    return model

# Function to get predictions for a model
def get_predictions(model, test_loader):
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    return np.array(true_labels), np.array(pred_labels)

# Generate confusion matrices for each model
for model_name in model_classes.keys():
    print(f"\nProcessing model: {model_name}")

    # Load model
    model = load_model(model_name)
    if model is None:
        continue

    # Get predictions
    true_labels, pred_labels = get_predictions(model, test_loader)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', ax=plt.gca())
    plt.title(f'Confusion Matrix - {model_name}')
    plt.grid(False)

    # Save plot
    plot_path = os.path.join(plots_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved confusion matrix for {model_name} to {plot_path}")

    # Print classification report
    from sklearn.metrics import classification_report
    report = classification_report(true_labels, pred_labels, target_names=classes)
    print(f"Classification Report for {model_name}:")
    print(report)

print("\nAll confusion matrices have been saved to the plots folder!")