import torch
import os
from classification_model import DeepCNN, DeepANN, ResNet, VGG, AlexNet, train_model
from preprocess import load_data, visualize_samples

# Dataset directory
DATA_DIR = os.path.join(os.getcwd(), "Dataset")
EPOCHS = 10
NUM_CLASSES = 2   # ✅ Fixed for binary classification (Normal, Cataract)

# Available models
model_classes = {
    "DeepCNN": DeepCNN,
    "DeepANN": DeepANN,
    "ResNet": ResNet,
    "VGG": VGG,
    "AlexNet": AlexNet
}

# Ask user to select a model
print("Available models:", ", ".join(model_classes.keys()))
MODEL_TYPE = input("Enter the model to train: ").strip()

# Validate model selection
if MODEL_TYPE not in model_classes:
    raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Choose from {list(model_classes.keys())}")

# Load data (disable blur filtering during initial debugging)
# set filter_blur=True once the logic is verified
train_loader, val_loader, classes = load_data(DATA_DIR, filter_blur=False)
visualize_samples(train_loader.dataset, classes, num_samples=5)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize selected model with fixed num_classes
model = model_classes[MODEL_TYPE](num_classes=NUM_CLASSES).to(device)
print(f"Training model: {MODEL_TYPE} with {NUM_CLASSES} classes")

# Train model
train_model(model, train_loader, val_loader, epochs=EPOCHS, device=device)

# Save trained model (with metadata)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"catarct_or_normal{MODEL_TYPE}.pth")

checkpoint = {
    "model_state_dict": model.state_dict(),
    "model_type": MODEL_TYPE,
    "num_classes": NUM_CLASSES,
    "classes": ["Cataract", "Normal"]  # ✅ Explicitly store 2 class names
}

torch.save(checkpoint, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
