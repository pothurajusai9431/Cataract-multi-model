import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import cv2  # needed for blur detection


def load_classes(data_dir):
    # Always return fixed binary classes
    return ["Cataract", "Normal"]


def load_data(data_dir, batch_size=64, blur_threshold=50, filter_blur=True):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For RGB images
    ])

    # helper to detect blur
    def is_blurry(img_path, threshold=blur_threshold):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        return cv2.Laplacian(img, cv2.CV_64F).var() < threshold

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Train"), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Test"), transform=transform)

    print(f"[load_data] original train samples: {len(train_dataset.samples)}")
    print(f"[load_data] original val   samples: {len(val_dataset.samples)}")
    
    
    
    if filter_blur:
        filtered_train = []
        for path, label in train_dataset.samples:
            if not is_blurry(path):
                filtered_train.append((path, label))
        train_dataset.samples = filtered_train

        filtered_val = []
        for path, label in val_dataset.samples:
            if not is_blurry(path):
                filtered_val.append((path, label))
        val_dataset.samples = filtered_val

        print(f"[load_data] filtered train samples: {len(train_dataset.samples)}")
        print(f"[load_data] filtered val   samples: {len(val_dataset.samples)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, ["Cataract", "Normal"]  # fixed classes


def visualize_samples(dataset, class_names, num_samples=5):
    print("inside func")
    indices = torch.randint(0, len(dataset), (num_samples,))
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 5))

    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image = image.numpy().transpose((1, 2, 0))
        image = (image * 0.5) + 0.5

        axs[i].imshow(image)
        axs[i].set_title(class_names[label])
        axs[i].axis("off")
    print("at plt ")
    plt.show()

train_loader, val_loader, class_names = load_data("Dataset", batch_size=64 )
# # fill params of below function
visualize_samples(train_loader.dataset, class_names)