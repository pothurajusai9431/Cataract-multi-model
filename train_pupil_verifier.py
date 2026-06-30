"""
Small training script to create a pupil/non-pupil verifier.

Usage:
- Prepare a folder structure:
  data/
    pupil/        <-- patches centered on real pupils (64x64 or larger)
    not_pupil/    <-- negative patches (car wheels, hands, backgrounds)

Run:
    .\myenv\Scripts\Activate.ps1
    python train_pupil_verifier.py --data data --epochs 15 --out models/pupil_verifier.pth

This script is a simple example; adjust augmentation and hyperparameters
for your dataset.
"""
from pathlib import Path
import argparse
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os


class PupilVerifierCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def main(args):
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = ImageFolder(args.data, transform=tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    model = PupilVerifierCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y).sum().item()
            total += x.size(0)

        print(f"Epoch {epoch+1}/{args.epochs}: loss={total_loss/total:.4f} acc={correct/total:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("Saved verifier to", args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to dataset root (ImageFolder structure)')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--out', default='models/pupil_verifier.pth')
    args = p.parse_args()
    main(args)
