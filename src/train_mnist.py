from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_utils import Tracker

train_dataset = datasets.MNIST(
    root="data", download=True, train=True, transform=(transforms.ToTensor())
)
val_dataset = datasets.MNIST(
    root="data", download=True, train=False, transform=(transforms.ToTensor())
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.inner = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.inner(x)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 512),
        )
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        return self.classifier(self.features(x))


run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
t = Tracker(run_name=run_name)

model = Classifier()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Register model and optimizer for checkpointing
t.register("model", model)
t.register("optimizer", optimizer)

for i, (imgs, labels) in enumerate(train_loader):
    logits = model(imgs)

    labels_onehot = torch.zeros((labels.shape[0], 10))
    labels_onehot[torch.arange(16), labels] = 1

    optimizer.zero_grad()
    loss = F.cross_entropy(logits, labels_onehot)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        t.log(i, train_loss=loss.item())
        t.plot("train_loss")


total_correct = 0
total_samples = 0

for imgs, labels in val_loader:
    logits = model(imgs)
    preds = torch.argmax(logits, dim=1)
    total_correct += (preds == labels).sum()
    total_samples += labels.shape[0]

print(f"Total correct: {total_correct}")
print(f"Total samples: {total_samples}")
print(f"Accuracy: {total_correct / total_samples}")
