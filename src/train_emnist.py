"""
Best accuracy: 89.79%

Ablations:
- Without data augmentation: 89.27%
- Without ReLU and dropout: 88.83%
- With batch size of 16: 88.44%
"""

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from train_utils import Tracker

BATCH_SIZE = 16
NUM_CLASSES = 47
EPOCHS = 15


train_dataset = datasets.EMNIST(
    split="balanced",
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ToTensor(),
        ]
    ),
)
val_dataset = datasets.EMNIST(
    split="balanced",
    root="data",
    train=False,
    download=True,
    transform=(transforms.ToTensor()),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
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
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        return self.classifier(self.features(x))


run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
t = Tracker(run_name=run_name)

model = Classifier()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Register model and optimizer for checkpointing
t.register("model", model)
t.register("optimizer", optimizer)

for epoch in range(EPOCHS):
    pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader))
    for i, (imgs, labels) in pbar:
        logits = model(imgs)

        optimizer.zero_grad()
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            pbar.set_postfix_str(f"Loss: {loss.item():.2f}")
            t.log(i, train_loss=loss.item())
            t.plot("train_loss")

    total_correct = 0
    total_samples = 0

    model.eval()
    for imgs, labels in tqdm(val_loader, desc="Validation"):
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum()
        total_samples += labels.shape[0]

    print(f"Total correct: {total_correct}")
    print(f"Total samples: {total_samples}")

    accuracy = total_correct * 100 / total_samples
    t.log(epoch, val_accuracy=accuracy)
    t.plot("val_accuracy")
    print(f"Accuracy: {accuracy:.2f}%")

    # Save checkpoint
    t.save(epoch=epoch, is_best=False, keep_last=3)

    model.train()
