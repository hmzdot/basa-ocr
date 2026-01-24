import torch
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader

from .dataset import train_dataset, val_dataset
from .model import CNN
from ..utils import Tracker


def run():
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    t = Tracker(run_name=run_name)

    model = CNN()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

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


if __name__ == "__main__":
    run()
