import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from .dataset import train_dataset, val_dataset
from .model import CNN_v2
from ..utils import Tracker


def run(epochs=20, batch_size=16):
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    t = Tracker(run_name=run_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = CNN_v2(num_classes=47)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    t.register("model", model)
    t.register("optimizer", optimizer)

    for epoch in range(epochs):
        pbar = tqdm(
            enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader)
        )
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


if __name__ == "__main__":
    run()
