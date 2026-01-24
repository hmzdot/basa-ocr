import string
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from tqdm import tqdm

from .dataset import FixedWordsDataset
from .model import CNN_v3
from ..utils import Tracker


def run(
    num_letters=5,
    batch_size=16,
    epochs=20,
    data_dir="data/even_words_random",
):
    lexicon = string.ascii_letters + string.digits

    # Prepare dataset
    dataset = FixedWordsDataset(data_dir=data_dir, lexicon=lexicon)
    len_train = int(len(dataset) * 0.8)
    len_val = int(len(dataset) - len_train)
    train_dataset, val_dataset = random_split(
        dataset, lengths=(len_train, len_val), generator=torch.Generator()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    t = Tracker(run_name=run_name)

    model = CNN_v3(num_letters=num_letters, len_letters=len(lexicon))
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    t.register("model", model)
    t.register("optimizer", optimizer)

    for epoch in range(epochs):
        pbar = tqdm(
            enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader)
        )
        for i, (imgs, labels) in pbar:
            B, N = labels.shape
            out = model(imgs)  # B, N, C
            out = out.reshape(B * N, -1)  # B x N, C
            labels = labels.reshape(B * N)  # B x N

            optimizer.zero_grad()

            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                pbar.set_postfix_str(f"Loss: {loss.item():.2f}")
                step = i + epoch * len(train_loader)
                t.log(step, train_loss=loss.item())
                t.plot("train_loss")

        correct_letters = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            model.eval()
            for imgs, labels in tqdm(val_loader, desc="Validation"):
                logits = model(imgs)  # B, N, C
                preds = torch.argmax(logits, dim=2)  # B, N
                correct_letters += (preds == labels).sum()
                total_correct += (preds == labels).all(dim=1).sum()
                total_samples += labels.shape[0]

            print(f"Total correct: {total_correct}")
            print(f"Total samples: {total_samples}")
            print(
                f"Letter acc: {correct_letters * 100 / total_samples / num_letters:.2f}%"
                f" ({correct_letters}/{total_samples * num_letters})"
            )

            accuracy = total_correct * 100 / total_samples
            t.log(epoch, val_accuracy=accuracy)
            t.plot("val_accuracy")
            print(f"Accuracy: {accuracy:.3f}%")

            # Save checkpoint
            t.save_logs()
            t.save(epoch=epoch, is_best=False, keep_last=3)

            model.train()


if __name__ == "__main__":
    run()
