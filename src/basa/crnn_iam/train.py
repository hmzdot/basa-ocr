import torch
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    collate_fn,
    train_dataset,
    val_dataset,
    vocab,
    len_train,
    len_val,
)
from .model import CRNN
from .ctc_decode import ctc_greedy_decode
from ..utils import Tracker


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run(img_height=32, epochs=20, batch_size=32):
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    t = Tracker(run_name=run_name)
    device = detect_device()
    print(f"Using device: {device}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    model = CRNN(
        height=img_height,
        in_chans=1,
        num_classes=len(vocab),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Register model and optimizer for checkpointing
    t.register("model", model)
    t.register("optimizer", optimizer)

    best_accuracy = 0.0
    for epoch in range(epochs):
        pbar = tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch}",
            total=len_train,
        )
        for i, batch in pbar:
            if i == 100:
                break
            imgs = batch["imgs"].to(device)
            labels = batch["labels"].to(device)

            out = model(imgs)  # N, T, C
            out = out.permute(1, 0, 2)  # T, N, C
            T, N, C = out.shape

            probs = F.log_softmax(out, dim=2)  # T, N

            optimizer.zero_grad()

            input_lengths = torch.full((N,), T, dtype=torch.long, device=device)
            target_lengths = (labels != vocab.blank_token).sum(dim=1)

            loss = F.ctc_loss(probs, labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                pbar.set_postfix_str(f"Loss: {loss.item():.2f}")
                step = i + epoch * len_train
                t.log(step, train_loss=loss.item())
                t.plot("train_loss")

        correct_letters = 0
        total_letters = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc="Validation", total=len_val):
                imgs = batch["imgs"].to(device)
                labels = batch["labels"].to(device)
                max_letters = labels.size(1)

                logits = model(imgs)  # N, T, C
                preds = torch.argmax(logits, dim=2)  # N, T
                preds = ctc_greedy_decode(
                    preds,
                    max_letters,
                    vocab.blank_token,
                )  # N, MAX_LETTERS

                B, N = labels.shape
                mask = labels != vocab.blank_token  # B, N

                # Letter accuracy: count correct non-blank predictions
                correct_letters += int(((preds == labels) & mask).sum().item())
                total_letters += int(mask.sum().item())

                # Word accuracy: all non-blank positions must be correct
                # For each sample, check if (correct OR is_blank) for ALL positions
                word_correct = ((preds == labels) | ~mask).all(dim=1)  # B,
                total_correct += int(word_correct.sum().item())
                total_samples += B

            print(f"Total correct: {total_correct}")
            print(f"Total samples: {total_samples}")
            print(
                f"Letter acc: {(correct_letters * 100 / total_letters):.2f}%"
                f" ({correct_letters}/{total_letters})"
            )

            accuracy = total_correct * 100 / total_samples
            t.log(epoch, val_accuracy=accuracy)
            t.plot("val_accuracy")
            print(f"Accuracy: {accuracy:.3f}%")

            # Save checkpoint
            t.save_logs()
            t.save(epoch=epoch, is_best=accuracy > best_accuracy, keep_last=3)
            if accuracy > best_accuracy:
                best_accuracy = accuracy

            model.train()


if __name__ == "__main__":
    run()
