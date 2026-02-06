import torch
import torch.optim as optim
import torch.nn.functional as F
import string
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import VarWordsDataset
from .model import CRNN
from .ctc_decode import ctc_greedy_decode
from ..utils import Tracker


def run(
    max_letters=10,
    img_height=32,
    blank_letter=0,
    epochs=20,
    batch_size=16,
    data_dir="data/var_words/",
):
    lexicon = string.ascii_letters + string.digits
    dataset = VarWordsDataset(
        data_dir=data_dir,
        max_letters=max_letters,
        lexicon=lexicon,
        blank_letter=blank_letter,
    )
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

    model = CRNN(height=img_height, num_classes=len(lexicon))
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Register model and optimizer for checkpointing
    t.register("model", model)
    t.register("optimizer", optimizer)

    best_accuracy = 0.0
    for epoch in range(epochs):
        pbar = tqdm(
            enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader)
        )
        for i, (imgs, labels) in pbar:
            out = model(imgs)  # N, T, C
            out = out.permute(1, 0, 2)  # T, N, C
            T, N, C = out.shape

            probs = F.log_softmax(out, dim=2)  # T, N
            input_lengths = torch.full((N,), T, dtype=torch.long)

            target_lengths = (labels != blank_letter).sum(dim=1)

            optimizer.zero_grad()

            loss = F.ctc_loss(probs, labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                pbar.set_postfix_str(f"Loss: {loss.item():.2f}")
                step = i + epoch * len(train_loader)
                t.log(step, train_loss=loss.item())
                t.plot("train_loss")

        correct_letters = 0
        total_letters = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            model.eval()
            for imgs, labels in tqdm(val_loader, desc="Validation"):
                logits = model(imgs)  # N, T, C
                preds = torch.argmax(logits, dim=2)  # N, T
                preds = ctc_greedy_decode(
                    preds,
                    max_letters,
                    blank_letter,
                )  # N, MAX_LETTERS

                B, N = labels.shape
                mask = labels != blank_letter  # B, N

                # Letter accuracy: count correct non-blank predictions
                correct_letters += ((preds == labels) & mask).sum()
                total_letters += mask.sum()

                # Word accuracy: all non-blank positions must be correct
                # For each sample, check if (correct OR is_blank) for ALL positions
                word_correct = ((preds == labels) | ~mask).all(dim=1)  # B,
                total_correct += word_correct.sum()
                total_samples += B

            print(f"Total correct: {total_correct}")
            print(f"Total samples: {total_samples}")
            print(
                f"Letter acc: {correct_letters * 100 / total_letters:.2f}%"
                f" ({correct_letters}/{total_samples * len(lexicon)})"
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
