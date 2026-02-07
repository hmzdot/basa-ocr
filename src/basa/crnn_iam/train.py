import torch
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import collate_fn_split, train_dataset, val_dataset, vocab
from .model import CRNN
from .ctc_decode import ctc_greedy_decode
from ..utils import Tracker


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lev_distance_ids(hyp: list[int], ref: list[int]) -> int:
    if hyp == ref:
        return 0
    if len(ref) == 0:
        return len(hyp)
    if len(hyp) == 0:
        return len(ref)

    prev = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, start=1):
        cur = [i] + [0] * len(ref)
        for j, r in enumerate(ref, start=1):
            cost = 0 if h == r else 1
            cur[j] = min(
                prev[j] + 1,  # deletion
                cur[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # subst/match
            )
        prev = cur
    return prev[-1]


def run(img_height=32, epochs=20, batch_size=32):
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    t = Tracker(run_name=run_name)
    device = detect_device()
    print(f"Using device: {device}")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn_split("train"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_split("val"),
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
            total=len(train_loader),
        )
        for i, batch in pbar:
            imgs = batch["imgs"].to(device)
            labels = batch["labels"].to(device)
            img_widths = batch["img_widths"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            out = model(imgs)  # N, T, C
            out = out.permute(1, 0, 2)  # T, N, C
            T, N, C = out.shape

            probs = F.log_softmax(out, dim=2)  # T, N

            optimizer.zero_grad()

            input_lengths = (img_widths // 4).clamp(min=1, max=T)
            target_lengths = label_lengths

            loss = F.ctc_loss(
                probs,
                labels,
                input_lengths,
                target_lengths,
                zero_infinity=True,
            )
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                pbar.set_postfix_str(f"Loss: {loss.item():.2f}")
                step = i + epoch * len(train_loader)
                t.log(step, train_loss=loss.item())
                t.plot("train_loss")

        total_correct = 0
        total_samples = 0

        cer_edits = 0
        cer_ref_chars = 0

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc="Validation", total=len(val_loader)):
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

                preds_cpu = preds.detach().cpu()
                labels_cpu = labels.detach().cpu()

                # Calculate CER on each batch
                for i in range(B):
                    hyp = preds_cpu[i][preds_cpu[i] != vocab.blank_token].tolist()
                    ref = labels_cpu[i][labels_cpu[i] != vocab.blank_token].tolist()

                    if hyp == ref:
                        total_correct += 1

                    cer_edits += lev_distance_ids(hyp, ref)
                    cer_ref_chars += len(ref)

                total_samples += B

            cer = cer_edits / max(1, cer_ref_chars)
            print(f"CER: {cer:.2f}")

            accuracy = total_correct * 100 / total_samples
            t.log(epoch, val_accuracy=accuracy)
            t.plot("val_accuracy")

            print(f"Total correct: {total_correct}")
            print(f"Total samples: {total_samples}")
            print(f"Accuracy: {accuracy:.3f}%")

            # Save checkpoint
            t.save_logs()
            t.save(epoch=epoch, is_best=accuracy > best_accuracy, keep_last=3)
            if accuracy > best_accuracy:
                best_accuracy = accuracy

            model.train()


if __name__ == "__main__":
    run()
