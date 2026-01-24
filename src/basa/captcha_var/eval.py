import os
import torch
import shutil
import random
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
    run_name: str,
    checkpoint_name: str,
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

    t.load(run_name, checkpoint_name) 

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        model.eval()

        incorrect = []
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

            incorrect_labels = ((preds != labels) & mask).any(dim=1)
            incorrect.extend(
                list(
                    zip(
                        labels[incorrect_labels, :].cpu().detach().tolist(),
                        preds[incorrect_labels, :].cpu().detach().tolist(),
                    )
                )
            )

            # Word accuracy: all non-blank positions must be correct
            # For each sample, check if (correct OR is_blank) for ALL positions
            word_correct = ((preds == labels) | ~mask).all(dim=1)  # B,
            total_correct += word_correct.sum()
            total_samples += B

        # Sample incorrect ones
        SAMPLE_SIZE = 20
        samples = random.choices(incorrect, k=SAMPLE_SIZE)
        os.makedirs(f"logs/{run_name}/failed", exist_ok=True)
        label_idx_to_char = {i + 1: c for i, c in enumerate(lexicon)}
        label_to_images: dict[str, list[str]] = {}
        for image_name in dataset.image_list:
            label_str = image_name.split("_")[1].split(".")[0]
            label_to_images.setdefault(label_str, []).append(image_name)

        copied = 0
        for sample_label_ids, sample_pred_ids in samples:
            sample_label = "".join(
                label_idx_to_char[idx] for idx in sample_label_ids if idx != blank_letter
            )
            sample_pred = "".join(
                label_idx_to_char[idx] for idx in sample_pred_ids if idx != blank_letter
            )
            matches = label_to_images.get(sample_label, [])
            if not matches:
                continue
            image_name = matches.pop(0)
            src_path = os.path.join(data_dir, image_name)
            if sample_pred:
                base, ext = os.path.splitext(image_name)
                image_name = f"{base}_pred-{sample_pred}{ext}"
            dst_path = os.path.join(f"logs/{run_name}/failed", image_name)
            shutil.copy2(src_path, dst_path)
            copied += 1

        print(f"Incorrect samples: {len(incorrect)}")
        print(f"Copied failed samples: {copied}")

        print(f"Total correct: {total_correct}")
        print(f"Total samples: {total_samples}")

        accuracy = total_correct * 100 / total_samples
        print(f"Accuracy: {accuracy:.3f}%")
