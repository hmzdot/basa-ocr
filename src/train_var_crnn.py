from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import string
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from tracker import Tracker

DATASET_DIR = "data/var_words/"
IMAGE_HEIGHT = 32
BATCH_SIZE = 16
LETTERS = string.ascii_letters + string.digits
BLANK_TOKEN = 0
LEN_LETTERS = len(LETTERS) + 1
MAX_LETTERS = 10
EPOCHS = 20

class VarWordsDataset(Dataset):
    image_list: list[str]

    def __init__(self):
        super().__init__()
        self.image_list = os.listdir(DATASET_DIR)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(DATASET_DIR, image_name)

        # Pad the image from the right
        img_raw = Image.open(image_path)
        _w, h = img_raw.size
        max_w = MAX_LETTERS * 16
        img = Image.new(img_raw.mode, (max_w, h), (255, 255, 255))
        img.paste(img_raw, (0, 0))
        img = np.array(img).transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        label = torch.full((MAX_LETTERS,), BLANK_TOKEN, dtype=torch.long)
        chars_to_idx = {v: (i+1) for (i, v) in enumerate(LETTERS)}
        label_str = image_name.split("_")[1].split(".")[0]
        label_ids = torch.tensor([chars_to_idx[l] for l in label_str])
        label[torch.arange(len(label_str))] = label_ids
        return img, label


dataset = VarWordsDataset()
len_train = int(len(dataset) * 0.8)
len_val = int(len(dataset) - len_train)
train_dataset, val_dataset = random_split(
    dataset, lengths=(len_train, len_val), generator=torch.Generator()
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


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool: tuple[int, int] | None = None,
    ):
        super().__init__()
        modules = [
            nn.Conv2d(
                in_chans,
                out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
        ]
        if pool:
            modules.append(nn.MaxPool2d(kernel_size=pool, stride=pool))

        self.layer = nn.Sequential(*modules)

    def forward(self, x):
        return self.layer(x)

class CRNN(nn.Module):
    def __init__(
        self,
        height: int,
        num_classes: int,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
    ):
        super().__init__()
        # In: B, 3, H, W
        self.cnn = nn.Sequential(
            ConvLayer(3, 32, pool=(2,2)),   # B, 32, H/2, W/2
            ConvLayer(32, 64, pool=(2,2)),  # B, 64, H/4, W/4 
            ConvLayer(64, 128, pool=(2,1)), # B, 128, H/8, W/4
        )
        # Out: B, 128, H/8, W/4

        # In: B, W/4, 128 * H/8
        self.rnn = nn.LSTM(
            input_size=128 * (height//8),
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )
        # Out: B, W/4, d_h * 2

        # In: B, W/4, d_h * 2
        self.transcription = nn.Linear(rnn_hidden * 2, num_classes)
        # Out: B, W/4, d_c

    def forward(self, x):
        x = self.cnn(x)

        B, C, H, W = x.shape

        x = x.permute(0, 3, 1, 2)       # B, W/8, 128, H/4
        x = x.reshape(B, W, -1)         # B, W/8, 128 * H/4 

        x, _ = self.rnn(x)              # B, W/8, d_h * 2

        x = self.transcription(x)       # B, W/8, d_c
        return x 

def ctc_greedy_decode(x: torch.Tensor) -> torch.Tensor:
    """
    In: x (B, T)
    Out: (B, max_len)
    """
    B, T = x.shape
    out = torch.full((B, MAX_LETTERS), BLANK_TOKEN, dtype=torch.long)
    
    for b in range(B):
        decoded = []
        prev_char = BLANK_TOKEN
        
        for t in range(T):
            cur_char = x[b, t].item()
            if cur_char != BLANK_TOKEN and cur_char != prev_char:
                decoded.append(cur_char)
            prev_char = cur_char
        
        for i, c in enumerate(decoded[:MAX_LETTERS]):
            out[b, i] = c
    
    return out

run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
t = Tracker(run_name=run_name)

model = CRNN(height=IMAGE_HEIGHT, num_classes=LEN_LETTERS)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Register model and optimizer for checkpointing
t.register("model", model)
t.register("optimizer", optimizer)

for epoch in range(EPOCHS):
    pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader))
    for i, (imgs, labels) in pbar:
        out = model(imgs)  # N, T, C
        out = out.permute(1, 0, 2) # T, N, C
        T, N, C = out.shape

        probs = F.softmax(out, dim=2) # T, N
        input_lengths = torch.full((N,), T, dtype=torch.long)

        target_lengths = (labels != BLANK_TOKEN).sum(dim=1)

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
            preds = ctc_greedy_decode(preds) # N, MAX_LETTERS

            B, N = labels.shape
            mask = labels != BLANK_TOKEN  # B, N

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
            f"Letter acc: {correct_letters * 100 / total_samples / LEN_LETTERS:.2f}%"
            f" ({correct_letters}/{total_samples * LEN_LETTERS})"
        )

        accuracy = total_correct * 100 / total_samples
        t.log(epoch, val_accuracy=accuracy)
        t.plot("val_accuracy")
        print(f"Accuracy: {accuracy:.3f}%")

        # Save checkpoint
        t.save_logs()
        t.save(epoch=epoch, is_best=False, keep_last=3)

        model.train()
