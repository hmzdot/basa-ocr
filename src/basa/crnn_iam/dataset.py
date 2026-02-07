import string
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms.v2 as transforms


class Vocabulary:
    def __init__(self):
        self.chars = string.ascii_letters + string.digits + string.punctuation

        self.blank_token = 0
        self.char_to_idx = {ch: i + 1 for (i, ch) in enumerate(self.chars)}
        self.idx_to_char = {i + 1: ch for (i, ch) in enumerate(self.chars)}
        self.idx_to_char[0] = "<blank>"

    def __len__(self):
        return len(self.chars) + 1

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return "".join([self.idx_to_char[idx] for idx in indices if idx != 0])


vocab = Vocabulary()


class IAMDataset(Dataset):
    # path -> label
    img_list: list[tuple[str, str]]

    def __init__(
        self,
        img_dir: str,
        words_file: str,
        lexicon,
        target_height=32,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.target_height = target_height
        self.lexicon = lexicon

        with open(words_file) as f:
            self.img_list = self._parse_words_file(f.read())

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        [img_path, label] = self.img_list[index]
        img = Image.open(img_path)

        # Resize tot target height
        w, h = img.size

        new_w = max(1, int(round(w * (self.target_height / h))))
        img = img.resize((new_w, self.target_height))

        return dict(img=img, label=label)

    def _parse_words_file(self, text: str) -> list[tuple[str, str]]:
        items = list()
        lines = text.split("\n")
        for line in lines:
            if line.startswith("#"):
                continue
            parts = line.split(" ")
            img_name = parts[0]
            path_parts = img_name.split("-")
            img_dir0 = path_parts[0]
            img_dir1 = f"{path_parts[0]}-{path_parts[1]}"

            img_path = os.path.join(
                self.img_dir,
                img_dir0,
                img_dir1,
                f"{img_name}.png",
            )

            # Try to open the image; if it fails, continue
            try:
                Image.open(img_path)
            except Exception:
                continue

            items.append((img_path, parts[-1]))
        return items


def collate_fn_split(split: str):
    if split == "train":
        transform = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.GaussianBlur(kernel_size=3),
                transforms.RandomRotation(degrees=(-10, 10)),
                transforms.RandomPerspective(0.2),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

    def collate_fn(items):
        imgs = [item["img"] for item in items]
        imgs = [transform(img) for img in imgs]
        labels = [item["label"] for item in items]

        img_widths = torch.tensor([img.shape[2] for img in imgs])
        max_width = img_widths.max().item()
        imgs = torch.stack(
            [F.pad(img, (0, max_width - img.size(2)), "constant", 0) for img in imgs],
            dim=0,
        )

        label_lengths = torch.tensor([len(label) for label in labels])
        max_label = int(label_lengths.max().item())
        labels = [
            vocab.encode(label) + [0] * (max_label - len(label)) for label in labels
        ]
        labels = torch.stack([torch.tensor(label) for label in labels])

        return {
            "imgs": imgs,
            "img_widths": img_widths,
            "labels": labels,
            "label_lengths": label_lengths,
        }

    return collate_fn


lexicon = string.ascii_letters + string.digits
dataset = IAMDataset(
    img_dir="data/iam_handwriting/iam_words/words",
    words_file="data/iam_handwriting/words_new.txt",
    lexicon=lexicon,
)
len_train = int(len(dataset) * 0.8)
len_val = int(len(dataset) - len_train)
train_dataset, val_dataset = random_split(
    dataset, lengths=(len_train, len_val), generator=torch.Generator()
)


if __name__ == "__main__":
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        collate_fn=collate_fn_split("train"),
    )
    batch = next(iter(train_loader))

    idx = 20
    img10_w = batch["img_widths"][idx].item()
    img10 = batch["imgs"][idx].squeeze(0).cpu().numpy()
    img10 = (img10 * 255).clip(0, 255).astype("uint8")
    img10_h, _ = img10.shape
    img = Image.fromarray(img10).crop((0, 0, img10_w, img10_h))
    img.show()

    for k, v in batch.items():
        print(f"{k}: {v.shape}")
