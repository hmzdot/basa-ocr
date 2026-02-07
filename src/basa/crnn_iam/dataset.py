import string
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, random_split


class IAMDataset(Dataset):
    # path -> label
    img_list: list[tuple[str, str]]

    def __init__(
        self,
        img_dir: str,
        words_file: str,
        lexicon,
        img_size=80,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.img_size = img_size
        self.lexicon = lexicon

        with open(words_file) as f:
            self.img_list = self._parse_words_file(f.read())

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        [img_name, label_str] = self.img_list[index]
        path_parts = img_name.split("-")
        img_dir0 = path_parts[0]
        img_dir1 = f"{path_parts[0]}-{path_parts[1]}"

        img_path = os.path.join(
            self.img_dir,
            img_dir0,
            img_dir1,
            f"{img_name}.png",
        )
        print(img_path)
        img = Image.open(img_path)
        print(img.size)
        img = np.array(img)
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        chars_to_idx = {v: i for (i, v) in enumerate(self.lexicon)}
        label = torch.tensor([chars_to_idx[li] for li in label_str])
        return img, label

    def _parse_words_file(self, text: str) -> list[tuple[str, str]]:
        items = list()
        lines = text.split("\n")
        for line in lines:
            if line.startswith("#"):
                continue
            parts = line.split(" ")
            items.append((parts[0], parts[-1]))
        return items


dataset = IAMDataset(
    img_dir="data/iam_handwriting/iam_words/words",
    words_file="data/iam_handwriting/words_new.txt",
    lexicon=string.ascii_letters + string.digits,
)
len_train = int(len(dataset) * 0.8)
len_val = int(len(dataset) - len_train)
train_dataset, val_dataset = random_split(
    dataset, lengths=(len_train, len_val), generator=torch.Generator()
)

print(dataset[10])
