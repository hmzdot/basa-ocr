import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class VarWordsDataset(Dataset):
    image_list: list[str]

    def __init__(
        self,
        data_dir: str,
        max_letters: int,
        lexicon,
        blank_letter=0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.max_letters = max_letters
        self.lexicon = lexicon
        self.blank_letter = blank_letter
        self.image_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(self.data_dir, image_name)

        # Pad the image from the right
        img_raw = Image.open(image_path)
        _w, h = img_raw.size
        max_w = self.max_letters * 16
        img = Image.new(img_raw.mode, (max_w, h), (255, 255, 255))
        img.paste(img_raw, (0, 0))
        img = np.array(img).transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        label = torch.full(
            (self.max_letters,),
            self.blank_letter,
            dtype=torch.long,
        )
        chars_to_idx = {v: (i + 1) for (i, v) in enumerate(self.lexicon)}
        label_str = image_name.split("_")[1].split(".")[0]
        label_ids = torch.tensor([chars_to_idx[l] for l in label_str])
        label[torch.arange(len(label_str))] = label_ids
        return img, label
