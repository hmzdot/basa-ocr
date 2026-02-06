import os
import torch
import numpy as np
import string
from PIL import Image
from torch.utils.data import Dataset


class FixedWordsDataset(Dataset):
    image_list: list[str]

    def __init__(self, data_dir: str, lexicon, img_size=80):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.lexicon = lexicon
        self.image_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(self.data_dir, image_name)
        img = Image.open(image_path).resize((self.img_size, self.img_size))
        img = np.array(img).transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        chars_to_idx = {v: i for (i, v) in enumerate(self.lexicon)}
        label_str = image_name.split("_")[1].split(".")[0]
        label = torch.tensor([chars_to_idx[l] for l in label_str])
        return img, label


dataset = FixedWordsDataset(
    data_dir="data/even_words_random/",
    lexicon=string.ascii_letters + string.digits,
)
len_train = int(len(dataset) * 0.8)
len_val = int(len(dataset) - len_train)
train_dataset, val_dataset = random_split(
    dataset, lengths=(len_train, len_val), generator=torch.Generator()
)
