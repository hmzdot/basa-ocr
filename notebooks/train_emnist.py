import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from torchvision import datasets, transforms

    train_dataset = datasets.EMNIST(
        download=True,
        root="data",
        split="balanced",
        train=True,
        transform=(transforms.ToTensor()),
    )
    return (train_dataset,)


@app.cell
def _(train_dataset):
    from torch.utils.data import DataLoader
    from PIL import Image
    import marimo as mo

    from tqdm import tqdm

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    from random import random

    pbar = tqdm(train_loader, desc="train")
    for imgs, labels in pbar:
        pbar.set_postfix(dict(loss=random()))
    return


if __name__ == "__main__":
    app.run()
