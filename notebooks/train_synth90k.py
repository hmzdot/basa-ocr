import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.optim as optim
    import torch.nn.functional as F
    import string
    from datetime import datetime
    from torch.utils.data import DataLoader, random_split
    from tqdm import tqdm

    from basa.crnn_v2.dataset import train_loader, val_loader, vocab, len_train, len_val
    from basa.crnn_v2.model import CRNN
    from basa.crnn_v2.ctc_decode import ctc_greedy_decode

    img_height = 32

    model = CRNN(height=img_height, num_classes=len(vocab))
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    return F, model, optimizer, torch, train_loader, vocab


@app.cell
def _(train_loader):
    batch = next(iter(train_loader))
    return (batch,)


@app.cell
def _(batch, model):
    imgs, labels = batch["images"], batch["labels"]
    print(imgs.shape, labels.shape)

    out = model(imgs)  # N, T, C
    print(out.shape)
    return labels, out


@app.cell
def _(batch):
    input_lengths = batch["image_widths"]
    target_lengths = batch["label_lengths"]

    print(input_lengths.shape, target_lengths.shape)
    return


@app.cell
def _(F, labels, optimizer, out, torch, vocab):
    out_v2 = out.permute(1, 0, 2)  # T, N, C
    T, N, C = out_v2.shape

    probs = F.log_softmax(out_v2, dim=2)  # T, N, C
    print('probs', probs.shape)
    print('labels', labels.shape) 

    input_lengths_v2 = torch.full((N,), T, dtype=torch.long)
    print(input_lengths_v2.shape)

    target_lengths_v2 = (labels != vocab.blank_token).sum(dim=1)
    print(target_lengths_v2.shape)

    optimizer.zero_grad()

    loss = F.ctc_loss(probs, labels, input_lengths_v2, target_lengths_v2)
    loss.item()
    return


if __name__ == "__main__":
    app.run()
