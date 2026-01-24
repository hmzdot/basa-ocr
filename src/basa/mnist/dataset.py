from torchvision import datasets, transforms

train_dataset = datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=(transforms.ToTensor()),
)

val_dataset = datasets.MNIST(
    root="data",
    download=True,
    train=False,
    transform=(transforms.ToTensor()),
)
