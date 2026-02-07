from torchvision import datasets, transforms

train_dataset = datasets.EMNIST(
    split="balanced",
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ToTensor(),
        ]
    ),
)

val_dataset = datasets.EMNIST(
    split="balanced",
    root="data",
    train=False,
    download=True,
    transform=(transforms.ToTensor()),
)
