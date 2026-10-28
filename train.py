import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import SubsetRandomSampler
from torchmetrics import Accuracy
from torchvision import transforms
from tqdm import tqdm

from imbalanced import ImbalancedDatasetSampler

train_dir = "./deep_learning_project/train_images"  # folder containing training images
test_dir = "./deep_learning_project/test_images"  # folder containing test images

transform = transforms.Compose(
    [
        transforms.Grayscale(),  # transforms to gray-scale (1 input channel)
        transforms.ToTensor(),  # transforms to Torch tensor (needed for PyTorch)
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ]
)  # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)

# Define two pytorch datasets (train/test)
train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

valid_size = 0.2  # proportion of validation set (80% train, 20% validation)
batch_size = 32

# Define randomly the indices of examples to use for training and for validation
num_train = len(train_data)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

# Define two "samplers" that will randomly pick examples from the training and validation set
train_sampler = ImbalancedDatasetSampler(train_data, indices=train_new_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Dataloaders (take care of loading the data from disk, batch by batch, during training)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1
)
valid_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True, num_workers=1
)

classes = (
    "noface",
    "face",
)  # indicates that "1" means "face" and "0" non-face (only used for display)

# Training


def train(model_class, device, num_epochs=10):

    print("\n===== Start Training Phase =====")

    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            data, targets = data.to(device), targets.to(device)
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()

    return model


def test(model, device):
    test_acc_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            test_acc_metric(preds, labels)
    print(f"Test accuracy: {test_acc_metric.compute().item():.4f}")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            prob, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % (100 * correct / total)
    )
