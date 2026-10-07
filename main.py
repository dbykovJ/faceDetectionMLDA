import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, roc_auc_score, \
    average_precision_score
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from deep_learning_project.torchsampler.imbalanced import ImbalancedDatasetSampler

train_dir = './deep_learning_project/train_images'    # folder containing training images
test_dir = './deep_learning_project/test_images'    # folder containing test images

transform = transforms.Compose(
    [transforms.Grayscale(),   # transforms to gray-scale (1 input channel)
     transforms.ToTensor(),    # transforms to Torch tensor (needed for PyTorch)
     transforms.Normalize(mean=(0.5,),std=(0.5,))]) # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)

# Define two pytorch datasets (train/test) 
train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

valid_size = 0.2   # proportion of validation set (80% train, 20% validation)
batch_size = 32    

# Define randomly the indices of examples to use for training and for validation
num_train = len(train_data)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

# Define two "samplers" that will randomly pick examples from the training and validation set
train_sampler = ImbalancedDatasetSampler(train_data, indices=train_new_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Dataloaders (take care of loading the data from disk, batch by batch, during training)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

classes = ('noface','face')  # indicates that "1" means "face" and "0" non-face (only used for display)

# Training

class CNNHigh(nn.Module):
    def __init__(self, num_classes=2, lateral_channels=16, grid_size=9):
        super(CNNHigh, self).__init__()

        # Main path (same as CNNMid)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(64)

        # Lateral branches: 1x1 conv to project channels, then pool to a common grid
        self.lat1 = nn.Sequential(  # taps features after conv2 (16 ch)
            nn.Conv2d(16, lateral_channels, kernel_size=1),
            nn.BatchNorm2d(lateral_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(grid_size),
        )
        self.lat2 = nn.Sequential(  # taps features after conv3 (32 ch)
            nn.Conv2d(32, lateral_channels, kernel_size=1),
            nn.BatchNorm2d(lateral_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(grid_size),
        )
        self.lat_main = nn.AdaptiveAvgPool2d(grid_size)  # ensures main path matches grid

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        fused_channels = 64 + 2 * lateral_channels
        self.fc1 = nn.Linear(fused_channels * grid_size * grid_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.act(self.batchnorm1(self.conv1(x)))
        x = self.act(self.batchnorm2(self.conv2(x)))
        f1 = self.lat1(x)                       # lateral feature 1 (low-level)
        x = self.maxpool(x)

        x = self.act(self.batchnorm3(self.conv3(x)))
        f2 = self.lat2(x)                       # lateral feature 2 (mid-level)

        x = self.maxpool(self.act(self.batchnorm4(self.conv4(x))))
        f3 = self.lat_main(x)                   # high-level features

        x = torch.cat([f1, f2, f3], dim=1)      # fuse multi-level features
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNMid(nn.Module):
    def __init__(self):
        super(CNNMid, self).__init__()


        self.conv1 = nn.Conv2d(1, 8, kernel_size = 3, padding = 1)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3, padding = 1)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.batchnorm4 = nn.BatchNorm2d(64)


        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.act = nn.ReLU()

        self.drouput = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64*9*9, 64)
        self.fc2 = nn.Linear(64, 2)


    def forward(self, x):
        x = self.act(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(self.act(self.batchnorm2(self.conv2(x))))
        x = self.act(self.batchnorm3(self.conv3(x)))
        x = self.maxpool(self.act(self.batchnorm4(self.conv4(x))))
        # x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.drouput(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)

        return x


class CNNLow(nn.Module):
    def __init__(self):
        super(CNNLow, self).__init__()


        self.conv1 = nn.Conv2d(1, 8, kernel_size = 3, padding = 1)
        self.batchnorm1 = nn.BatchNorm2d(8)


        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.act = nn.ReLU()


        self.fc1 = nn.Linear(8*18*18, 2)


    def forward(self, x):
        x = self.act(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 6 * 6, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate(model, loader, criterion, device):
        model.eval()
        probs, ys, total_loss, n = [], [], 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                total_loss += criterion(out, y).item() * y.size(0)
                n += y.size(0)
                probs.append(F.softmax(out, dim=1)[:, 1].cpu())
                ys.append(y.cpu())
        p = torch.cat(probs).numpy()
        y = torch.cat(ys).numpy()
        pred = (p > 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, pred, average='binary', zero_division=0)
        return {
            'loss': total_loss / n,
            'acc': float((pred == y).mean()),
            'bal_acc': balanced_accuracy_score(y, pred),
            'precision': prec, 'recall': rec, 'f1': f1,
            'roc_auc': roc_auc_score(y, p),
            'pr_auc': average_precision_score(y, p),
        }


def show_errors(model, loader, device, n=16):
    def unnormalize(img):
        # undo Normalize(0.5, 0.5) so pixels are back in [0, 1]
        return (img * 0.5 + 0.5).clamp(0, 1)
    
    model.eval()
    wrong = []
    with torch.no_grad():
        for images, labels in loader:
            probs = F.softmax(model(images.to(device)), dim=1).cpu()
            preds = probs.argmax(dim=1)
            for i in (preds != labels).nonzero().flatten():
                wrong.append((images[i], labels[i].item(), preds[i].item(), probs[i, preds[i]].item()))
            if len(wrong) >= n:
                break

    wrong = wrong[:n]
    cols = 8
    rows = max(1, (len(wrong) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (img, t, p, conf) in zip(axes.flat, wrong):
        ax.imshow(unnormalize(img).squeeze(), cmap='gray')
        ax.set_title(f'pred: {classes[p]} ({conf:.0%})\ntrue: {classes[t]}', fontsize=9, color='red')
    plt.tight_layout()
    plt.show()

def show_confusion_matrix(model, loader, device):
    model.eval()
    cm = torch.zeros(2, 2, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu()
            for t, p in zip(labels, preds):
                cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1], classes)
    ax.set_yticks([0, 1], classes)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j].item(), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.show()

    print(evaluate(model, loader, criterion=nn.CrossEntropyLoss(), device=device))


def cross_validate(model_class, device, num_epochs = 10):

    model = model_class().to(device)

    k_folds = 3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    targets = np.array(train_data.targets)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results, fold_histories = [], []
    for fold, (train_ids, val_ids) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n===== Fold {fold + 1}/{k_folds} =====")

        train_sampler_cv = ImbalancedDatasetSampler(train_data, indices=train_ids.tolist())
        val_sampler = SubsetRandomSampler(val_ids.tolist())
        train_loader_cv = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, sampler=train_sampler_cv, num_workers=1)
        val_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, sampler=val_sampler, num_workers=1)



        # Fresh model + optimizer per fold (important!)
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        history=[]
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for data, targets in tqdm(train_loader_cv, desc=f"Fold {fold + 1} Epoch {epoch + 1}"):
                data, targets = data.to(device), targets.to(device)
                scores = model(data)
                loss = criterion(scores, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()

            # Validate on this fold's held-out split
            m = evaluate(model, val_loader, criterion, device)
            m['train_loss'] = running_loss / len(train_loader_cv)
            history.append(m)
            print(f"  epoch {epoch + 1}: train_loss {m['train_loss']:.4f} | val_loss {m['loss']:.4f} "
                  f"| P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
                  f"| PR-AUC {m['pr_auc']:.3f}")

        fold_histories.append(history)
        fold_results.append(history[-1])  # metrics after the last epoch


    print("\n===== Cross-validation summary (mean ± std over folds) =====")
    for key in ['acc', 'bal_acc', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'loss']:
        vals = [r[key] for r in fold_results]
        print(f"{key:>10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")



def train(model_class, device, num_epochs = 10):


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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def main(model_class, name):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 10


    # cross_validate(model_class, device, num_epochs=num_epochs)

    model = train(model_class, device, num_epochs=num_epochs)

    test(model, device)

    show_confusion_matrix(model, test_loader, device)
    show_errors(model, test_loader, device)

    torch.save(model.state_dict(), './' + name + '.pt')

if __name__ == '__main__':


    # models = [[CNNLow, 'CNN_low'], [CNNMid, 'CNN_mid'], [CNNHigh, 'CNN_high']]

    models = [[CNNMid, 'CNN_Mid_1'], [Net, 'Net']]
    for model in models:
        main(model[0], model[1])

