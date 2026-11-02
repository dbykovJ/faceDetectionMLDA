import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, precision_recall_curve, \
    balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms

from imbalanced import ImbalancedDatasetSampler


def get_transforms(transformations):
    test_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=25, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=128),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.3, fill=128),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, shear=(-8, 8, -8, 8)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    if not transformations:
        return test_tf, test_tf
    return train_tf, test_tf

def build_train_data(train_dir, hard_negative_dir, tf):
    """ImageFolder (+ optional hard negatives) using transform `tf`. Order is deterministic."""
    base = torchvision.datasets.ImageFolder(train_dir, transform=tf)
    assert base.class_to_idx == {'0': 0, '1': 1}, base.class_to_idx
    if hard_negative_dir is None:
        return base  # already has .targets

    from FixedLabelFolder import FixedLabelFolder
    hard_tf = transforms.Compose([transforms.Grayscale(), transforms.Resize((36, 36)), tf])
    hard = FixedLabelFolder(hard_negative_dir, label=0, transform=hard_tf)
    data = ConcatDataset([base, hard])
    data.targets = base.targets + hard.targets
    return data

def prep_data(train_dir: str, test_dir: str, transformations, hard_negative_dir: str | None = None):
    train_tf, test_tf = get_transforms(transformations)
    train_data= build_train_data(train_dir, hard_negative_dir, train_tf)
    train_targets = train_data.targets

    get_label = lambda ds, idx: train_targets[idx]
    train_sampler = ImbalancedDatasetSampler(
        train_data,
        indices=list(range(len(train_data))),
        callback_get_label=get_label)
    train_loader = DataLoader(train_data, batch_size=32, sampler=train_sampler, num_workers=4)

    test_data = torchvision.datasets.ImageFolder(test_dir, transform=test_tf)
    tv_idx, tf_idx = train_test_split(
        np.arange(len(test_data)), test_size=0.5,
        stratify=np.array(test_data.targets), random_state=0)
    test_val_loader = DataLoader(Subset(test_data, tv_idx), batch_size=64)
    test_final_loader = DataLoader(Subset(test_data, tf_idx), batch_size=64)

    return train_loader, test_val_loader, test_final_loader

def to_logit(out):
    """Any model output -> one logit per sample, shape (N,)."""
    if out.dim() == 1:                 # already (N,)
        return out
    out = out.flatten(1)               # (N,1,1,1) -> (N,1) ; (N,2) stays (N,2)
    if out.shape[1] == 2:              # 2-class head: logit of "face" vs "noface"
        return out[:, 1] - out[:, 0]
    return out.squeeze(1)              # 1-logit head

@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(to_logit(model(x.to(device)).flatten())).cpu())
        ys.append(y)
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def metrics_at(p, y, thr=0.5):
    pred = (p >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average='binary', zero_division=0)
    return {'precision': prec, 'recall': rec, 'f1': f1,
            'pr_auc': average_precision_score(y, p)}


def best_threshold(p, y):
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    return float(thr[f1s.argmax()])

def evaluate(model, loader, criterion, device):
    model.eval()
    probs, ys, total_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = to_logit(model(x).flatten())                       # (N,1,1,1) -> (N,)
            total_loss += criterion(out, y.float()).item() * y.size(0)
            n += y.size(0)
            probs.append(torch.sigmoid(out).cpu())         # sigmoid, not softmax
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

def show_errors(model, loader, device, threshold, classes, n=16):
    def unnormalize(img):
        # undo Normalize(0.5, 0.5) so pixels are back in [0, 1]
        return (img * 0.5 + 0.5).clamp(0, 1)

    model.eval()
    wrong = []
    with torch.no_grad():
        for images, labels in loader:
            p_face = torch.sigmoid(to_logit(model(images.to(device)).flatten())).cpu()
            preds = (p_face >= threshold).long()
            probs = torch.stack([1 - p_face, p_face], dim=1)
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
    plt.savefig('errors_matrix.png')
    plt.show()


def show_confusion_matrix(model, loader, device, threshold, classes):
    model.eval()
    p, y = get_probs(model, loader, device)
    pred = (p >= threshold).astype(int)
    cm = torch.zeros(2, 2, dtype=torch.int64)
    for t, q in zip(y, pred):
        cm[t, q] += 1
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1], classes)
    ax.set_yticks([0, 1], classes)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j].item()), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    print(evaluate(model, loader, criterion=nn.BCEWithLogitsLoss(), device=device))



