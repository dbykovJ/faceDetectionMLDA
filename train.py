"""
    Training a face recognition model
    Usage:
        python train.py
    Arguments:
        -f, --model_file: file with the model class
        -m, --model_class: model class name
        -n, --name: name for saving the model
        -tr, --train-folder: path to training folder
        -te, --test-folder: path to test folder
        -hn, --hard-negatives-folder: path to hard negatives folder (optional)
        -cv, --cross-validate: perform cross-validation (optional)
        -t, --transformations: perform more complex transformations (optional)
"""


import argparse
from pkgutil import resolve_name
from helpers.train_help import *
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchmetrics import Accuracy
from tqdm import tqdm
from imbalanced import ImbalancedDatasetSampler

def cross_validate(model_class, device,
                   train_dir, transformations, hard_negative_dir=None,
                   num_epochs=10, k_folds=3, batch_size=32):
    train_tf, test_tf = get_transforms(transformations)
    aug_data = build_train_data(train_dir, hard_negative_dir, train_tf)    # for training
    clean_data = build_train_data(train_dir, hard_negative_dir, test_tf)   # for validation
    labels = np.array(aug_data.targets)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (tr_ids, val_ids) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n===== Fold {fold + 1}/{k_folds} =====")

        sampler = ImbalancedDatasetSampler(
            aug_data, indices=tr_ids.tolist(),
            callback_get_label=lambda ds, idx: labels[idx])
        train_loader = DataLoader(aug_data, batch_size=batch_size, sampler=sampler, num_workers=4)
        val_loader = DataLoader(Subset(clean_data, val_ids.tolist()), batch_size=64)

        # fresh model + optimizer per fold, same setup as train()
        model = model_class().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.25, device=device))

        for epoch in range(num_epochs):
            model.train()
            running = 0.0
            for data, y in tqdm(train_loader, desc=f"Fold {fold + 1} Epoch {epoch + 1}"):
                data, y = data.to(device), y.to(device)
                loss = criterion(to_logit(model(data)), y.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()
            scheduler.step()

            p, yv = get_probs(model, val_loader, device)
            thr = best_threshold(p, yv)
            m = metrics_at(p, yv, thr)
            m['train_loss'] = running / len(train_loader)
            print(f"  epoch {epoch + 1}: train_loss={m['train_loss']:.4f} thr={thr:.2f} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f} PR-AUC={m['pr_auc']:.4f}")

        fold_results.append(m)  # last epoch, so no best-epoch selection bias

    print("\n===== Cross-validation summary (mean ± std over folds) =====")
    for key in ['precision', 'recall', 'f1', 'pr_auc']:
        vals = [r[key] for r in fold_results]
        print(f"{key:>10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    return fold_results

def train(model_class, device, train_loader, test_val_loader, num_epochs = 10):

    print("\n===== Start Training Phase =====")
    model = model_class().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.25, device=device))  # weight for class imbalance

    best_f1, best_thr = 0.0, 0.5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            data, targets = data.to(device), targets.to(device)
            loss = criterion(to_logit(model(data)), targets.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # ---- validation ----
        p, y = get_probs(model, test_val_loader, device)

        # the training process automatically calculates the best threshold value for deciding whether an image is a face
        thr = best_threshold(p, y)

        m = metrics_at(p, y, thr)
        print(f"epoch {epoch}: train_loss={running_loss / len(train_loader):.4f} "
              f"thr={thr:.2f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} PR-AUC={m['pr_auc']:.4f}")

        if m['f1'] > best_f1:  # keep best epoch
            best_f1, best_thr = m['f1'], thr
            torch.save(model.state_dict(), 'best.pt')

    model.load_state_dict(torch.load('best.pt'))
    model.eval()
    return model, best_thr

def test(model, device, test_loader, threshold=0.5):
    acc = Accuracy(task="binary").to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            probs = torch.sigmoid(to_logit(model(images).flatten())).cpu()
            preds = (probs >= threshold).long()
            acc.update(preds, labels)
    print(f"Test accuracy @ thr={threshold:.2f}: {acc.compute().item():.4f}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--model_file", required=True, help="file with the model class")
    ap.add_argument("-m", "--model_class", required=True, help="model class name")
    ap.add_argument("-n", "--name", required=True, help="name for saving the model")
    ap.add_argument("-tr", "--train-folder", required=True, help="path to training folder")
    ap.add_argument("-te", "--test-folder", required=True, help="path to test folder")
    ap.add_argument("-hn", "--hard-negatives-folder", required=False, help="path to hard negatives folder")
    ap.add_argument("-cv", "--cross-validate", action="store_true", help="perform cross-validation")
    ap.add_argument("-t", "--transformations", action="store_true", help="perform more complex transformations" )

    return ap.parse_args()

def main():
    classes = ('noface', 'face')  # indicates that "1" means "face" and "0" non-face (only used for display)
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    num_epochs = 10

    args = parse_args()

    model_name = args.model_class
    model_class = resolve_name(args.model_file + ':' + args.model_class)

    train_loader, test_val_loader, test_final_loader = prep_data(
        train_dir=args.train_folder,
        test_dir=args.test_folder,
        hard_negative_dir=args.hard_negatives_folder,
        transformations=args.transformations,
    )

    if args.cross_validate:
        cross_validate(model_class, device, args.train_folder, args.transformations,
                       args.hard_negatives_folder, num_epochs=num_epochs)

    model, threshold = train(model_class = model_class, device = device,
                  train_loader = train_loader, test_val_loader = test_val_loader,
                  num_epochs=num_epochs)

    test(model=model, device=device, test_loader=test_final_loader)

    show_confusion_matrix(model=model, loader=test_final_loader, device=device, threshold=threshold, classes=classes)
    show_errors(model=model, loader=test_final_loader, device=device, threshold=threshold, classes=classes)

    torch.save(model.state_dict(), "./" + model_name + ".pt")

if __name__ == "__main__":
    main()
