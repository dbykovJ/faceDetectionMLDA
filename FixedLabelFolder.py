from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class FixedLabelFolder(Dataset):
    """All images in a folder (recursively), every one with the same label."""

    def __init__(self, folder, label=0, transform=None):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.files = sorted(p for p in Path(folder).rglob("*")
                            if p.is_file() and p.suffix.lower() in exts)
        self.label = label
        self.transform = transform
        self.targets = [label] * len(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label