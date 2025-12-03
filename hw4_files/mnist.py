import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets


class MNIST(Dataset):
    def __init__(self, split: str = "train", transform=None):
        """
        Custom MNIST wrapper used by the homework.

        Instead of relying on pre-generated PNGs and a label text file,
        this version reads from the standard torchvision MNIST files that
        live under ../data (relative to this hw4_files directory).

        Args:
            split: "train" or "test".
            transform: Optional transform applied to the tensor image
                (expected to accept and return a torch.Tensor).
        """
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        self.split = split
        self.transform = transform

        # project root is one level above hw4_files
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_root = os.path.join(project_root, "data")

        # train=True gives 60k images, train=False gives 10k images.
        self.base = datasets.MNIST(
            root=data_root,
            train=(self.split == "train"),
            download=True,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        """
        Return (image, label) where image is a float tensor of shape
        (1, H, W) with pixel values in [0, 255], matching the original
        homework's expectations before padding and scaling.
        """
        img_pil, label = self.base[idx]  # PIL.Image, int

        # Convert to numpy array with shape (H, W) and values in [0, 255]
        img_np = np.asarray(img_pil, dtype=np.uint8)
        image = torch.from_numpy(img_np).unsqueeze(0).float()  # (1, H, W)

        if self.transform is not None:
            image = self.transform(image)

        return image, int(label)
