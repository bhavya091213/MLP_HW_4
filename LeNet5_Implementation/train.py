"""
Training script for CS461 Homework 4 (LeNet-5).

Part 1.2:
    - Train LeNet-5 on MNIST for 20 epochs (batch_size=1, lr=1e-3)
    - Track training error each epoch
    - Plot training error vs epoch to 1.2_training_error_rates.png

Part 1.3:
    - Also track test error each epoch
    - Save both training and test error arrays to 1.3_error_rates.npz
    - Save the trained model as LeNet1.pth (state_dict) for test1.py
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project root and imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from LeNet5_Implementation.main import create_lenet5_from_digit  # type: ignore


# ---------------------------------------------------------------------------
# Reproducibility / device helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int = 0) -> None:
    """Set random seeds for reproducible runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def get_mnist_loaders(
    batch_size: int = 1,
    data_dir: str | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for MNIST.

    28x28 images are zero-padded to 32x32 and mapped from [0,1] to [-1,1]
    to match the LeNet-5 input size and activation scale.
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data")

    transform = transforms.Compose(
        [
            transforms.Pad(2),                # 28x28 -> 32x32
            transforms.ToTensor(),            # [0,1]
            transforms.Normalize((0.5,), (0.5,)),  # -> [-1,1]
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # num_workers=0 and pin_memory=False to keep macOS / MPS happy
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Build digit templates (the "DIGIT" data) from MNIST itself
# ---------------------------------------------------------------------------


def build_digit_templates_from_mnist(
    num_per_class: int = 200,
    data_dir: str | None = None,
):
    """
    Build per-digit prototype images from the MNIST training data.

    For each digit k in {0,...,9}, we:
        - collect up to `num_per_class` training images with label k,
        - convert them to 28x28 float tensors in [0,1],
        - pass the list of tensors for each digit to create_lenet5_from_digit,
          which will take care of resizing to 7x12, averaging, thresholding,
          and mapping to {-1, +1}.

    Returns:
        digit_data: a list of length 10 where element k is a list of
                    2D tensors (28x28) representing digit k.
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data")

    template_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transforms.ToTensor()
    )

    digit_images: List[List[torch.Tensor]] = [[] for _ in range(10)]

    for img, label in template_dataset:
        label_int = int(label)
        if 0 <= label_int <= 9 and len(digit_images[label_int]) < num_per_class:
            digit_images[label_int].append(img.squeeze(0))  # (28, 28)
        if all(len(lst) >= num_per_class for lst in digit_images):
            break

    return digit_images


# ---------------------------------------------------------------------------
# Loss (Eq. 9)
# ---------------------------------------------------------------------------


def lenet_rbf_loss(
    penalties: torch.Tensor, targets: torch.Tensor, j: float = 0.1
) -> torch.Tensor:
    """
    LeNet-5 loss from Eq. (9):

        E_p = y_{D^p} + log( exp(-j) + sum_i exp(-y_i) )

    where penalties y_i are squared distances from the RBF layer.
    """
    if penalties.dim() != 2:
        raise ValueError(f"penalties must have shape (B, C), got {penalties.shape}")
    if targets.dim() != 1 or targets.size(0) != penalties.size(0):
        raise ValueError(
            f"targets must have shape (B,), got {targets.shape}, "
            f"while penalties has shape {penalties.shape}"
        )

    batch_indices = torch.arange(penalties.size(0), device=penalties.device)
    y_correct = penalties[batch_indices, targets]  # (B,)

    log_term = torch.log(
        torch.exp(torch.tensor(-j, device=penalties.device))
        + torch.sum(torch.exp(-penalties), dim=1)
    )

    loss = y_correct + log_term
    return loss.mean()


# ---------------------------------------------------------------------------
# Training loop (records train + test error for 1.3)
# ---------------------------------------------------------------------------


def train_lenet5(
    num_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 1,
    seed: int = 0,
) -> Tuple[List[float], List[float]]:
    """
    Train LeNet-5 on MNIST with online SGD and Eq. (9) loss.

    Returns:
        (train_errors, test_errors): lists of length num_epochs containing
        the training and test error rates at the end of each epoch.
    """
    set_seed(seed)
    device = get_device()

    data_dir = os.path.join(PROJECT_ROOT, "data")

    digit_data = build_digit_templates_from_mnist(num_per_class=200, data_dir=data_dir)
    model = create_lenet5_from_digit(digit_data).to(device)

    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, data_dir=data_dir)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_errors: List[float] = []
    test_errors: List[float] = []

    for epoch in range(1, num_epochs + 1):
        # -------------------- Train --------------------
        model.train()
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            penalties = model(images)
            loss = lenet_rbf_loss(penalties, labels)
            loss.backward()
            optimizer.step()

            preds = penalties.argmin(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_error = 1.0 - (correct_train / total_train)
        train_errors.append(train_error)

        # -------------------- Test ---------------------
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                penalties = model(images)
                preds = penalties.argmin(dim=1)

                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

        test_error = 1.0 - (correct_test / total_test)
        test_errors.append(test_error)

        print(
            f"Epoch {epoch:2d}/{num_epochs}: "
            f"train error = {train_error:.4f}, "
            f"test error = {test_error:.4f}"
        )

    # ------------------------------------------------------------------
    # Save model state_dict for hw4_files/test1.py
    # ------------------------------------------------------------------
    state_dict = model.state_dict()

    impl_dir = os.path.dirname(os.path.abspath(__file__))
    hw4_files_dir = os.path.join(PROJECT_ROOT, "hw4_files")

    os.makedirs(hw4_files_dir, exist_ok=True)

    torch.save(state_dict, os.path.join(impl_dir, "LeNet1.pth"))
    torch.save(state_dict, os.path.join(hw4_files_dir, "LeNet1.pth"))
    print(f"Saved model state_dict to {impl_dir}/LeNet1.pth and hw4_files/LeNet1.pth")

    # ------------------------------------------------------------------
    # Part 1.2: training-error-only plot
    # ------------------------------------------------------------------
    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_errors, label="Train error")
    plt.xlabel("Epoch")
    plt.ylabel("Error rate")
    plt.title("LeNet-5 Training Error Rate (Part 1.2)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    train_fig_path = os.path.join(impl_dir, "1.2_training_error_rates.png")
    plt.savefig(train_fig_path, dpi=200)
    plt.close()
    print(f"Saved 1.2 training error plot to {train_fig_path}")

    # ------------------------------------------------------------------
    # Part 1.3: save both train and test errors for later plotting
    # ------------------------------------------------------------------
    error_path = os.path.join(PROJECT_ROOT, "1.3_error_rates.npz")
    np.savez(
        error_path,
        train_errors=np.array(train_errors, dtype=np.float32),
        test_errors=np.array(test_errors, dtype=np.float32),
    )
    print(f"Saved train/test error history to {error_path}")

    return train_errors, test_errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    train_lenet5()


if __name__ == "__main__":
    main()
