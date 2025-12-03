"""
LeNet-5 implementation for CS461 Homework 4 (Part 1.1).

Required packages (install in your virtualenv):
    pip install torch torchvision

This file defines:
- A scaled hyperbolic tangent activation (as in Appendix A of LeCun et al. 1998).
- Subsampling layers S2 and S4 with trainable scale and bias per feature map.
- The LeNet-5 convolutional architecture (C1, S2, C3, S4, C5, F6).
- An RBF output layer whose centers are constructed from 7x12 digit bitmaps.

The training loop and evaluation code for Parts 1.2 and 1.3 will be added later.
"""
from __future__ import annotations
import numpy

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: scaled hyperbolic tangent activation
# ---------------------------------------------------------------------------


class ScaledTanh(nn.Module):
    """
    Scaled hyperbolic tangent activation used in LeNet-5.

    The original paper uses f(a) = A * tanh(S * a) with A = 1.7159 and S = 2/3.
    This choice gives an output range close to [-1.7159, 1.7159] and a slope
    close to 1 at the origin.
    """

    def __init__(self, A: float = 1.7159, S: float = 2.0 / 3.0) -> None:
        super().__init__()
        self.A = float(A)
        self.S = float(S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.A * torch.tanh(self.S * x)


# ---------------------------------------------------------------------------
# Utility: subsampling (S2, S4)
# ---------------------------------------------------------------------------


class Subsampling(nn.Module):
    """
    Subsampling layer used for S2 and S4 in LeNet-5.

    Each feature map is processed as:
        y = activation( w * avg_pool_2x2(x) + b )

    where w and b are learned per-feature-map scale and bias parameters.

    In the original LeNet-5, subsampling uses non-overlapping 2x2 windows
    (stride 2) and a squashing nonlinearity after the affine transform.
    """

    def __init__(self, num_maps: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # One learnable scale and bias per feature map.
        self.weight = nn.Parameter(torch.ones(num_maps))
        self.bias = nn.Parameter(torch.zeros(num_maps))
        self.activation = ScaledTanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.pool(x)
        # Reshape parameters for broadcasting: (1, C, 1, 1)
        w = self.weight.view(1, -1, 1, 1)
        b = self.bias.view(1, -1, 1, 1)
        x = x * w + b
        return self.activation(x)


# ---------------------------------------------------------------------------
# RBF output layer
# ---------------------------------------------------------------------------


class RBFOutputLayer(nn.Module):
    """
    Euclidean RBF output layer as described in Section II-C of LeNet-5.

    Each unit i has a fixed center (prototype) vector c_i in R^84 and the
    output is the squared Euclidean distance between the input x and c_i:

        y_i = ||x - c_i||^2

    A smaller y_i means x is closer to prototype i (better match).
    """

    def __init__(self, centers: torch.Tensor) -> None:
        """
        Args:
            centers: Tensor of shape (num_classes, D) containing the
                RBF centers, where D is typically 84 (flattened 7x12).
                This tensor is registered as a buffer so it is moved
                with the module between devices but is not trainable.
        """
        super().__init__()
        if centers.dim() != 2:
            raise ValueError(f"centers must have shape (num_classes, D), got {centers.shape}")
        self.num_classes, self.dim = centers.shape
        # Register as buffer so it is part of the state dict but not a parameter.
        self.register_buffer("centers", centers.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            x: Tensor of shape (batch_size, D).

        Returns:
            Tensor of shape (batch_size, num_classes) with squared distances.
        """
        if x.dim() != 2 or x.size(1) != self.dim:
            raise ValueError(f"Expected input of shape (batch_size, {self.dim}), got {x.shape}")
        # Compute squared Euclidean distance to each center:
        # ||x - c_i||^2 = sum_j (x_j - c_ij)^2
        x_expanded = x.unsqueeze(1)  # (B, 1, D)
        c_expanded = self.centers.unsqueeze(0)  # (1, C, D)
        diff = x_expanded - c_expanded
        return torch.sum(diff * diff, dim=2)  # (B, C)


# ---------------------------------------------------------------------------
# Helper: build RBF centers from DIGIT data
# ---------------------------------------------------------------------------


def _to_tensor(arr: "torch.Tensor | Sequence | 'numpy.ndarray'") -> torch.Tensor:
    """
    Best-effort conversion of various array-like inputs to a float Tensor.
    """
    if isinstance(arr, torch.Tensor):
        return arr.float()
    try:
        import numpy as np  # type: ignore[import]
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float()
    except Exception:
        # numpy may not be available, fall back to generic conversion
        pass
    return torch.tensor(arr, dtype=torch.float32)


def build_rbf_centers_from_digit(
    digit_data: Sequence,
    num_classes: int = 10,
    height: int = 7,
    width: int = 12,
    foreground_value: float = 1.0,
    background_value: float = -1.0,
) -> torch.Tensor:
    """
    Construct RBF centers from the provided DIGIT data.

    The homework provides a DIGIT dataset that encodes stylized digit
    bitmaps. There are many ways to obtain a single 7x12 bitmap per digit.
    This helper implements a simple and robust strategy:

    1. Interpret `digit_data` as a sequence indexed by class label (0..9).
       Each entry may itself be:
           - a single bitmap, or
           - a collection of candidate bitmaps for that class.
    2. Convert each bitmap to shape (height, width). If necessary, this
       function will:
           - reshape 1D arrays of length height*width to (height, width);
           - transpose arrays of shape (width, height) to (height, width).
    3. If multiple bitmaps are available for a class, compute their
       elementwise mean and then binarize by thresholding at 0.5.
       This acts as a "majority vote" per pixel and yields a single
       stylized bitmap per digit.
    4. Map 0 -> background_value (typically -1) and 1 -> foreground_value
       (typically +1), so the prototype values roughly match the range
       of the F6 activations.
    5. Flatten each 7x12 bitmap to a vector of length 84 and stack
       them into a tensor of shape (num_classes, 84).

    Args:
        digit_data: Sequence containing per-class bitmap data. The exact
            structure depends on the homework's `DIGIT` object, but this
            function is written to handle several common cases (single
            bitmap per class or list of bitmaps per class).
        num_classes: Number of digit classes to use (default: 10).
        height: Desired template height (default: 7).
        width: Desired template width (default: 12).
        foreground_value: Value to assign to foreground pixels (default: +1).
        background_value: Value to assign to background pixels (default: -1).

    Returns:
        Tensor of shape (num_classes, height * width) with values in
        {background_value, foreground_value}.
    """

    templates: List[torch.Tensor] = []
    target_length = height * width

    if len(digit_data) < num_classes:
        raise ValueError(
            f"digit_data must contain at least {num_classes} entries, got {len(digit_data)}"
        )

    # Process each digit independently
    for digit in range(num_classes):
        entry = digit_data[digit]

        # Entry might be a single bitmap or a collection.
        # Normalize to a list of tensors.
        if isinstance(entry, (list, tuple)):
            bitmaps = [_to_tensor(b) for b in entry]
        else:
            bitmaps = [_to_tensor(entry)]

        # Normalize shapes and accumulate.
        norm_bitmaps: List[torch.Tensor] = []
        for bm in bitmaps:
            if bm.dim() == 1 and bm.numel() == target_length:
                bm = bm.view(height, width)
            elif bm.dim() == 2:
                h, w = bm.shape
                if h == height and w == width:
                    pass  # already correct
                elif h == width and w == height:
                    bm = bm.t()
                else:
                    # If the bitmap has a different resolution, resize with
                    # nearest-neighbor interpolation to (height, width).
                    bm = bm.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                    bm = F.interpolate(
                        bm, size=(height, width), mode="nearest"
                    ).squeeze(0).squeeze(0)
            else:
                # Attempt to flatten and reshape
                bm = bm.view(-1)
                if bm.numel() != target_length:
                    raise ValueError(
                        f"Cannot reshape bitmap for digit {digit}: "
                        f"got {bm.numel()} elements, expected {target_length}"
                    )
                bm = bm.view(height, width)

            norm_bitmaps.append(bm)

        # If multiple bitmaps exist, average them and threshold.
        if len(norm_bitmaps) == 1:
            agg = norm_bitmaps[0]
        else:
            stacked = torch.stack(norm_bitmaps, dim=0)
            agg = stacked.mean(dim=0)
        # Threshold at 0.5 assuming original values in [0, 1] or {0,1}.
        binary = (agg >= 0.5).float()

        # Map to {-1, +1} (or custom values).
        proto = torch.where(
            binary > 0.0,
            torch.tensor(foreground_value, dtype=torch.float32),
            torch.tensor(background_value, dtype=torch.float32),
        )
        templates.append(proto.flatten())

    centers = torch.stack(templates, dim=0)  # (num_classes, height*width)
    return centers


# ---------------------------------------------------------------------------
# LeNet-5 network
# ---------------------------------------------------------------------------


class LeNet5(nn.Module):
    """
    LeNet-5 architecture with an RBF output layer.

    The architecture follows Section II of LeCun et al. (1998) with the
    standard modern simplification that C3 is implemented as a regular
    5x5 convolution over all input feature maps (rather than the original
    hand-crafted partial connectivity pattern).

    Input:
        - Grayscale image of shape (batch_size, 1, 32, 32)

    Output:
        - RBF penalties of shape (batch_size, num_classes)
          (smaller value = better match).
    """

    def __init__(self, rbf_centers: torch.Tensor) -> None:
        super().__init__()
        if rbf_centers.dim() != 2 or rbf_centers.size(1) != 84:
            raise ValueError(
                f"rbf_centers must have shape (num_classes, 84), got {rbf_centers.shape}"
            )
        self.num_classes = rbf_centers.size(0)

        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.s2 = Subsampling(num_maps=6)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = Subsampling(num_maps=16)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.f6 = nn.Linear(in_features=120, out_features=84)

        self.activation = ScaledTanh()
        self.rbf = RBFOutputLayer(rbf_centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass through LeNet-5.

        Args:
            x: Tensor of shape (batch_size, 1, 32, 32) containing
               grayscale input images padded/embedded to 32x32.

        Returns:
            Tensor of shape (batch_size, num_classes) with RBF penalties.
        """
        if x.dim() != 4 or x.size(1) != 1 or x.size(2) != 32 or x.size(3) != 32:
            raise ValueError(
                f"Expected input of shape (batch_size, 1, 32, 32), got {x.shape}"
            )

        x = self.c1(x)
        x = self.activation(x)
        x = self.s2(x)

        x = self.c3(x)
        x = self.activation(x)
        x = self.s4(x)

        x = self.c5(x)
        x = self.activation(x)

        # C5 output is (batch_size, 120, 1, 1); flatten to (batch_size, 120)
        x = x.view(x.size(0), -1)
        x = self.f6(x)
        x = self.activation(x)

        # F6 output is (batch_size, 84)
        penalties = self.rbf(x)
        return penalties


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_lenet5_from_digit(
    digit_data: Sequence,
    num_classes: int = 10,
) -> LeNet5:
    """
    Convenience factory that builds LeNet-5 using RBF centers derived
    from the DIGIT data structure provided by the homework.

    Args:
        digit_data: DIGIT-like data structure encoding stylized bitmaps.
        num_classes: Number of digit classes (default: 10).

    Returns:
        A LeNet5 instance ready for training.
    """
    centers = build_rbf_centers_from_digit(digit_data, num_classes=num_classes)
    model = LeNet5(centers)
    return model


# ---------------------------------------------------------------------------
# Simple self-test (shape check)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """
    Run a quick shape check to verify that the implementation is consistent
    with the LeNet-5 architecture. This function does *not* train the model;
    it simply constructs a dummy network with random RBF centers and feeds
    a random batch of images through it.
    """
    batch_size = 4
    dummy_centers = torch.randn(10, 84)
    model = LeNet5(dummy_centers)

    x = torch.randn(batch_size, 1, 32, 32)
    y = model(x)
    assert y.shape == (batch_size, 10), f"Unexpected output shape: {y.shape}"
    print("Self-test passed. Output shape:", y.shape)


if __name__ == "__main__":
    _self_test()
