"""
Test script for Part 2 (Robust LeNet with STN).

This script:
1. Loads the trained LeNet2.pth model
2. Evaluates on standard MNIST test set
3. Generates confusion matrix
4. Identifies most confusing misclassified examples per digit
5. Plots training history if available
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mnist
import torch
import numpy as np
import torchvision
import os
import sys
import matplotlib.pyplot as plt


def test(dataloader, model):
    """
    Test the robust LeNet model on MNIST test set.
    
    Args:
        dataloader: DataLoader for test data
        model: Loaded model (either nn.Module or state_dict)
    """
    # ------------------------------------------------------------------
    # Setup paths and imports
    # ------------------------------------------------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Line 28 - Replace:

    from LeNet5_Q2.robust_model import create_robust_lenet  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Handle model loading (state_dict or full module)
    # ------------------------------------------------------------------
    loaded_obj = model
    if isinstance(loaded_obj, torch.nn.Module):
        model = loaded_obj.to(device)
    elif isinstance(loaded_obj, dict):
        # Rebuild the RobustLeNetSTN architecture and load weights
        model = create_robust_lenet(use_stn=True).to(device)
        model.load_state_dict(loaded_obj)
        print("✓ Loaded model from state_dict")
    else:
        raise TypeError(
            "LeNet2.pth is neither a torch.nn.Module nor a state_dict. "
            "Make sure it was saved with torch.save(model.state_dict(), ...) "
            "or torch.save(model, ...)."
        )

    model.eval()

    # ------------------------------------------------------------------
    # Evaluation: confusion matrix and most confusing examples
    # ------------------------------------------------------------------
    confusion = torch.zeros(10, 10, dtype=torch.int64)  # on CPU

    # Track most confusing (highest confidence misclassification) per digit
    best_conf = [-1.0 for _ in range(10)]
    best_idx = [-1 for _ in range(10)]
    best_pred = [-1 for _ in range(10)]

    total = 0
    correct = 0

    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            # images from mnist.MNIST: (1, 32, 32) with values in [0, 255]
            images = images.to(device)
            labels = labels.to(device)

            # Match training preprocessing: [0, 255] -> [0, 1] -> [-1, 1]
            images = images / 255.0
            images = (images - 0.5) / 0.5

            # Forward pass (RobustLeNetSTN outputs logits directly)
            logits = model(images)  # (1, 10)
            probs = torch.softmax(logits, dim=1)  # (1, 10)
            preds = probs.argmax(dim=1)  # predicted digit

            true_digit = int(labels[0].item())
            pred_digit = int(preds[0].item())
            
            # Update confusion matrix
            confusion[true_digit, pred_digit] += 1

            # Track accuracy
            total += 1
            if pred_digit == true_digit:
                correct += 1
            else:
                # For misclassifications, track the most confident one per digit
                conf_value = float(probs[0, pred_digit].item())
                if conf_value > best_conf[true_digit]:
                    best_conf[true_digit] = conf_value
                    best_idx[true_digit] = idx
                    best_pred[true_digit] = pred_digit

            # Progress indicator
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(dataloader)} samples...")

    test_accuracy = correct / total if total > 0 else 0.0
    error_rate = 1.0 - test_accuracy

    print(f"\n{'='*60}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Error Rate: {error_rate*100:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Save confusion matrix plot
    # ------------------------------------------------------------------
    print("Generating confusion matrix...")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion.numpy(), cmap="Blues")

    ax.set_xlabel("Predicted digit", fontsize=12)
    ax.set_ylabel("True digit", fontsize=12)
    ax.set_title("Confusion Matrix - Robust LeNet (Part 2)", fontsize=14, fontweight='bold')

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_yticklabels([str(i) for i in range(10)])

    # Add text annotations
    max_val = confusion.max().item() if confusion.numel() > 0 else 0
    for i in range(10):
        for j in range(10):
            count = int(confusion[i, j].item())
            color = "white" if count > max_val / 2 else "black"
            ax.text(j, i, str(count), ha="center", va="center", 
                   color=color, fontsize=10, fontweight='bold')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    
    output_path = "2.1_confusion_matrix.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved confusion matrix to {output_path}")

    # ------------------------------------------------------------------
    # Save most confusing examples summary
    # ------------------------------------------------------------------
    print("Saving most confusing examples...")
    output_txt = "2.1_most_confusing_examples.txt"
    with open(output_txt, "w") as f:
        f.write("Most Confusing Misclassified Examples - Robust LeNet (Part 2)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Overall Test Accuracy: {test_accuracy*100:.2f}%\n")
        f.write(f"Overall Test Error Rate: {error_rate*100:.2f}%\n\n")
        f.write("Format: digit X: image Y was misclassified as digit Z "
                "with confidence C\n\n")
        f.write("-"*70 + "\n\n")
        
        for digit in range(10):
            if best_idx[digit] == -1:
                f.write(
                    f"Digit {digit}: ✓ No misclassifications found in test set\n"
                )
            else:
                f.write(
                    f"Digit {digit}: Image {best_idx[digit]:5d} was misclassified "
                    f"as digit {best_pred[digit]} with confidence {best_conf[digit]:.4f}\n"
                )
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Saved most confusing examples to {output_txt}")

    # ------------------------------------------------------------------
    # Plot training history if available
    # ------------------------------------------------------------------
    print("\nLooking for training history...")
    history_file = os.path.join(project_root, "2.1_training_history.npz")
    
    if os.path.exists(history_file):
        print(f"Found training history: {history_file}")
        data = np.load(history_file)
        train_losses = data["train_losses"]
        val_losses = data["val_losses"]
        test_accuracies = data["test_accuracies"]
        epochs = np.arange(1, len(train_losses) + 1)

        # Create comprehensive training plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss curves
        axes[0].plot(epochs, train_losses, label="Train Loss", linewidth=2)
        axes[0].plot(epochs, val_losses, label="Val Loss", linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Test accuracy curve
        test_errors = 1.0 - test_accuracies
        axes[1].plot(epochs, test_errors, label="Test Error Rate", 
                    color='red', linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Error Rate", fontsize=11)
        axes[1].set_title("Test Error Rate Over Training", fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Test accuracy (alternative view)
        axes[2].plot(epochs, test_accuracies * 100, label="Test Accuracy", 
                    color='green', linewidth=2)
        axes[2].set_xlabel("Epoch", fontsize=11)
        axes[2].set_ylabel("Accuracy (%)", fontsize=11)
        axes[2].set_title("Test Accuracy Over Training", fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_ylim([90, 100])

        plt.tight_layout()
        plot_path = "2.1_training_curves.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"✓ Saved training curves to {plot_path}")
    else:
        print(f"Warning: {history_file} not found.")
        print("Cannot plot training history. Run train2.py first.")

    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}\n")

    # Final summary printed to console (as required by original test.py)
    print("test accuracy:", test_accuracy)


def main():
    """Main entry point for testing."""
    # Setup transform (pad 28x28 to 32x32)
    pad = torchvision.transforms.Pad(2, fill=0, padding_mode='constant')

    # Load MNIST test set
    mnist_test = mnist.MNIST(split="test", transform=pad)
    test_dataloader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    # Load trained model
    model_path = "LeNet2.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        print("Please run train2.py first to train the robust model.")
        return
    
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path, map_location='cpu')

    # Run test
    test(test_dataloader, model)


if __name__ == "__main__":
    main()