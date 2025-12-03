from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mnist
import torch
import numpy as np
import torchvision
import os
import sys
import matplotlib.pyplot as plt

 
def test(dataloader,model):

    #please implement your test code#
    ##HERE##

    # ------------------------------------------------------------------
    # Make sure we have access to the LeNet5 code and template builder
    # ------------------------------------------------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from LeNet5_Implementation.main import create_lenet5_from_digit  # type: ignore
    from LeNet5_Implementation.train import build_digit_templates_from_mnist  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Handle the case where LeNet1.pth is a state_dict (dict) *or*
    # a full nn.Module. We support both.
    # ------------------------------------------------------------------
    loaded_obj = model
    if isinstance(loaded_obj, torch.nn.Module):
        model = loaded_obj.to(device)
    elif isinstance(loaded_obj, dict):
        # Rebuild the LeNet-5 architecture and load the saved weights.
        data_dir = os.path.join(project_root, "data")
        digit_data = build_digit_templates_from_mnist(
            num_per_class=200, data_dir=data_dir
        )
        model = create_lenet5_from_digit(digit_data).to(device)
        model.load_state_dict(loaded_obj)
    else:
        raise TypeError(
            "LeNet1.pth is neither a torch.nn.Module nor a state_dict. "
            "Make sure it was saved with torch.save(model.state_dict(), ...) "
            "or torch.save(model, ...)."
        )

    model.eval()

    # ------------------------------------------------------------------
    # Confusion matrix and 'most confusing' examples
    # ------------------------------------------------------------------
    confusion = torch.zeros(10, 10, dtype=torch.int64)  # on CPU

    best_conf = [-1.0 for _ in range(10)]
    best_idx = [-1 for _ in range(10)]
    best_pred = [-1 for _ in range(10)]

    total = 0
    correct = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            # images from mnist.MNIST: (1, 32, 32) with values in [0, 255]
            images = images.to(device)
            labels = labels.to(device)

            # Match training scaling: map [0,255] -> [0,1] -> [-1,1]
            images = images / 255.0
            images = (images - 0.5) / 0.5  # 2x - 1

            penalties = model(images)             # (1, 10)
            logits = -penalties                   # smaller distance -> higher logit
            probs = torch.softmax(logits, dim=1)  # (1, 10)
            preds = probs.argmax(dim=1)           # predicted digit

            true_digit = int(labels[0].item())
            pred_digit = int(preds[0].item())
            confusion[true_digit, pred_digit] += 1

            total += 1
            if pred_digit == true_digit:
                correct += 1
            else:
                # Confidence of this wrong prediction = P(pred_digit)
                conf_value = float(probs[0, pred_digit].item())
                if conf_value > best_conf[true_digit]:
                    best_conf[true_digit] = conf_value
                    best_idx[true_digit] = idx      # test-set index
                    best_pred[true_digit] = pred_digit

    test_accuracy = correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Save confusion matrix plot
    # ------------------------------------------------------------------
    confusion_cpu = confusion  # already on CPU
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion_cpu.numpy(), cmap="Blues")

    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("Confusion Matrix (MNIST Test Set)")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_yticklabels([str(i) for i in range(10)])

    max_val = confusion_cpu.max().item() if confusion_cpu.numel() > 0 else 0
    for i in range(10):
        for j in range(10):
            count = int(confusion_cpu[i, j].item())
            color = "black" if max_val == 0 or count < max_val / 2 else "white"
            ax.text(j, i, str(count), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig("1.3_confusion_matrix.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Save most confusing examples summary
    # ------------------------------------------------------------------
    with open("1.3_most_confusing_examples.txt", "w") as f:
        f.write(
            "Most confusing misclassified examples per true digit "
            "(MNIST test set)\n\n"
        )
        f.write(
            "Format: digit x: image y was misclassified as digit b "
            "with highest confidence a\n\n"
        )
        for digit in range(10):
            if best_idx[digit] == -1:
                f.write(
                    f"digit {digit}: no misclassifications found in the test set\n"
                )
            else:
                f.write(
                    f"digit {digit}: image {best_idx[digit]} was misclassified "
                    f"as digit {best_pred[digit]} with highest confidence "
                    f"{best_conf[digit]:.4f}\n"
                )
        error_file = os.path.join(project_root, "1.3_error_rates.npz")
    if os.path.exists(error_file):
        data = np.load(error_file)
        train_errors = data["train_errors"]
        test_errors = data["test_errors"]
        epochs = np.arange(1, len(train_errors) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_errors, label="Train error")
        plt.plot(epochs, test_errors, label="Test error")
        plt.xlabel("Epoch")
        plt.ylabel("Error rate")
        plt.title("LeNet-5 Training and Test Error Rates (Part 1.3)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("1.3_training_test_error_rates.png", dpi=200)
        plt.close()
        print(
            "Saved training+test error plot to "
            "1.3_training_test_error_rates.png"
        )
    else:
        print(
            f"Warning: {error_file} not found; "
            "cannot plot training+test errors for 1.3."
        )
    ###########################
                                                                                                                                                                             

    print("test accuracy:", test_accuracy)

 

def main():

    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')

    mnist_test=mnist.MNIST(split="test",transform=pad)

    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)

    model = torch.load("LeNet1.pth")

    test(test_dataloader,model)

 

if __name__=="__main__":

    main()
