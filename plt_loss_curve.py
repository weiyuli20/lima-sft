"""Plot the two LIMA SFT training-loss logs in one figure."""

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
lr1e_5_loss_path = ROOT / "wandb/run-20261019_141225-jtwieoif/files/output.log"
lr2e_4_loss_path = ROOT / "wandb/run-20261019_162319-n6dt4weq/files/output.log"


def read_losses(log_path):
    """Extract (epoch, loss) pairs from Hugging Face Trainer log lines."""
    epochs, losses = [], []
    for line in Path(log_path).read_text().splitlines():
        try:
            record = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if "epoch" in record and "loss" in record:
            epochs.append(record["epoch"])
            losses.append(record["loss"])
    return epochs, losses


def plot_losses(lr1e_5_path, lr2e_4_path, output_path):
    """Save the learning-rate comparison loss plot to output_path."""
    epochs_1e_5, losses_1e_5 = read_losses(lr1e_5_path)
    epochs_2e_4, losses_2e_4 = read_losses(lr2e_4_path)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs_1e_5, losses_1e_5, marker="o", markersize=3, label="lr = 1e-5")
    axis.plot(epochs_2e_4, losses_2e_4, marker="o", markersize=3, label="lr = 2e-4")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training loss")
    axis.set_title("LIMA SFT Training Loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    plot_losses(
        lr1e_5_loss_path,
        lr2e_4_loss_path,
        ROOT / "lima_sft_loss_comparison.png",
    )
