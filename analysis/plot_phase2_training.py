#!/usr/bin/env python3
"""Plot Phase 2 training and Phase 3 contrastive training curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
P2_LOG = REPO_ROOT / "data" / "processed" / "checkpoints" / "training_log.json"
P3_LOG = REPO_ROOT / "data" / "processed" / "checkpoints" / "contrastive_log.json"
OUT = REPO_ROOT / "analysis" / "figures"


def plot_phase2(ax):
    if not P2_LOG.exists():
        ax.set_title("(no Phase 2 log)")
        return
    log = json.loads(P2_LOG.read_text())
    eps = [e["epoch"] for e in log["epochs"]]
    train = [e["train_loss"] for e in log["epochs"]]
    val   = [e["val_loss"]   for e in log["epochs"]]
    ax.plot(eps, train, label="train", color="steelblue")
    ax.plot(eps, val,   label="val",   color="darkred")
    best_ep = min(log["epochs"], key=lambda e: e["val_loss"])
    ax.scatter([best_ep["epoch"]], [best_ep["val_loss"]], s=80, c="gold",
               edgecolor="black", zorder=5,
               label=f"best (ep {best_ep['epoch']}, val={best_ep['val_loss']:.3f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Phase 2 — Coptic→Syriac transformer\n"
                 f"({log['args']['d_model']}d × {log['args']['n_layers']}L, "
                 f"peak_lr={log['args']['lr']:.0e})")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_phase3(ax):
    if not P3_LOG.exists():
        ax.set_title("(no Phase 3 log yet)")
        return
    log = json.loads(P3_LOG.read_text())
    eps = [e["epoch"] for e in log["epochs"]]
    train_acc = [e["train_acc"] for e in log["epochs"]]
    val_acc = [e["val_acc"] for e in log["epochs"]]
    alpha_max = [e.get("alpha_max", 0) for e in log["epochs"]]
    train_loss = [e["train_loss"] for e in log["epochs"]]
    val_loss = [e["val_loss"]   for e in log["epochs"]]

    ax2 = ax.twinx()
    ax.plot(eps, train_loss, label="train loss", color="steelblue")
    ax.plot(eps, val_loss,   label="val loss",   color="darkred")
    ax2.plot(eps, train_acc, "--", label="train acc", color="steelblue", alpha=0.5)
    ax2.plot(eps, val_acc,   "--", label="val acc",   color="darkred", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("InfoNCE loss")
    ax2.set_ylabel("Accuracy (acc_a2c + acc_c2a) / 2")
    ax.set_title("Phase 3 — Syriac strophe contrastive model\n"
                 "Loss = lower better; Accuracy = higher better")
    ax2.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_phase2(a1)
    plot_phase3(a2)
    plt.tight_layout()
    fig.savefig(OUT / "phase2_phase3_training.png", dpi=140, bbox_inches="tight")
    fig.savefig(OUT / "phase2_phase3_training.pdf", bbox_inches="tight")
    print(f"Wrote {OUT}/phase2_phase3_training.png")


if __name__ == "__main__":
    main()
