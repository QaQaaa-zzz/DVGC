#!/usr/bin/env python3
"""Render measured pilot evidence and explicitly empty final-result templates.

Uses saved JSON only. Templates contain no fabricated points, curves or error
bars. The displayed axis ranges and category labels specify the planned plot.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
SOURCE = PAPER.parent / "evidence/retention_next_20261010/T1_paired_ppo_002/summary.json"
COLORS = ["#66737E", "#187D9C", "#D68D2D", "#54976C", "#8C6AAD"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "axes.titlesize": 11, "figure.dpi": 160,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    fig.savefig(HERE / (name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / (name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pending(ax, note, note_y=.36):
    ax.text(.5, .58, "RESULTS PENDING", ha="center", va="center", transform=ax.transAxes,
            fontsize=15, weight="bold", color="#6A7680",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#BFC8D0"))
    ax.text(.5, note_y, note, ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color="#69737C", linespacing=1.5)


def main():
    data = json.loads(SOURCE.read_text())
    methods = ["PPO + keep", "PPO + demo + keep"]
    keys = ["PPO_keep", "PPO_demo_keep"]
    # Every plotted value is derived from the compact immutable pilot summary.
    train = [data["arms"][key]["stable_train"] for key in keys]
    dev = [data["source_DEV"]["stable_success"]] + [data["arms"][key]["DEV"]["stable_success"] for key in keys]
    assert train == [1, 1] and dev == [35, 56, 54]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
    axs[0].bar(np.arange(2), np.array(train) / 8 * 100, color=[COLORS[1], COLORS[2]], width=.5)
    axs[0].set_xticks(np.arange(2), ["PPO\n+ keep", "PPO + demo\n+ keep"])
    axs[0].set_title("(a) Independent recovery on TRAIN roots", loc="left", pad=15)
    for i, n in enumerate(train):
        axs[0].text(i, n / 8 * 100 + 4, f"{n}/8", ha="center", weight="bold")
    axs[1].bar(np.arange(3), np.array(dev) / 64 * 100, color=COLORS[:3], width=.5)
    axs[1].set_xticks(np.arange(3), ["Initial policy", "PPO\n+ keep", "PPO + demo\n+ keep"])
    axs[1].set_title("(b) Complete-task development conditions", loc="left", pad=15)
    for i, n in enumerate(dev):
        axs[1].text(i, n / 64 * 100 + 3, f"{n}/64", ha="center", weight="bold")
    for ax in axs:
        ax.set_ylim(0, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Stable success (%)")
        ax.yaxis.grid(True, color="#E6EBEF", linewidth=.7)
        ax.set_axisbelow(True)
    fig.text(.5, -.015, "Measured pilot: one training seed; 32,000 transitions per arm; success in all three fixed repetitions.",
             ha="center", fontsize=9)
    fig.tight_layout(w_pad=2)
    save(fig, "pilot_stable_success")

    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.5))
    names = ["M0", "M1", "M2", "M3", "M4"]
    for ax, title in zip(axs, ["(a) Unseen-condition performance", "(b) Retention versus the initial policy"]):
        ax.set_title(title, loc="left")
        ax.set_xlim(-.5, 4.5)
        ax.set_xticks(range(5), names)
        ax.yaxis.grid(True, color="#E5E9ED")
    axs[0].set_ylim(0, 100)
    axs[0].set_ylabel("Macro stable success (%)")
    axs[1].set_ylim(-15, 15)
    axs[1].set_ylabel("Paired success-rate difference (pp)")
    axs[1].axhline(-5, color="#B26543", linestyle="--", linewidth=1)
    axs[1].text(4.38, -5.5, "Predeclared −5 pp margin", ha="right", va="top", fontsize=8, color="#995237")
    pending(axs[0], "Three independent outer-loop seeds\nPaired condition-level estimates and intervals")
    pending(axs[1], "Separate estimates for random-initial\nand combined-perturbation conditions", note_y=.17)
    fig.text(.5, -.025, "EMPTY FIGURE TEMPLATE — no performance estimates are shown.", ha="center", fontsize=9, weight="bold")
    fig.tight_layout(w_pad=2)
    save(fig, "template_main_comparison")

    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.5))
    for ax, xlabel, title in zip(axs, ["Charged environment control steps", "Measured GPU wall-clock time (h)"],
                                ["(a) Interaction efficiency", "(b) Computational efficiency"]):
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel("Development stable success (%)")
        ax.set_xlabel(xlabel)
        ax.yaxis.grid(True, color="#E5E9ED")
        pending(ax, "M0–M4: five predeclared candidates\nShow each training seed and paired aggregate")
    fig.text(.5, -.025, "EMPTY FIGURE TEMPLATE — horizontal limits will follow measured cost; no synthetic curves.", ha="center", fontsize=9, weight="bold")
    fig.tight_layout(w_pad=2)
    save(fig, "template_learning_curves")

    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.axis("off")
    rows = ["Initial policy", "M0, seed 1", "M0, seed 2", "M0, seed 3", "M1, seed 1", "M1, seed 2", "M1, seed 3"]
    cols = ["Model", "Source engine", "Target engine", "Paired difference"]
    cells = [[name, "PENDING / 33", "PENDING / 33", "PENDING"] for name in rows]
    table = ax.table(cellText=cells, colLabels=cols, cellLoc="center", loc="center", colWidths=[.25, .25, .25, .25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.75)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CFD7DF")
        if r == 0:
            cell.set_facecolor("#EAF0F4")
            cell.set_text_props(weight="bold")
        elif c > 0:
            cell.set_text_props(color="#7A858E")
    ax.set_title("Same-model sim-to-sim evaluation: identical endpoint and action interface", fontsize=12, weight="bold", pad=12)
    fig.text(.5, .015, "RESULTS PENDING — 33 fixed conditions × 3 repetitions per engine; zero target-policy training.",
             ha="center", fontsize=9, weight="bold")
    save(fig, "template_sim2sim")
    manifest = {
        "schema": "jit_english_manuscript_figures_v1", "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "measured_figures": ["pilot_stable_success"],
        "empty_templates": ["template_main_comparison", "template_learning_curves", "template_sim2sim"],
        "synthetic_measurements": False, "new_physical_steps": 0, "new_optimizer_updates": 0,
        "pilot_counts": {"TRAIN_stable": train, "DEV_stable_pi0_control_demo": dev,
                         "TRAIN_denominator": 8, "DEV_denominator": 64, "training_seeds": 1,
                         "fixed_evaluation_repetitions": 3},
    }
    (HERE / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
