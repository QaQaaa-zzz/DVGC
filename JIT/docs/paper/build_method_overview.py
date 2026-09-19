#!/usr/bin/env python3
"""Editable method schematic; synthetic geometry only, no simulator or run data.

Run from any directory. SVG text stays editable; PDF uses embedded TrueType
fonts. Numbered ports connect panels without suggesting a differentiable
simulator. The exported CSV explicitly labels every trajectory as schematic.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np


JIT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = JIT / "runs/paper_figures/method_overview_20260915"
INK = "#243747"
MUTED = "#62727C"
BLUE = "#2673A8"
TEAL = "#248B78"
AMBER = "#BD7B22"
CORAL = "#BE665C"
PURPLE = "#805AA6"
GRAY = "#8D98A0"
PANEL = ["#F0F6FA", "#F0F8F4", "#F6F2F9", "#FBF8F1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.hashsalt": "jit-method-overview", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(18, 12.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 18), ylim=(0, 12.5))
    ax.axis("off")
    text_boxes = []

    def text(x, y, value, size=10, color=INK, bold=False, ha="center", va="center", **kw):
        return ax.text(x, y, value, fontsize=size, color=color, ha=ha, va=va,
                       fontweight="bold" if bold else "normal", linespacing=1.3, **kw)

    def box(x, y, w, h, title, detail="", color=INK, fill="white", size=10.5):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.07",
                              facecolor=fill, edgecolor=color, linewidth=1.1, zorder=3)
        ax.add_patch(patch)
        if detail:
            a = text(x+w/2, y+h*.72, title, size=size, color=color, bold=True, zorder=4)
            b = text(x+w/2, y+h*.32, detail, size=size-1.2, zorder=4)
            text_boxes.extend([(a, patch), (b, patch)])
        else:
            a = text(x+w/2, y+h/2, title, size=size, color=color, bold=True, zorder=4)
            text_boxes.append((a, patch))

    def arrow(points, color=INK, dashed=False, lw=1.25):
        for p, q in zip(points[:-2], points[1:-1]):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw,
                    ls=(0, (4, 3)) if dashed else "-", zorder=2)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=11, linewidth=lw, color=color,
                                    linestyle=(0, (4, 3)) if dashed else "-", zorder=2))

    def port(x, y, number, color=INK):
        ax.add_patch(Circle((x, y), .115, facecolor="white", edgecolor=color, lw=1.2, zorder=6))
        text(x, y-.002, str(number), size=9, color=color, bold=True, zorder=7)

    def panel(x, y, w, h, letter, title, tint):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.13",
                                  fc=tint, ec="#D5DEE3", lw=.85, zorder=0))
        text(x+.23, y+h-.27, f"({letter})  {title}", size=13, bold=True, ha="left")

    text(.35, 12.12, "Iterative Jumping-Tube Expansion through Learned Short-Pulse Exploration",
         size=19, bold=True, ha="left")
    text(.35, 11.74, "Method overview  |  Physical exploration, policy refinement and retained trajectory witnesses",
         size=11.2, color=MUTED, ha="left")
    panel(.30, 6.85, 10.55, 4.55, "a", "Phase-aware short-pulse exploration", PANEL[0])
    panel(11.15, 6.85, 6.55, 4.55, "b", "Same-state continuation and labeling", PANEL[1])
    panel(.30, .92, 10.55, 5.64, "c", "Retention-aware policy improvement", PANEL[2])
    panel(11.15, .92, 6.55, 5.64, "d", "Empirical tube and final controller", PANEL[3])

    # (a) All signal arrows terminate at the receiving block boundary.
    box(.64, 9.50, 1.70, .90, "Observation", "3 × 25 FIFO +\njump signal = 76-D", BLUE)
    box(.64, 8.61, 1.70, .57, "Simulator features", "+30 privileged", GRAY, size=9.8)
    box(2.83, 10.10, 2.76, .68, "Current policy πᵢ", "76-D · frozen Actor + normalizer", BLUE, size=10.4)
    box(2.83, 8.60, 3.18, 1.05, "Residual explorer Eφ", "106 → 256 → 256 → 256 → 8\nSwish · Gaussian → tanh → 4-D", AMBER, size=10.7)
    box(6.31, 8.71, 1.49, .85, "Event gate", "3 steps\n+ amplitude bound", AMBER, size=10)
    arrow([(2.34, 10.02), (2.59, 10.02), (2.59, 10.44), (2.83, 10.44)], BLUE)
    arrow([(2.34, 9.72), (2.55, 9.72), (2.55, 9.28), (2.83, 9.28)], BLUE)
    arrow([(2.34, 8.89), (2.83, 8.89)], GRAY)
    ax.add_patch(Circle((7.40, 10.44), .19, fc="white", ec=INK, lw=1.2, zorder=3))
    text(7.40, 10.44, "+", size=19, zorder=4)
    arrow([(5.59, 10.44), (7.21, 10.44)], BLUE)
    text(6.38, 10.66, "Base action", size=9, color=BLUE)
    arrow([(6.01, 9.14), (6.31, 9.14)], AMBER)
    arrow([(7.40, 9.56), (7.40, 10.25)], AMBER)
    text(7.71, 9.92, "mₜ ε ⊙ δₜ", size=9.4, color=AMBER)
    box(7.92, 10.10, .86, .68, "Clip", "[−1, 1]", size=9.8)
    box(9.12, 9.98, 1.43, .94, "Actuator map", "MuJoCo dynamics\n50 Hz control", BLUE, size=10)
    arrow([(7.59, 10.44), (7.92, 10.44)])
    arrow([(8.78, 10.44), (9.12, 10.44)])
    arrow([(9.84, 10.92), (9.84, 11.00), (1.49, 11.00), (1.49, 10.40)], BLUE, lw=.9)
    text(1.51, 10.73, "Next observation", size=8.8, color=BLUE, ha="left")
    arrow([(9.84, 9.98), (9.84, 9.53)])
    port(9.84, 9.40, 1)
    text(9.84, 9.13, "Real prefix + s*", size=9.5)
    port(5.48, 10.78, 6, PURPLE)
    port(5.88, 9.65, 7, PURPLE)
    text(.68, 8.21, "aₜ = clip[πᵢ(oₜ) + mₜ ε ⊙ δₜ, −1, 1]", size=12, ha="left")
    text(.68, 7.92, "ε ∈ {0.10, 0.15}: normalized bound   |   steer · rear drive · hip · knee", size=9.6, ha="left")

    # Alternative trigger locations, plus three nonconstant residual samples.
    t = np.linspace(0, 1, 120)
    ex = .87 + 4.22*t
    ey = 7.15 + .39*np.sin(np.pi*t)**2
    ax.plot(ex, ey, color=BLUE, lw=1.4)
    for k, label, shift in [(0, "start", 0), (23, "liftoff", -.02), (60, "apex", 0), (98, "descent", .10)]:
        ax.plot(ex[k], ey[k], "o", ms=4, color=AMBER)
        text(ex[k]+shift, ey[k]-.17, label, size=8.7)
    text(3.05, 7.72, "One selected event per rollout", size=9.1, color=MUTED)
    ax.plot([5.87, 7.95], [7.19, 7.19], color=GRAY, lw=.8)
    for x, h in [(6.26, .19), (6.56, -.11), (6.86, .29)]:
        ax.plot([x, x], [7.19, 7.19+h], color=AMBER, lw=6, solid_capstyle="butt")
    text(6.91, 7.70, "3 ticks = 0.06 s", size=9.1)
    text(6.91, 6.99, "Re-sampled each tick", size=8.7, color=MUTED)

    # Deliberately simple two inline wheels and articulated pendulum, side view.
    for cx in (8.93, 10.16):
        ax.add_patch(Circle((cx, 7.19), .18, ec=INK, fc="none", lw=1.3))
    ax.plot([8.93, 9.48, 10.16], [7.19, 7.53, 7.19], color=BLUE, lw=2)
    ax.plot([8.93, 10.16], [7.19, 7.19], color=BLUE, lw=1.2)
    ax.plot([9.48, 9.64, 9.37], [7.53, 7.78, 8.02], color=AMBER, lw=2.3)
    ax.plot([9.48, 9.64, 9.37], [7.53, 7.78, 8.02], "o", ms=3.4, color=INK)
    text(9.56, 8.26, "Two inline wheels\n+ articulated pendulum", size=8.7)

    # (b) Current-policy verification and successor retest use the same snapshot.
    port(11.43, 10.49, 1)
    arrow([(11.55, 10.49), (11.74, 10.49)])
    box(11.74, 10.10, 2.06, .83, "Complete snapshot s*", "Physical state + FIFO\n+ event / actuator context", size=10)
    arrow([(13.80, 10.49), (14.10, 10.49)])
    box(14.10, 10.10, 3.20, .83, "Continue with current πᵢ", "Pulse OFF · stable recovery 0.5 s", BLUE, size=10.5)
    ax.plot([12.55, 16.49], [9.83, 9.83], color=INK, lw=1.1)
    ax.plot([15.70, 15.70], [10.10, 9.83], color=INK, lw=1.1)
    for x in (12.55, 14.51, 16.49):
        arrow([(x, 9.83), (x, 9.59)])
    box(11.73, 8.98, 1.64, .61, "Success W", "Verified witness", TEAL, size=10)
    box(13.64, 8.98, 1.74, .61, "Pending P", "Eligible failed suffix", AMBER, size=10)
    box(15.60, 8.98, 1.78, .61, "Unknown", "No negative label", GRAY, size=10)
    for x, n, c in [(12.55, 2, TEAL), (14.51, 3, AMBER)]:
        arrow([(x, 8.98), (x, 8.79)], c)
        port(x, 8.68, n, c)
    text(16.42, 8.66, "Excluded from PPO", size=8.8, color=MUTED)
    text(11.75, 8.37, "Recovery: grounded + forward + stable posture, continuously", size=9.0, ha="left")
    port(11.43, 7.75, 4, BLUE)
    arrow([(11.55, 7.75), (11.75, 7.75)], BLUE)
    box(11.75, 7.33, 2.58, .82, "Re-test SAME s*", "With candidate π′ · pulse OFF", BLUE, size=10.3)
    arrow([(14.33, 7.75), (14.65, 7.75)])
    box(14.65, 7.33, 2.69, .82, "Re-test outcomes", "Success / tested failure / unknown", PURPLE, size=10)
    arrow([(15.08, 7.33), (15.08, 7.17)], TEAL)
    port(15.08, 7.05, 2, TEAL)
    text(14.84, 7.05, "New W", size=8.5, color=TEAL, ha="right")
    arrow([(16.77, 7.33), (16.77, 7.17)], PURPLE, True)
    port(16.77, 7.05, 5, PURPLE)
    text(16.52, 7.05, "Quality + gain", size=8.5, color=PURPLE, ha="right")

    # (c) Stored states initialize NEW on-policy rollout samples.
    box(.73, 5.30, 2.57, .78, "Reset-state buffer", "Old W + pending P + full start", BLUE, size=10.4)
    port(.89, 6.08, 2, TEAL)
    port(1.21, 6.08, 3, AMBER)
    box(3.63, 5.30, 1.40, .78, "Fresh rollouts", "MuJoCo", BLUE, size=10)
    box(5.39, 5.18, 2.70, 1.02, "Controller PPO", "Warm Actor from πᵢ\nOriginal task reward + KL constraint\nFresh critic / optimizer", BLUE, size=10.3)
    box(8.43, 5.30, 1.97, .78, "Candidate π′", "New controller", BLUE, size=10.4)
    for x0, x1 in [(3.30, 3.63), (5.03, 5.39), (8.09, 8.43)]:
        arrow([(x0, 5.69), (x1, 5.69)], BLUE)
    arrow([(10.40, 5.69), (10.58, 5.69)], BLUE)
    port(10.68, 5.69, 4, BLUE)
    text(.74, 4.99, "20% full starts + 80% phase-balanced snapshots", size=9.7, ha="left")
    text(.74, 4.71, "Within each available phase: old / pending = 50 / 50", size=9, color=MUTED, ha="left")

    box(5.78, 3.46, 2.80, 1.08, "Retention gate", "New capability + old-support retention\n+ valid full nominal jump", PURPLE, size=10.2)
    port(6.02, 4.54, 2, TEAL)
    text(6.00, 4.69, "Old-support panel + start", size=8.3, color=TEAL)
    arrow([(9.42, 5.30), (9.42, 4.79), (7.18, 4.79), (7.18, 4.54)], BLUE)
    port(5.40, 3.99, 5, PURPLE)
    arrow([(5.52, 3.99), (5.78, 3.99)], PURPLE, True)
    text(5.43, 4.28, "Re-test", size=8.8, color=PURPLE)
    box(2.35, 3.59, 2.25, .79, "Accepted next policy", "πᵢ₊₁", BLUE, size=10.2)
    arrow([(5.78, 3.70), (5.03, 3.70), (5.03, 3.98), (4.60, 3.98)], PURPLE, True)
    text(5.14, 3.47, "accept", size=8.8, color=PURPLE)
    box(9.06, 3.59, 1.36, .79, "Keep πᵢ", "if rejected", GRAY, size=10)
    arrow([(8.58, 3.98), (9.06, 3.98)], PURPLE, True)
    arrow([(2.35, 3.98), (1.99, 3.98)], PURPLE, True)
    port(1.87, 3.98, 6, PURPLE)
    text(.76, 3.98, "Next iteration", size=9.6, ha="left", color=PURPLE)
    text(2.38, 3.28, "Accepted policy also defines the final-Actor output (6).", size=9, color=MUTED, ha="left")
    ax.plot([.62, 10.52], [3.04, 3.04], color="#DDD4E6", lw=1)
    text(.74, 2.82, "Separate explorer-learning path", size=10.6, bold=True, color=PURPLE, ha="left")
    box(.91, 1.64, 2.51, .92, "Exploration reward", "Novelty + delayed quality", AMBER, size=10.4)
    port(.64, 2.29, 5, PURPLE)
    arrow([(.76, 2.29), (.91, 2.29)], PURPLE, True)
    port(.64, 1.91, 8, AMBER)
    arrow([(.76, 1.91), (.91, 1.91)], AMBER)
    box(4.07, 1.64, 3.71, .92, "Explorer PPO", "Residual actor Eφ + own value critic Vψ\nBoth: 106-D privileged observation", PURPLE, size=10.6)
    arrow([(3.42, 2.10), (4.07, 2.10)], PURPLE, True)
    arrow([(7.78, 2.10), (8.26, 2.10)], PURPLE, True)
    port(8.39, 2.10, 7, PURPLE)
    text(8.68, 2.12, "Update explorer", size=10.1, color=PURPLE, ha="left")
    text(.74, 1.30, "rᴱ = 0.25 × new-cell novelty + outcome quality", size=10.7, ha="left")
    text(.74, 1.04, "+1 success; −1 tested failure after refinement; unknown excluded", size=9.1, color=MUTED, ha="left")

    # (d) Archive and final actor are separate outputs, with no arrow between them.
    port(11.46, 5.73, 2, TEAL)
    arrow([(11.58, 5.73), (11.80, 5.73)], TEAL)
    box(11.80, 5.37, 5.48, .76, "Successful-witness archive", "Real prefix + successful same-state continuation", TEAL, size=11)
    arrow([(14.54, 5.37), (14.54, 5.08)], TEAL)
    text(14.54, 4.96, "Conceptual x–z trajectory bundle · method schematic", size=9.6, color=MUTED)
    plot = fig.add_axes([11.92/18, 2.80/12.5, 5.00/18, 1.93/12.5])
    plot.set_facecolor(PANEL[3])
    plot.spines[["top", "right"]].set_visible(False)
    plot.spines[["left", "bottom"]].set_color(GRAY)
    plot.spines[["left", "bottom"]].set_linewidth(.8)
    plot.set(xticks=[], yticks=[], xlim=(-.04, 1.62), ylim=(0, 1.34))
    plot.set_xlabel("Forward position x", fontsize=9, labelpad=2)
    plot.set_ylabel("Base height z", fontsize=9, labelpad=2)
    plot.axhline(.12, color="#B4B7B4", lw=.7, zorder=0)
    records = []
    # Deterministic single-peak lift: z = z0 + H sin²(π min(x/L, 1)).
    # After x=L, each trajectory has a short constant-height recovery tail.
    formula = "z=0.18+H*sin(pi*min(x/L,1))**2; x in [0,L+0.16]"
    for index in range(16):
        initial = index < 5
        height = .64 + .033*index if not initial else .71 + .021*index
        length = .97 + .024*index if not initial else 1.055 + .018*index
        xx = np.linspace(0, length+.16, 170)
        zz = .18 + height*np.sin(np.pi*np.minimum(xx/length, 1))**2
        group = "initial_tube" if initial else "added_successful_witness"
        plot.plot(xx, zz, color=BLUE if initial else TEAL, lw=1.0,
                  alpha=.58 if initial else .44, zorder=2)
        plot.scatter(xx[20:140:27], zz[20:140:27], s=4,
                     color=BLUE if initial else TEAL, alpha=.65, zorder=3)
        records.extend(dict(kind="trajectory", trajectory_id=index, group=group,
                            sample=i, x=float(x), z=float(z), height=height,
                            landing_length=length, seed="none_deterministic",
                            formula=formula, evidence_role="schematic_not_experimental")
                       for i, (x, z) in enumerate(zip(xx, zz)))
    # Continuous orange prefix joins one chosen teal continuation at s*.
    length, height = 1.23, 1.00
    xx = np.linspace(0, length+.16, 170)
    zz = .18+height*np.sin(np.pi*np.minimum(xx/length, 1))**2
    k = 45
    plot.plot(xx[:k+1], zz[:k+1], color=AMBER, lw=2.0, zorder=5)
    plot.plot(xx[k:], zz[k:], color=TEAL, lw=1.65, zorder=4)
    plot.plot(xx[k], zz[k], "o", ms=4.3, color=AMBER, zorder=6)
    plot.annotate("s*", (xx[k], zz[k]), xytext=(-11, 8), textcoords="offset points", fontsize=10)
    records.extend(dict(kind="trajectory", trajectory_id=16, group="prefix_continuation_example",
                        sample=i, x=float(x), z=float(z), height=height,
                        landing_length=length, seed="none_deterministic", formula=formula,
                        evidence_role="schematic_not_experimental") for i, (x,z) in enumerate(zip(xx,zz)))
    failures = [(.43, 1.15), (1.11, .97), (1.40, .67)]
    for index, (x, z) in enumerate(failures):
        plot.plot(x, z, "x", color=CORAL, ms=6, mew=1.3)
        records.append(dict(kind="marker", trajectory_id=f"failure_{index}", group="tested_no_witness",
                            sample=0, x=x, z=z, height="", landing_length="", seed="none_deterministic",
                            formula="declared conceptual marker", evidence_role="schematic_not_experimental"))
    plot.plot(0, .18, "o", color=INK, ms=3.5)
    plot.text(.025, .06, "start", fontsize=8)
    plot.text(.80, 1.26, "Unexplored", color=MUTED, fontsize=8.5)
    plot.annotate("Stable 0.5 s", (1.40, .18), xytext=(1.10, .36), fontsize=8.3,
                  arrowprops=dict(arrowstyle="->", color=MUTED, lw=.7), color=MUTED)
    plot.plot([], [], color=BLUE, lw=1.6, label="Initial tube")
    plot.plot([], [], color=TEAL, lw=1.6, label="Added witnesses")
    plot.plot([], [], color=CORAL, marker="x", ls="none", label="Tested, no witness")
    plot.legend(loc="upper left", bbox_to_anchor=(0, -.15), ncol=3, fontsize=7.8,
                frameon=False, handlelength=1.1, columnspacing=.8, handletextpad=.35)
    # Visited ledger is a data ledger, distinct from the successful archive.
    port(11.48, 2.11, 8, AMBER)
    text(11.73, 2.11, "Visited ledger: keep tested failures for novelty accounting", size=9.0, ha="left")
    port(11.46, 1.48, 6, PURPLE)
    arrow([(11.58, 1.48), (11.80, 1.48)], PURPLE, True)
    box(11.80, 1.09, 5.48, .77, "Frozen final controller π*", "Independent perturbed-jump evaluation", BLUE, size=11)

    # Every inter-panel link has an explicit named port in this legend.
    arrow([(.40, .61), (1.07, .61)])
    text(1.17, .61, "data / control", size=9.1, ha="left")
    arrow([(2.72, .61), (3.39, .61)], PURPLE, True)
    text(3.49, .61, "learning feedback / parameters", size=9.1, ha="left")
    text(7.04, .61, "Matching numbered ports connect panels; arrowheads indicate the local direction.", size=9.2, ha="left", color=MUTED)
    text(.40, .27, "1 snapshot   2 successful W   3 pending P   4 candidate π′   5 re-test quality / gain   6 accepted controller   7 explorer parameters   8 visited ledger",
         size=9.3, ha="left")

    # Check all block-contained text before exporting; diagram-only validation.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflow = []
    for item, patch in text_boxes:
        tb, pb = item.get_window_extent(renderer), patch.get_window_extent(renderer)
        if tb.x0 < pb.x0+1 or tb.x1 > pb.x1-1 or tb.y0 < pb.y0+1 or tb.y1 > pb.y1-1:
            overflow.append(item.get_text().replace("\n", " / "))
    if overflow:
        raise RuntimeError("Text exceeds block bounds: " + "; ".join(overflow))
    csv_path = args.output / "overview_schematic_curves.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {"Title": "JIT method overview", "Creator": "build_method_overview.py",
                "Description": "Conceptual method schematic; synthetic curves are not experimental data."}
    for extension in ("svg", "pdf", "png"):
        path = args.output / f"overview_vector.{extension}"
        selected = metadata if extension != "pdf" else {
            "Title": metadata["Title"], "Creator": metadata["Creator"], "Subject": metadata["Description"]}
        fig.savefig(path, dpi=args.dpi, metadata=selected)
        print(path)
    plt.close(fig)
    print(csv_path)
    print(f"Canvas: 18 × 12.5 inches; PNG: {18*args.dpi} × {int(12.5*args.dpi)} pixels")
    print(f"Checked {len(text_boxes)} text blocks; synthetic trajectory / marker rows: {len(records)}")


if __name__ == "__main__":
    main()
