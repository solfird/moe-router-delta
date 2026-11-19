"""Figures of the second paper, computed from results/ only (no model, no network).

Run from the repository root:
    uv run --no-project --python 3.12 --with matplotlib==3.9.2 --with numpy python analysis/make_figures.py

Writes each figure to figures/ as PDF (the one in the report) and as PNG.

The untrained rows (QA 72.6, AG News 86.8 on items 300 to 599) are the runs that
check_numbers.py recomputes from the run files and finds in the report.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
    }
)
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID, MUTED, GREY = "#d9d8d4", "#52514e", "#8d8c87"
ARMS = (
    ("a", "DBpedia only", BLUE, "o"),
    ("dbrand", "DBpedia, random order", YELLOW, "D"),
    ("known", "Mixed", ORANGE, "s"),
)
UNTRAINED = {"qa_f1": 72.6, "cls_acc": 86.8}


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def bar_line(ax: plt.Axes) -> None:
    ax.axhline(90, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def strip(ax: plt.Axes, x: float, values: list[float], color: str, marker: str) -> None:
    n = len(values)
    xs = [x + 0.22 * (i - (n - 1) / 2) / max(n - 1, 1) * 2 for i in range(n)]
    ax.scatter(xs, values, s=22, color=color, marker=marker, zorder=3, edgecolor="white", linewidth=0.4)
    med = sorted(values)[n // 2 - 1 : n // 2 + 1]
    ax.hlines(sum(med) / len(med), x - 0.3, x + 0.3, color=color, linewidth=1.6, zorder=2)


def fig_teaser() -> None:
    """What training adds: untrained list against the six seeds of each arm, and the judged summaries."""
    h64, h68, jury = load("clsprobe_h64.json"), load("clsprobe_h68.json"), load("jury_h67.json")
    fig, axes = plt.subplots(
        1, 3, figsize=(6.5, 2.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.25, 1]}
    )
    # Panel 1 and 3: seeds against the untrained list
    for ax, key, title in ((axes[0], "qa_f1", "Question answering"), (axes[2], "cls_acc", "Classification, AG News")):
        bar_line(ax)
        ax.axhline(UNTRAINED[key], color=GREY, linewidth=1.4)
        ax.text(2.55, UNTRAINED[key] - 1.2, "untrained", color=GREY, fontsize=7.5, ha="right", va="top")
        for i, (arm, _label, color, marker) in enumerate(ARMS):
            src = h68 if arm == "dbrand" or key == "cls_acc" else h64
            vals = src["metrics"][key][arm]["seeds"]
            strip(ax, i, vals, color, marker)
        ax.set_xticks(range(3), ["DBpedia\nonly", "DBpedia,\nrandom", "Mixed"], fontsize=7.5)
        ax.set_title(title, fontsize=9, color=MUTED)
        ax.set_xlim(-0.6, 2.6)
    axes[0].set_ylim(60, 100)
    axes[0].set_ylabel("Kept, % of the full model")
    axes[2].set_ylim(60, 100)
    # Panel 2: the jury
    ax = axes[1]
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    cells = (
        ("untrained", "Untrained\nlist", GREY),
        ("dbpedia", "DBpedia-only\ndelta", BLUE),
        ("mixed", "Mixed\ndelta", ORANGE),
    )
    for i, (cell, _label, color) in enumerate(cells):
        k = jury["cells"][cell]["kept"]["accepted"]
        m, lo, hi = 100 * k["mean"], 100 * k["lo"], 100 * k["hi"]
        ax.bar(i, m, width=0.55, color=color)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=MUTED, capsize=2.5, linewidth=1)
        ax.text(i, hi + 2, f"{m:.1f}", ha="center", fontsize=8, color=MUTED)
    ax.axhline(100, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(2.45, 101.5, "full model", fontsize=7.5, color=MUTED, ha="right")
    ax.set_xticks(range(3), [c[1] for c in cells], fontsize=7.5)
    ax.set_ylim(0, 112)
    ax.set_title("Summaries accepted by a jury", fontsize=9, color=MUTED)
    save(fig, "fig_teaser")


def fig_seeds() -> None:
    """Classification across six seeds, three arms, two label orders, two tasks."""
    h68 = load("clsprobe_h68.json")["metrics"]
    panels = (
        ("cls_acc", "AG News, listed order"),
        ("ag_rot", "AG News, rotated order"),
        ("emotion", "Emotion, listed order"),
        ("emotion_rot", "Emotion, rotated order"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.5), sharey=True, constrained_layout=True)
    for ax, (key, title) in zip(axes, panels, strict=True):
        bar_line(ax)
        for i, (arm, _label, color, marker) in enumerate(ARMS):
            strip(ax, i, h68[key][arm]["seeds"], color, marker)
        if key == "cls_acc":
            ax.axhline(UNTRAINED["cls_acc"], color=GREY, linewidth=1.4)
            ax.text(2.55, UNTRAINED["cls_acc"] - 1.2, "untrained", color=GREY, fontsize=7.5, ha="right", va="top")
        ax.set_xticks(range(3), ["DB\nfixed", "DB\nrandom", "Mixed"], fontsize=7.5)
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(title, fontsize=8, color=MUTED)
    axes[0].set_ylabel("Kept, % of the full model")
    axes[0].set_ylim(45, 105)
    handles = [
        plt.Line2D([], [], color=c, marker=m, linestyle="none", markersize=5, label=lab) for _a, lab, c, m in ARMS
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
    save(fig, "fig_seeds")


def fig_jury() -> None:
    """Per question: what the jury accepts, as a share of the full model's rate."""
    jury = load("jury_h67.json")["cells"]
    qs = (
        ("supported", "Backed by\nthe article"),
        ("condensed", "Short, two or\nthree sentences"),
        ("covers", "States the\nmain points"),
    )
    cells = (
        ("untrained", "Untrained list", GREY),
        ("dbpedia", "DBpedia-only delta", BLUE),
        ("mixed", "Mixed delta", ORANGE),
    )
    fig, ax = plt.subplots(figsize=(6.5, 2.4), constrained_layout=True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    w = 0.26
    for j, (cell, label, color) in enumerate(cells):
        for i, (q, _ql) in enumerate(qs):
            k = jury[cell]["kept"][q]
            m, lo, hi = 100 * k["mean"], 100 * k["lo"], 100 * k["hi"]
            x = i + (j - 1) * w
            ax.bar(x, m, width=w * 0.92, color=color, label=label if i == 0 else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=MUTED, capsize=2, linewidth=0.9)
            ax.text(x, hi + 2, f"{m:.0f}", ha="center", fontsize=7.5, color=MUTED)
    ax.axhline(100, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_xticks(range(3), [q[1] for q in qs])
    ax.set_ylim(0, 118)
    ax.set_ylabel("Jury's rate, % of the\nfull model's rate")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig_jury")


def fig_method() -> None:
    """The router delta on one layer: the same router, plus a trained term that exists only under the short list."""
    fig, ax = plt.subplots(figsize=(6.5, 2.1), constrained_layout=True)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, face: str, edge: str, bold: bool = False) -> None:
        ax.add_patch(
            mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=face, ec=edge, lw=1.1)
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=8,
            color="#1a1a18",
            weight="bold" if bold else None,
        )

    def arrow(x0: float, y0: float, x1: float, y1: float, color: str = MUTED) -> None:
        ax.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.1})

    box(0.1, 1.55, 1.7, 1.3, "token\nstate", "#eeeeea", GREY)
    box(3.0, 2.75, 2.6, 1.2, "router\n(frozen)", "#e3eefb", BLUE)
    box(3.0, 0.45, 2.6, 1.2, "router delta\n(trained)", "#fde9df", ORANGE, bold=True)
    box(6.7, 1.55, 1.9, 1.3, "add, then\nmask to 12\nof 256", "#eeeeea", GREY)
    box(9.7, 1.55, 1.7, 1.3, "top 8\nexperts", "#e0f4ec", AQUA)
    arrow(1.9, 2.4, 3.0, 3.35)
    arrow(1.9, 2.0, 3.0, 1.05, ORANGE)
    arrow(5.6, 3.35, 6.7, 2.5)
    arrow(5.6, 1.05, 6.7, 1.9, ORANGE)
    arrow(8.7, 2.2, 9.7, 2.2)
    ax.text(4.3, 0.1, "added only when the short list is on", ha="center", fontsize=7.5, color=ORANGE, style="italic")
    save(fig, "fig_method")


if __name__ == "__main__":
    fig_teaser()
    fig_seeds()
    fig_jury()
    fig_method()
