"""Write the shaded tables of the second paper from results/ (no model, no network).

Run: python3 analysis/tables_gen.py   (writes the four table fragments to results/tables/)
Shading rule, as in the first paper: above = the whole 95% interval is above the reference;
under = the whole interval is under it; no mark = undecided.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "tables"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def cell(text: str, lo: float, hi: float, ref: float) -> str:
    if lo > ref:
        return rf"\ab{{{text}}}"
    if hi < ref:
        return rf"\ub{{{text}}}"
    return text


def qa_seeds() -> str:
    h64 = load("clsprobe_h64.json")["arms"]
    rows = []
    for arm, name in (("a", "DBpedia only"), ("known", "Mixed")):
        cells = []
        for d in h64[arm]:
            q = d["qa_f1"]
            m, lo, hi = 100 * q["mean"], 100 * q["lo"], 100 * q["hi"]
            cells.append(cell(f"{m:.1f}", lo, hi, 90.0))
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    head = " & ".join(f"Seed {i}" for i in range(6))
    return (
        r"\begin{tabular}{l" + "r" * 6 + "}\n\\toprule\n"
        f"Arm & {head} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def gain() -> str:
    paired = load("clsprobe_h68_paired.json")
    rows = []
    for arm, name in (
        ("a", "DBpedia only, fixed order"),
        ("dbrand", "DBpedia only, random order"),
        ("known", "Mixed, random order"),
    ):
        cells = []
        for d in paired[arm]:
            c = d["cls_acc"]
            cells.append(cell(f"{c['mean']:+.1f}".replace("-", "$-$"), c["lo"], c["hi"], 0.0))
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    head = " & ".join(f"Seed {i}" for i in range(6))
    return (
        r"\begin{tabular}{l" + "r" * 6 + "}\n\\toprule\n"
        f"Arm & {head} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


TASKS = (
    ("cls_acc", "AG News, listed order"),
    ("ag_rot", "AG News, rotated order"),
    ("emotion", "Emotion, listed order"),
    ("emotion_rot", "Emotion, rotated order"),
)
ARMS = (("a", "DBpedia only, fixed order"), ("dbrand", "DBpedia only, random order"), ("known", "Mixed, random order"))


def medians() -> str:
    h64, h68 = load("clsprobe_h64.json")["metrics"], load("clsprobe_h68.json")["metrics"]
    rows = []
    for key, name in TASKS:
        m = h68[key]
        p1 = (
            f"{h64[key]['p_known_above_a']:.3f}"
            if key in ("cls_acc", "ag_rot")
            else f"{h64[key]['p_known_above_a']:.2f}"
        )
        if key in ("cls_acc", "ag_rot"):
            p2, p3 = f"{m['p_dbrand_above_a']:.3f}", f"{m['p_known_above_dbrand']:.3f}"
        else:
            p2 = p3 = "--"
        meds = f"{m['a']['median']:.1f} & {m['dbrand']['median']:.1f} & {m['known']['median']:.1f}"
        rows.append(f"{name} & {meds} & {p1} & {p2} & {p3} \\\\")
    return (
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        " & \\multicolumn{3}{c}{Median kept (\\%)} & \\multicolumn{3}{c}{Rank test $p$, first over second} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Task & Fixed & Random & Mixed & Mixed, fixed & Random, fixed & Mixed, random \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def all_seeds() -> str:
    h68 = load("clsprobe_h68.json")["metrics"]
    rows = []
    for key, name in TASKS:
        for arm, arm_name in ARMS:
            vals = ", ".join(f"{v:.1f}" for v in h68[key][arm]["seeds"])
            rows.append(f"{name if arm == 'a' else ''} & {arm_name} & {vals} \\\\")
        rows.append("\\addlinespace[2pt]")
    return (
        "\\begin{tabular}{llr}\n\\toprule\nTask & Arm & Kept (\\%), seeds 0 to 5 \\\\\n\\midrule\n"
        + "\n".join(rows[:-1])
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "tab_qa_seeds.tex").write_text(qa_seeds())
    (OUT / "tab_gain.tex").write_text(gain())
    (OUT / "tab_medians.tex").write_text(medians())
    (OUT / "tab_allseeds.tex").write_text(all_seeds())
