"""Check that every number the report states is in the report, and that the result files give it.

    python analysis/check_numbers.py [REPORT.pdf]      # default: moe-router-delta.pdf at the repository root

Exit 1 and a list of missing or wrong numbers, otherwise "all numbers found". The text of the report comes from the PDF
(`pdftotext`, from poppler). A needle written with LaTeX markup is turned into plain text and searched with the white
space removed, so that line breaks do not matter. The shading of a table cell (`\\ub`, `\\ab`) is not in the PDF text;
those needles are searched in the table fragments that `tables_gen.py` writes to `results/tables/`.
Run `python analysis/tables_gen.py` first.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import selrule as sl  # noqa: E402
import short  # noqa: E402

RESULTS = ROOT / "results"
PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "moe-router-delta.pdf"


def squash(text: str) -> str:
    """Plain text with no white space: minus signs and ligatures unified."""
    text = text.replace("\u2212", "-").replace("\u00ad", "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    return re.sub(r"\s+", "", text)


def plain(latex: str) -> str:
    """A needle written in LaTeX, as the PDF shows it."""
    text = re.sub(r"\\(?:ub|ab)\{([^}]*)\}", r"\1", latex)
    text = text.replace("\\%", "%").replace("$", "").replace("~", " ").replace(" & ", " ")
    return text


PAPER = squash(
    subprocess.run(["pdftotext", "-layout", str(PDF), "-"], capture_output=True, text=True, check=True).stdout
)
TABLES = "".join(f.read_text() for f in sorted((RESULTS / "tables").glob("tab_*.tex")))
M, N = "qwen36", 300
problems: list[str] = []


def need(label: str, text: str) -> None:
    marked = "\\ub{" in text or "\\ab{" in text
    found = text in TABLES if marked else squash(plain(text)) in PAPER
    if not found:
        problems.append(f"{label}: '{text}' not in the report")


def f1(x: float) -> str:
    return f"{100 * x:.1f}"


def seeds(vals: list[float]) -> str:
    return ", ".join(f"{v:.1f}" for v in vals)


h64 = json.loads((RESULTS / "clsprobe_h64.json").read_text())
h65 = json.loads((RESULTS / "clsprobe_h65.json").read_text())
jury = json.loads((RESULTS / "jury_h67.json").read_text())
metrics = h64["metrics"]
for key, arm in (
    ("qa_f1", "a"),
    ("qa_f1", "known"),
    ("cls_acc", "a"),
    ("cls_acc", "known"),
    ("ag_rot", "a"),
    ("ag_rot", "known"),
    ("emotion", "a"),
    ("emotion", "known"),
    ("emotion_rot", "a"),
    ("emotion_rot", "known"),
):
    if key == "qa_f1":
        for v in metrics[key][arm]["seeds"]:
            need(f"{key} {arm} seed", f"{v:.1f}")
    else:
        need(f"{key} {arm} seeds", seeds(metrics[key][arm]["seeds"]))
    need(f"{key} {arm} median", f"{metrics[key][arm]['median']:.1f}")
for key in ("cls_acc", "ag_rot"):
    need(f"{key} one-sided p", f"{metrics[key]['p_known_above_a']:.3f}")
need("QA two-sided p", f"{metrics['qa_f1']['p_two_sided']:.2f}")

# counts in the prose
qa = [r["qa_f1"] for arm in h64["arms"].values() for r in arm]
at_bar_mean = sum(d["mean"] >= 0.9 for d in qa)
holds = sum(d["mean"] >= 0.9 or d["lo"] <= 0.9 <= d["hi"] for d in qa)
under = [d for d in qa if d["hi"] < 0.9]
print(f"QA: mean at or above 90 in {at_bar_mean} of {len(qa)}; interval holds 90 in {holds}; fully under: {len(under)}")
need("QA seeds reaching 90", f"reaches 90\\% in {at_bar_mean} of {len(qa)} seeds")
need("QA seeds at the bar", f"interval holds 90\\% in {holds} of {len(qa)}")
for d in under:
    need("QA seed under 90 interval", f"{100 * d['mean']:.1f} [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]")
for k in ("cls_acc", "ag_rot"):
    need(f"{k} two-sided p", f"{metrics[k]['p_two_sided']:.3f}")
need("seed range", f"{max(metrics['cls_acc']['a']['seeds']) - min(metrics['cls_acc']['a']['seeds']):.1f} points")
need(
    "QA seeds fully under 90",
    "In two mixed-arm seeds the whole interval lies under 90" if len(under) == 2 else "COUNT-CHANGED",
)
for d in under:
    need("QA seed under 90 marked", rf"\ub{{{100 * d['mean']:.1f}}}")
below = sum(v < 86.8 for v in metrics["cls_acc"]["a"]["seeds"])
need(
    "DBpedia-only seeds below untrained",
    f"in {'five' if below == 5 else below} of the six seeds of the DBpedia-only arm the trained list scores lower",
)

# final test and untrained rows
for arm in ("a", "known"):
    r = h65["arms"][arm]
    for k in ("qa_f1", "cls_acc"):
        d = r[k]
        need(f"final {arm} {k}", f"{100 * d['mean']:.1f} [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]")
    need(f"final {arm} emotion", f"{100 * r['emotion']['mean']:.1f}")
    need(f"final {arm} emotion rot", f"{100 * r['emotion_rot']['mean']:.1f}")
base = short.read_records(short.run_path(sl.tag(M, sl.WHOLE_ARM, 600), N, 0, 32))
teacher = short.teacher_outputs(base)
sc = short.item_metrics(base, teacher)
untrained = short.item_metrics(short.read_records(short.run_path(sl.tag(M, sl.ENDS_ARM, 600), N, 12, 32)), teacher)
for k in ("qa_f1", "cls_acc"):
    d = sl.kept_ci(untrained[k], sc[k])
    need(f"untrained 600-899 {k}", f"{100 * d['mean']:.1f} [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]")

# jury
for cell, _name in (("untrained", "Untrained list"), ("dbpedia", "DBpedia-only delta"), ("mixed", "Mixed delta")):
    c = jury["cells"][cell]
    k = c["kept"]["accepted"]
    need(f"jury {cell} accepted kept", f"{100 * k['mean']:.1f} [{100 * k['lo']:.1f}, {100 * k['hi']:.1f}]")
    need(f"jury {cell} accepted", f"{100 * c['accepted_masked']:.1f}")
need("jury unmasked accepted", f"{100 * jury['cells']['mixed']['accepted_unmasked']:.1f}")


# paired differences against the untrained list, items 300 to 599
def paired(a: dict, b: dict) -> tuple[float, float, float]:
    keys = sorted(a.keys() & b.keys())
    x = np.array([a[k] - b[k] for k in keys], float)
    idx = np.random.default_rng(0).integers(0, len(keys), size=(2000, len(keys)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return 100 * x.mean(), 100 * lo, 100 * hi


b3 = short.read_records(short.run_path(sl.tag(M, sl.WHOLE_ARM, 300), N, 0, 32))
t3 = short.teacher_outputs(b3)
un3 = short.item_metrics(short.read_records(short.run_path(sl.tag(M, sl.ENDS_ARM, 300), N, 12, 32)), t3)["cls_acc"]
known = ["stop12known"] + [f"stop12knownseed{i}" for i in range(1, 6)]
diffs = []
for lab in known:
    tr = short.item_metrics(
        short.read_records(short.run_path(sl.tag(M, sl.ENDS_ARM, 300, True, "_" + lab), N, 12, 32)), t3
    )["cls_acc"]
    diffs.append(paired(tr, un3))
means = [d[0] for d in diffs]
print("mixed arm paired differences:", [round(m, 1) for m in means], "excluding zero:", sum(d[1] > 0 for d in diffs))
need("mixed range low", f"by {min(means):.1f} to {max(means):.1f} points")
need(
    "mixed seeds excluding zero",
    "in three of the six seeds the interval excludes zero"
    if sum(d[1] > 0 for d in diffs) == 3
    else "EXCLUDES-ZERO-COUNT-CHANGED",
)

# label-order control: three arms, rank tests and paired medians
h68 = json.loads((RESULTS / "clsprobe_h68.json").read_text())
h68p = json.loads((RESULTS / "clsprobe_h68_paired.json").read_text())
cm = h68["metrics"]
for k in ("cls_acc", "ag_rot", "emotion", "emotion_rot"):
    need(
        f"medians row {k}",
        f"{cm[k]['a']['median']:.1f} & {cm[k]['dbrand']['median']:.1f} & {cm[k]['known']['median']:.1f}",
    )
need("control p listed", f"{cm['cls_acc']['p_dbrand_above_a']:.3f}")
need("control p rotated", f"{cm['ag_rot']['p_dbrand_above_a']:.3f}")
need("mixed above control p listed", f"{cm['cls_acc']['p_known_above_dbrand']:.3f}")
need("mixed above control p rotated", f"{cm['ag_rot']['p_known_above_dbrand']:.3f}")
for arm in ("a", "dbrand", "known"):
    med = float(np.median([r["cls_acc"]["mean"] for r in h68p[arm]]))
    sign = "+" if med >= 0 else "-"
    need(f"{arm} paired median", f"${sign}{abs(med):.1f}$")
db = h68p["dbrand"]
need(
    "control below zero",
    "in five of six seeds it is lower" if sum(r["cls_acc"]["mean"] < 0 for r in db) == 5 else "COUNT-CHANGED",
)
if sum(r["cls_acc"]["hi"] < 0 for r in db) != 3:
    problems.append("control intervals below zero changed")
need("control median text", f"median of ${float(np.median([r['cls_acc']['mean'] for r in db])):.1f}$ points")
if h68["reading"] != "data explains" or not h68["valid"]:
    problems.append("control reading changed")

# overlap audit
ov = json.loads((RESULTS / "overlap_audit.json").read_text())
sc = ov["scored_items_by_teacher_file"]


def shared(rng: str, f: str, task: str) -> int:
    return sc[rng][f][task]["sharing_a_window"]


kn, un = "qwen36_c3b_teacher_known", "qwen36_c3b_teacher"
got = (
    shared("items 300 to 599", kn, "classify"),
    shared("items 600 to 899", kn, "classify"),
    *(shared(r, kn, t) for r in sc for t in ("qa", "rewrite")),
)
if got != (5, 4, 0, 0, 0, 0):
    problems.append(f"overlap counts changed: {got}")
if (shared("items 300 to 599", un, "summarize"), shared("items 600 to 899", un, "summarize")) != (0, 1):
    problems.append("summarize overlap count changed")
need("overlap text", "five in items 300 to 599 and four in items 600 to 899")
need("dup sensitivity", "92.1\\%, against 81.3\\%")
if (
    "median 92.1" not in (RESULTS / "dup_sens.txt").read_text()
    or "median 81.3" not in (RESULTS / "dup_sens.txt").read_text()
):
    problems.append("dup_sens.txt changed")

# Holm over the six two-sided tests, and the juror agreement quoted in the text
ps = sorted(
    (metrics[m]["p_two_sided"], m) for m in ("cls_acc", "ag_rot", "emotion", "emotion_rot", "sum_rougeL_ref", "qa_f1")
)
adj, prev = {}, 0.0
for i, (pv, m) in enumerate(ps):
    prev = max(prev, min(1.0, (len(ps) - i) * pv))
    adj[m] = prev
need("Holm values", f"{adj['cls_acc']:.3f} for the listed order and {adj['ag_rot']:.3f}")
ag = sorted(jury["agreement"].values())
need("juror agreement", f"{round(100 * ag[0])} to {round(100 * ag[-1])}")

import glob as _g
import re as _re
import statistics as _st

_loads = [
    float(x)
    for f in _g.glob(str(RESULTS / "qwen36*stop12*scores*.json"))
    for x in _re.findall(r'"loads_per_layer": ([0-9.]+)', Path(f).read_text())
]
if not _loads or round(_st.median(_loads)) != 18:
    problems.append(f"loads per layer: median {_st.median(_loads) if _loads else None}, text says about 18")
if any(jury["jurors"][j]["n"] != 1200 for j in jury["jurors"]):
    problems.append("juror agreement: n is not 1,200")

print("\n".join(problems) if problems else "all numbers found")
sys.exit(1 if problems else 0)
