"""H67: does a jury accept the summaries of the trained deltas? Qwen3.6 K' = 12, items 600 to 899, four cells.

jury_d.py build     # results/jury_d/items.jsonl, shuffled, 1,200 answers
jury_d.py report    # results/jury_h67.json (after the jurors ran: jury.py --round d run ..., blindbatch.py)
"""

import json
import pathlib
import sys
from typing import Any

import numpy as np

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / "analysis")]

import copyaudit as ca  # noqa: E402
import corpus  # noqa: E402
import failures
import jury
import selrule as sl
import short

MODEL, K, START = "qwen36", 12, 600
TRAINED = ("dbpedia", "mixed")


def cells() -> list[tuple[str, str, int]]:
    """(cell name, output tag, K') of the four H67 cells."""
    return [
        ("unmasked", sl.tag(MODEL, sl.WHOLE_ARM, START), 0),
        ("untrained", sl.tag(MODEL, sl.ENDS_ARM, START), K),
        ("dbpedia", sl.tag(MODEL, sl.ENDS_ARM, START, True, "_stop12seed6"), K),
        ("mixed", sl.tag(MODEL, sl.ENDS_ARM, START, True, "_stop12knownseed6"), K),
    ]


def bar_state(d: dict[str, float]) -> str:
    """above: the whole interval is at 90% or more; under: the whole interval is under 90%; else undecided."""
    if d["lo"] >= 0.90:
        return "above"
    if d["hi"] < 0.90:
        return "under"
    return "undecided"


def h67_verdict(kept: dict[str, dict[str, float]]) -> str:
    """Holds if both trained deltas are at the bar (mean 90% or more, or an interval that holds 90%)."""
    return "holds" if all(sl.at_bar([kept[c]]) for c in TRAINED) else "fails"


def build() -> None:
    jury.use_round("d")
    articles = {i: p.split("\n\n", 1)[1] for t, i, p, _ in short.items(300, START) if t == "summarize"}
    rows = []
    for name, tag, k in cells():
        for r in short.read_records(short.run_path(tag, 300, k, 32)):
            if r["task"] == "summarize":
                rows.append({"cell": name, "i": r["i"], "article": articles[r["i"]], "answer": r["out"]})
    jury.JURY.mkdir(parents=True, exist_ok=True)
    with jury.ITEMS.open("w") as f:
        for r in jury.shuffled(rows):
            f.write(json.dumps(r) + "\n")
    print(len(rows))


def report() -> dict:
    jury.use_round("d")
    jury.CLAUDE = "claude_blind"
    items = jury._read(jury.ITEMS)
    votes = jury._votes()
    by: dict[str, dict[int, dict[str, Any]]] = {}
    for it in items:
        cast = [votes[j][it["id"]] for j in votes if it["id"] in votes[j]]
        if len(cast) == 3:
            by.setdefault(it["cell"], {})[it["i"]] = {
                **jury.majority(cast),
                "copied": it["copied"] if "copied" in it else ca.copied_share(it["answer"], corpus.article_of(it)),
            }
    loops = {}
    for name, tag, k in cells():
        recs = short.read_records(short.run_path(tag, 300, k, 32))
        loops[name] = {r["i"] for r in recs if r["task"] == "summarize" and failures.is_loop(r["gen_ids"])}
    base = by["unmasked"]
    cellsout: dict[str, Any] = {}
    for name in ("untrained", *TRAINED):
        arm = by[name]
        kept = {
            q: sl.kept_ci({i: float(r[q]) for i, r in arm.items()}, {i: float(r[q]) for i, r in base.items()})
            for q in (*jury.QUESTIONS, "accepted")
        }
        lp = [arm[i]["accepted"] for i in loops[name] if i in arm]
        cellsout[name] = {
            "n": len(arm),
            "accepted_masked": float(np.mean([r["accepted"] for r in arm.values()])),
            "accepted_unmasked": float(np.mean([r["accepted"] for r in base.values()])),
            "kept": kept,
            "state": bar_state(kept["accepted"]),
            "accepted_over_half_copied": [
                float(np.mean([r["accepted"] for r in arm.values() if r["copied"] > 0.5] or [0])),
                sum(r["copied"] > 0.5 for r in arm.values()),
            ],
            "accepted_loops": [float(np.mean(lp)) if lp else None, len(lp)],
        }
    ids = {it["id"]: it for it in items}
    jurors = {}
    for j, v in votes.items():
        unm = [all(x.values()) for i, x in v.items() if ids[i]["cell"] == "unmasked"]
        jurors[j] = {"n": len(v), "accepted_unmasked": float(np.mean(unm)) if unm else None}
    names = list(votes)
    agree = {}
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            both = votes[names[x]].keys() & votes[names[y]].keys()
            if both:
                agree[f"{names[x]}_{names[y]}"] = float(
                    np.mean([votes[names[x]][i] == votes[names[y]][i] for i in both])
                )
    out = {
        "verdict": h67_verdict({c: cellsout[c]["kept"]["accepted"] for c in cellsout}),
        "valid": all((j["accepted_unmasked"] or 0) >= 0.5 for j in jurors.values()),
        "cells": cellsout,
        "jurors": jurors,
        "agreement": agree,
        "spent_usd": jury.spent(),
    }
    (jury.RESULTS / "jury_h67.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    {"build": build, "report": lambda: print(json.dumps(report(), indent=1))}[sys.argv[1]]()
