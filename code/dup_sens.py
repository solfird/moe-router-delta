import json
import statistics as st
import sys

sys.path.insert(0, ".")
import numpy as np
from transformers import AutoTokenizer

import c3b
import clsprobe as cp
import overlap_audit as oa
import selrule as sl
import short


def keep(d: dict, dup: set) -> dict:
    return {key: v for key, v in d.items() if key[1] not in dup}


tok = AutoTokenizer.from_pretrained(c3b.MODELS["qwen36"])
with open("distill_data/qwen36_c3b_teacher_known.jsonl") as f:
    known = [oa.body(tok.decode(json.loads(line)["prompt"])) for line in f]
idx = oa.index(known, 12)  # pooled known-file prompts; only AG News items can match AG test items
m, n, k = "qwen36", 300, 12
res = {}
for start in (300,):
    its = short.items(n, start)
    cls = [(i, oa.body(p)) for t, i, p, _ in its if t == "classify"]
    dup = {i for i, b in cls if oa.shingles(b, 12) & idx}
    print("items", start, "duplicates (item numbers):", sorted(dup))
    base_recs = short.read_records(short.run_path(sl.tag(m, sl.WHOLE_ARM, start), n, 0, 32))
    teacher = short.teacher_outputs(base_recs)
    un = short.item_metrics(short.read_records(short.run_path(sl.tag(m, sl.ENDS_ARM, start), n, k, 32)), teacher)[
        "cls_acc"
    ]
    full = short.item_metrics(base_recs, teacher)["cls_acc"]
    out = {}
    for arm, labels in cp.H68_ARMS.items():
        accs, gains = [], []
        for lab in labels:
            sc = short.item_metrics(
                short.read_records(short.run_path(sl.tag(m, sl.ENDS_ARM, start, True, "_" + lab), n, k, 32)), teacher
            )["cls_acc"]
            a, f, u = (keep(x, dup) for x in (sc, full, un))
            accs.append(100 * np.mean(list(a.values())) / np.mean(list(f.values())))
            gains.append(100 * (np.mean(list(a.values())) - np.mean(list(u.values()))))
        out[arm] = (accs, gains)
        print(
            arm,
            "kept",
            [round(x, 1) for x in accs],
            "median",
            round(st.median(accs), 1),
            "| gain",
            [round(x, 1) for x in gains],
            "median",
            round(st.median(gains), 1),
        )
    print(
        "p known>a",
        round(cp.exact_rank_p(out["known"][0], out["a"][0]), 4),
        "p known>dbrand",
        round(cp.exact_rank_p(out["known"][0], out["dbrand"][0]), 4),
        "p dbrand>a",
        round(cp.exact_rank_p(out["dbrand"][0], out["a"][0]), 4),
    )
