"""First-token label scores on classify (H38a): is the collapse to World a label shift, a default, or lost article?

clsprobe.py run --model-tag gemma --k 12 --label stop12 --delta CKPT   # all rule and rotation cells, plus unmasked
clsprobe.py run --model-tag gemma --k 12 --label untrained
clsprobe.py report                                                     # results/clsprobe_h38a.json
clsprobe.py run --model-tag qwen36 --k 12 --label stop12known --delta CKPT --rules ends --suite emotion
clsprobe.py report-h38                                                 # results/clsprobe_h38.json
clsprobe.py report-h63                                                 # results/clsprobe_h63.json
clsprobe.py run --model-tag gemma --k 12 --label stop12 --delta CKPT --rules ends,labels1,labels2 --orders 0,1,2,3
clsprobe.py report-h39                                                 # results/clsprobe_h39.json
clsprobe.py run --model-tag gemma --k 24 --label untrained --rules ends --orders 0,1,2,3
clsprobe.py report-h40                                                 # results/clsprobe_h40.json
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

import selrule as sl
import short

RESULTS = Path(__file__).resolve().parents[1] / "results"
START, N = 300, 300
LIST = "World, Sports, Business, Sci/Tech"
ROTATED = "Sports, Business, Sci/Tech, World"
LABELS = ("World", "Sports", "Business", "Sci/Tech")
COLLAPSE = {"gemma": (12, "stop12"), "qwen36": (12, "stop12"), "nemotron": (8, "untrained")}
COMPARE = {"gemma": (12, "untrained"), "qwen36": (12, "untrained"), "nemotron": (8, "stop8")}
STORED = {
    "gemma": "gemma_d300_ends64_c3b_stop12",
    "qwen36": "qwen36_d300_ends64_c3b_stop12",
    "nemotron": "nemotron_d300_ends64",
}
EMOTIONS = ("sadness", "joy", "love", "anger", "fear", "surprise")
ARMS = {"a": "stop12", "a2": "stop12seed1", "known": "stop12known", "zero": "stop12zero"}
FRONTIER = {"gemma": 24, "nemotron": 16, "qwen36": 24}
MIN_SHIFT = 0.15  # 45 of 300 items
MIN_GAIN = 0.10


def rotate_labels(prompt: str) -> str:
    """The suite's classify prompt with World moved from first to last in the label list."""
    if LIST not in prompt:
        raise ValueError("no label list in this prompt")
    return prompt.replace(LIST, ROTATED, 1)


def rotate_list(prompt: str, labels: tuple[str, ...], n: int = 1) -> str:
    """The prompt with its label list rotated by n: the first n labels move to the end."""
    written = ", ".join(labels)
    if written not in prompt:
        raise ValueError("no label list in this prompt")
    return prompt.replace(written, ", ".join((*labels[n:], *labels[:n])), 1)


def find_spans(ids: list[int], spellings: dict[str, list[list[int]]]) -> dict[str, tuple[int, int]]:
    """First token range of each label, under any of its spellings. Refuses a label it cannot find."""
    out = {}
    for label, forms in spellings.items():
        hits = [
            (i, i + len(form))
            for form in forms
            for i in range(len(ids) - len(form) + 1)
            if ids[i : i + len(form)] == form
        ]
        if not hits:
            raise ValueError(f"label {label} not found")
        out[label] = min(hits)
    return out


def rule_parts(rule: str) -> tuple[str, int]:
    """(selection rule, experts reserved per label): labelsN is ends64 with N experts kept for each label's tokens."""
    return ("ends", int(rule[6:])) if rule.startswith("labels") else (rule, 0)


def order_summary(kept: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(kept)), "worst": min(kept), "gap": max(kept) - min(kept)}


def h39_collapse(rule: dict[str, dict[str, float]], ends: dict[str, dict[str, float]]) -> str:
    """Holds if mean kept across orders rises by MIN_GAIN, with no wider gap, on at least 2 collapse cases."""
    hits = sum(rule[m]["mean"] - ends[m]["mean"] >= MIN_GAIN and rule[m]["gap"] <= ends[m]["gap"] for m in rule)
    return "holds" if hits >= 2 else "fails"


def h39_gap(rule: dict[str, dict[str, float]], ends: dict[str, dict[str, float]]) -> str:
    """Holds if the gap between orders at least halves with no lower mean kept, on at least 2 collapse cases."""
    hits = sum(rule[m]["gap"] <= ends[m]["gap"] / 2 and rule[m]["mean"] >= ends[m]["mean"] for m in rule)
    return "holds" if hits >= 2 else "fails"


def h40_verdict(models: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    """Holds if classify is at the bar in every label order on every model."""
    bars = {m: all(sl.at_bar([d]) for d in cells) for m, cells in models.items()}
    return {"verdict": "holds" if all(bars.values()) else "fails", "at_bar": bars}


def emotion_prompt(text: str) -> str:
    """Classify prompt for SetFit/emotion, in the suite's format; no delta trains on these labels."""
    return f"Classify this text into one of: {', '.join(EMOTIONS)}. Reply with the label only.\n\n{text}"


def emotion_window(start: int) -> tuple[int, int]:
    """Rows of the shuffled emotion test for a run at this item start: 300 gives the first N, 600 the next N."""
    lo = start - START
    return lo, lo + N


def emotion_items(start: int = START) -> list[tuple[str, int, str, str]]:
    from datasets import load_dataset

    lo, hi = emotion_window(start)
    d = load_dataset("SetFit/emotion", split="test").shuffle(seed=0).select(range(lo, hi))
    return [("classify", j, emotion_prompt(r["text"]), r["label_text"]) for j, r in enumerate(d)]


def beats_noise(arm: dict[str, float], a: float, a2: float) -> bool:
    """At the bar, and above both DBpedia deltas by more than the gap between those two seeds."""
    at_bar = arm["mean"] >= 0.90 or arm["lo"] <= 0.90 <= arm["hi"]
    return bool(at_bar and arm["mean"] - max(a, a2) > abs(a - a2))


def article_span(ids: list[int], empty: list[int]) -> tuple[int, int]:
    """Token range of the article: what differs from the same prompt with an empty article."""
    suffix = 0
    while suffix < len(empty) and ids[-1 - suffix] == empty[-1 - suffix]:
        suffix += 1
    prefix = 0
    while prefix < len(empty) - suffix and ids[prefix] == empty[prefix]:
        prefix += 1
    return prefix, len(ids) - suffix


def offset_share(x: np.ndarray) -> float:
    """Share of the per-item label score changes (n, labels) explained by one constant offset per label."""
    c = x - x.mean(axis=1, keepdims=True)
    total = (c**2).sum()
    return float(len(c) * (c.mean(axis=0) ** 2).sum() / total) if total else 0.0


def rotation_verdict(first: dict[str, float], world: dict[str, float]) -> str:
    """First position if, on at least 2 models, rotation raises the new first label and lowers World by MIN_SHIFT."""
    hits = sum(first[m] >= MIN_SHIFT and world[m] <= -MIN_SHIFT for m in first)
    return "first_position" if hits >= 2 else "not_first_position"


def article_verdict(gain: dict[str, float]) -> str:
    """Holds if selecting from the article raises kept accuracy by MIN_GAIN on at least 2 models."""
    return "holds" if sum(g >= MIN_GAIN for g in gain.values()) >= 2 else "fails"


def cell_path(model_tag: str, k: int, label: str, rule: str, rotated: int, suite: str = "", start: int = START) -> Path:
    rot = {0: "", 1: "_rot"}.get(int(rotated), f"_rot{int(rotated)}")
    pre = f"clsprobe_{suite}_" if suite else "clsprobe_"
    at = "" if start == START else f"_d{start}"
    if k == 0:
        return RESULTS / f"{pre}{model_tag}_k0{rot}{at}.jsonl"
    return RESULTS / f"{pre}{model_tag}_k{k}_{label}_{rule}{rot}{at}.jsonl"


def cmd_run(a: argparse.Namespace) -> None:
    from mlx_lm import load

    import c3b
    import segroute as sr
    from models import MODELS

    model, tok = load(MODELS[a.model_tag])
    ctl = sl.install(model, lambda: sl.TAIL, "ends")
    deltas = c3b.attach(model, ctl)
    if a.delta:
        print("deltas loaded:", c3b.load_delta(deltas, a.delta))
    emotion = a.suite == "emotion"
    labels = EMOTIONS if emotion else LABELS
    first = {lab: tok.encode(lab, add_special_tokens=False)[0] for lab in labels}
    items = emotion_items(a.start) if emotion else [x for x in short.items(N, a.start) if x[0] == "classify"]
    orders = [int(x) for x in a.orders.split(",")]
    spellings = {
        lab: [tok.encode(lab, add_special_tokens=False), tok.encode(" " + lab, add_special_tokens=False)]
        for lab in labels
    }
    cells = [(0, "", "", rot) for rot in orders]
    cells += [(a.k, a.label, rule, rot) for rule in a.rules.split(",") for rot in orders]
    for k, label, rule, rot in cells:
        path = cell_path(a.model_tag, k, label, rule, rot, a.suite, a.start)
        if path.exists() and sum(1 for _ in path.open()) == N:
            continue
        ctl.mode, ctl.k, ctl.s = ("mask", k, 32) if k else ("off", None, None)
        ctl.rule, ctl.reserve_m = rule_parts(rule or "ends")
        with path.open("w") as f:
            for _, i, prompt, ref in items:
                text = rotate_list(prompt, labels, rot)
                ids = sr.chat(tok, text)
                ctl.reset()
                ctl.span = article_span(ids, sr.chat(tok, text.split("\n\n", 1)[0] + "\n\n"))
                ctl.reserved = list(find_spans(ids, spellings).values()) if ctl.reserve_m else []
                logits = model(mx.array(ids)[None])[0, -1].astype(mx.float32)
                lp = logits - mx.logsumexp(logits)
                mx.eval(lp)
                scores = {lab: lp[t].item() for lab, t in first.items()}
                f.write(json.dumps({"i": i, "ref": ref, "lp": scores, "top": mx.argmax(lp).item()}) + "\n")
        print(path.name, flush=True)


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.open()]


def _pred(r: dict) -> str:
    return max(LABELS, key=lambda lab: r["lp"][lab])


def _hit(r: dict, labels: tuple[str, ...] = LABELS) -> bool:
    """Whether the best-scoring label is the dataset label. Shipped records carry `hit` and no label text."""
    if "hit" in r:
        return bool(r["hit"])
    return max(labels, key=lambda lab: r["lp"][lab]) == r["ref"]


def _stats(recs: list[dict], base: list[dict]) -> dict[str, Any]:
    acc = float(np.mean([_hit(r) for r in recs]))
    base_acc = float(np.mean([_hit(r) for r in base]))
    x = np.array([[r["lp"][lab] - b["lp"][lab] for lab in LABELS] for r, b in zip(recs, base, strict=True)])
    return {
        "accuracy": acc,
        "kept": acc / base_acc,
        "share": {lab: float(np.mean([_pred(r) == lab for r in recs])) for lab in LABELS},
        "offset_share": offset_share(x),
        "offset": dict(zip(LABELS, (x - x.mean(axis=1, keepdims=True)).mean(axis=0).tolist(), strict=True)),
    }


def report() -> dict:
    """Write results/clsprobe_h38a.json; first-token agreement with the stored run validates the probe."""
    out: dict[str, Any] = {"models": {}}
    first_gain, world_gain, article_gain = {}, {}, {}
    for m, (k, label) in COLLAPSE.items():
        base = {rot: _read(cell_path(m, 0, "", "", rot)) for rot in (False, True)}
        row: dict[str, Any] = {"unmasked": {str(rot): _stats(base[rot], base[rot]) for rot in base}}
        for kk, lab in (COLLAPSE[m], COMPARE[m]):
            for rule in ("ends", "article"):
                for rot in (False, True):
                    recs = _read(cell_path(m, kk, lab, rule, rot))
                    row[f"{lab}_{rule}{'_rot' if rot else ''}"] = _stats(recs, base[rot])
        stored = {
            r["i"]: r["gen_ids"][:1]
            for r in short.read_records(short.run_path(STORED[m], N, k, 32))
            if r["task"] == "classify"
        }
        probe = _read(cell_path(m, k, label, "ends", False))
        row["agrees_with_stored"] = float(np.mean([stored[r["i"]] == [r["top"]] for r in probe]))
        out["models"][m] = row
        a, b = row[f"{label}_ends"], row[f"{label}_ends_rot"]
        first_gain[m] = b["share"]["Sports"] - a["share"]["Sports"]
        world_gain[m] = b["share"]["World"] - a["share"]["World"]
        article_gain[m] = row[f"{label}_article"]["kept"] - a["kept"]
    out["rotation"] = rotation_verdict(first_gain, world_gain)
    out["article"] = article_verdict(article_gain)
    out["gains"] = {"first": first_gain, "world": world_gain, "article": article_gain}
    (RESULTS / "clsprobe_h38a.json").write_text(json.dumps(out, indent=1))
    return out


def _kept(recs: list[dict], base: list[dict], labels: tuple[str, ...]) -> dict[str, float]:
    def right(rows: list[dict]) -> dict[int, float]:
        return {r["i"]: float(_hit(r, labels)) for r in rows}

    return sl.kept_ci(right(recs), right(base))


def _arm_row(
    m: str, k: int, n: int, label: str, teacher: Any, scores: Any, suite_base: Any, start: int = START
) -> dict[str, Any]:
    """One arm's kept scores, loops, loads, and AG News and emotion probes, listed and rotated."""
    recs = short.read_records(short.run_path(sl.tag(m, sl.ENDS_ARM, start, True, f"_{label}"), n, k, 32))
    off = short.read_records(short.run_path(sl.tag(m, sl.ENDS_ARM, start, True, f"_{label}"), n, 0, 32))
    s = short.item_metrics(recs, teacher)
    kept = [sl.kept_ci(s[x], scores[x]) for x in sl.KEPT_METRICS]
    row: dict[str, Any] = {
        **dict(zip(sl.KEPT_METRICS, kept, strict=True)),
        "at_bar": sl.at_bar(kept),
        "unmasked_identical": sl.identical_share(off, suite_base),
        "loops": {t: sl.loop_rate(recs, t) for t in ("summarize", "rewrite")},
        "loads_per_layer": float(np.mean([r["loads_per_layer"] for r in recs])),
    }
    for rot in (False, True):
        tagged = "_rot" if rot else ""
        ag = _read(cell_path(m, k, label, "ends", rot, "", start))
        em = _read(cell_path(m, k, label, "ends", rot, "emotion", start))
        row[f"ag{tagged}"] = _kept(ag, _read(cell_path(m, 0, "", "", rot, "", start)), LABELS)
        row[f"emotion{tagged}"] = _kept(em, _read(cell_path(m, 0, "", "", rot, "emotion", start)), EMOTIONS)
    return row


def report_h38() -> dict:
    """Write results/clsprobe_h38.json: known and zero-shot deltas against two seeds of the DBpedia delta, Qwen3.6."""
    m, k, n = "qwen36", 12, N
    suite_base = short.read_records(short.run_path(sl.tag(m, sl.WHOLE_ARM, START), n, 0, 32))
    teacher = short.teacher_outputs(suite_base)
    scores = short.item_metrics(suite_base, teacher)
    arms = {name: _arm_row(m, k, n, label, teacher, scores, suite_base) for name, label in ARMS.items()}
    a, a2 = arms["a"]["cls_acc"]["mean"], arms["a2"]["cls_acc"]["mean"]
    known = beats_noise(arms["known"]["cls_acc"], a, a2)
    zero = beats_noise(arms["zero"]["cls_acc"], a, a2) and sl.at_bar([arms["zero"]["emotion"]])
    identical = all(r["unmasked_identical"] == 1.0 for r in arms.values())
    out = {
        "known": "holds" if known else "fails",
        "zero": "holds" if zero else "fails",
        "valid": identical,
        "arms": arms,
    }
    (RESULTS / "clsprobe_h38.json").write_text(json.dumps(out, indent=1))
    return out


H63_ARMS = {
    "a": ["stop12", "stop12seed1", "stop12seed2", "stop12seed3"],
    "known": ["stop12known", "stop12knownseed1", "stop12knownseed2"],
}


def h63_gate(a_cls: list[dict[str, float]], known_cls: list[dict[str, float]]) -> bool:
    """Known holds if every known seed beats every a seed and the lowest known seed is at the bar."""
    low = min(known_cls, key=lambda c: c["mean"])
    at_bar = low["mean"] >= 0.90 or low["lo"] <= 0.90 <= low["hi"]
    return bool(at_bar and low["mean"] > max(c["mean"] for c in a_cls))


def report_h63() -> dict:
    """Write results/clsprobe_h63.json: four seeds of arm a and three of arm known, Qwen3.6."""
    m, k, n = "qwen36", 12, N
    suite_base = short.read_records(short.run_path(sl.tag(m, sl.WHOLE_ARM, START), n, 0, 32))
    teacher = short.teacher_outputs(suite_base)
    scores = short.item_metrics(suite_base, teacher)
    arms = {
        name: [_arm_row(m, k, n, lab, teacher, scores, suite_base) for lab in labels]
        for name, labels in H63_ARMS.items()
    }
    cls = {name: [r["cls_acc"] for r in rows] for name, rows in arms.items()}
    means = {name: sorted(c["mean"] for c in v) for name, v in cls.items()}
    out = {
        "known": "holds" if h63_gate(cls["a"], cls["known"]) else "fails",
        "valid": all(r["unmasked_identical"] == 1.0 for rows in arms.values() for r in rows),
        "classify_sorted": means,
        "classify_median": {name: float(np.median(v)) for name, v in means.items()},
        "arms": arms,
    }
    (RESULTS / "clsprobe_h63.json").write_text(json.dumps(out, indent=1))
    return out


H64_ARMS = {
    "a": [f"stop12seed{i}" if i else "stop12" for i in range(6)],
    "known": [f"stop12knownseed{i}" if i else "stop12known" for i in range(6)],
}
H64_METRICS = ("cls_acc", "ag_rot", "emotion", "emotion_rot", "sum_rougeL_ref", "qa_f1")


def exact_rank_p(known: list[float], a: list[float], two_sided: bool = False) -> float:
    """Exact Mann-Whitney p: share of label splits whose known rank sum is at least (or as far as) the observed one."""
    pooled = known + a
    ranks = {
        i: sum(v < pooled[i] for v in pooled) + (sum(v == pooled[i] for v in pooled) + 1) / 2
        for i in range(len(pooled))
    }
    k = len(known)
    total = sum(ranks.values())
    mid = k * total / len(pooled)
    obs = sum(ranks[i] for i in range(k))
    sums = [sum(ranks[i] for i in c) for c in itertools.combinations(range(len(pooled)), k)]
    if two_sided:
        return sum(abs(x - mid) >= abs(obs - mid) - 1e-9 for x in sums) / len(sums)
    return sum(x >= obs - 1e-9 for x in sums) / len(sums)


def report_h64() -> dict:
    """Write results/clsprobe_h64.json: six seeds per arm, medians, ranges and exact rank tests, Qwen3.6."""
    m, k, n = "qwen36", 12, N
    suite_base = short.read_records(short.run_path(sl.tag(m, sl.WHOLE_ARM, START), n, 0, 32))
    teacher = short.teacher_outputs(suite_base)
    scores = short.item_metrics(suite_base, teacher)
    arms = {
        name: [_arm_row(m, k, n, lab, teacher, scores, suite_base) for lab in labels]
        for name, labels in H64_ARMS.items()
    }
    metrics: dict[str, Any] = {}
    for met in H64_METRICS:
        v = {name: [100 * r[met]["mean"] for r in rows] for name, rows in arms.items()}
        metrics[met] = {
            **{
                name: {"seeds": x, "median": float(np.median(x)), "min": min(x), "max": max(x)} for name, x in v.items()
            },
            "p_known_above_a": exact_rank_p(v["known"], v["a"]),
            "p_two_sided": exact_rank_p(v["known"], v["a"], two_sided=True),
        }
    out = {
        "valid": all(r["unmasked_identical"] == 1.0 for rows in arms.values() for r in rows),
        "metrics": metrics,
        "loops": {name: [r["loops"] for r in rows] for name, rows in arms.items()},
        "arms": arms,
    }
    (RESULTS / "clsprobe_h64.json").write_text(json.dumps(out, indent=1))
    return out


H68_ARMS = {
    **H64_ARMS,
    "dbrand": [f"stop12dbrandseed{i}" if i else "stop12dbrand" for i in range(6)],
}
H68_MEASURES = ("cls_acc", "ag_rot")
H68_ALPHA = 0.05


def h68_reading(p: dict[str, tuple[float, float]]) -> str:
    """Apply the fixed rule of H68. p[measure] = (p of A: dbrand above a, p of B: known above dbrand)."""
    names = {(True, False): "order explains", (False, True): "data explains", (True, True): "both contribute"}
    verdicts = {names.get((pa <= H68_ALPHA, pb <= H68_ALPHA), "inconclusive") for pa, pb in p.values()}
    return verdicts.pop() if len(verdicts) == 1 else "split"


def report_h68() -> dict:
    """Write results/clsprobe_h68.json: label-order control, three arms of six seeds, Qwen3.6, items 300 to 599."""
    m, k, n = "qwen36", 12, N
    suite_base = short.read_records(short.run_path(sl.tag(m, sl.WHOLE_ARM, START), n, 0, 32))
    teacher = short.teacher_outputs(suite_base)
    scores = short.item_metrics(suite_base, teacher)
    arms = {
        name: [_arm_row(m, k, n, lab, teacher, scores, suite_base) for lab in labels]
        for name, labels in H68_ARMS.items()
    }
    metrics: dict[str, Any] = {}
    for met in (*H64_METRICS,):
        v = {name: [100 * r[met]["mean"] for r in rows] for name, rows in arms.items()}
        metrics[met] = {
            **{
                name: {"seeds": x, "median": float(np.median(x)), "min": min(x), "max": max(x)} for name, x in v.items()
            },
            "p_dbrand_above_a": exact_rank_p(v["dbrand"], v["a"]),
            "p_known_above_dbrand": exact_rank_p(v["known"], v["dbrand"]),
        }
    reading = h68_reading(
        {x: (metrics[x]["p_dbrand_above_a"], metrics[x]["p_known_above_dbrand"]) for x in H68_MEASURES}
    )
    out = {
        "reading": reading,
        "valid": all(r["unmasked_identical"] == 1.0 for rows in arms.values() for r in rows),
        "metrics": metrics,
        "loops": {name: [r["loops"] for r in rows] for name, rows in arms.items()},
        "arms": arms,
    }
    (RESULTS / "clsprobe_h68.json").write_text(json.dumps(out, indent=1))
    return out


def paired_diff(arm: dict[Any, float], base: dict[Any, float]) -> dict[str, float]:
    """Mean of (arm - base) in points on matched items, with a 95% bootstrap CI (2,000 resamples)."""
    keys = sorted(arm.keys() & base.keys())
    d = 100 * (np.array([arm[k] for k in keys]) - np.array([base[k] for k in keys]))
    idx = np.random.default_rng(0).integers(0, len(keys), size=(2000, len(keys)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return {"n": len(keys), "mean": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def report_h68_paired() -> dict:
    """Write results/clsprobe_h68_paired.json: each seed's AG News (generated) and QA gain over the untrained list."""
    m, k, n = "qwen36", 12, N
    suite_base = short.read_records(short.run_path(sl.tag(m, sl.WHOLE_ARM, START), n, 0, 32))
    teacher = short.teacher_outputs(suite_base)
    base = short.item_metrics(short.read_records(short.run_path(sl.tag(m, sl.ENDS_ARM, START), n, k, 32)), teacher)
    out: dict[str, Any] = {}
    for name, labels in H68_ARMS.items():
        rows = []
        for lab in labels:
            recs = short.read_records(short.run_path(sl.tag(m, sl.ENDS_ARM, START, True, f"_{lab}"), n, k, 32))
            sc = short.item_metrics(recs, teacher)
            rows.append({met: paired_diff(sc[met], base[met]) for met in ("cls_acc", "qa_f1")})
        out[name] = rows
    (RESULTS / "clsprobe_h68_paired.json").write_text(json.dumps(out, indent=1))
    return out


H65_LABELS = {"a": "stop12seed6", "known": "stop12knownseed6"}
FINAL_START = 600


def report_h65() -> dict:
    """Write results/clsprobe_h65.json: the one test on items 600 to 899, two fresh seeds, Qwen3.6."""
    m, k, n = "qwen36", 12, N
    suite_base = short.read_records(short.run_path(sl.tag(m, sl.WHOLE_ARM, FINAL_START), n, 0, 32))
    teacher = short.teacher_outputs(suite_base)
    scores = short.item_metrics(suite_base, teacher)
    arms = {name: _arm_row(m, k, n, lab, teacher, scores, suite_base, FINAL_START) for name, lab in H65_LABELS.items()}
    holds = all(sl.at_bar([r["sum_rougeL_ref"], r["qa_f1"]]) for r in arms.values())
    out = {
        "holds": "holds" if holds else "fails",
        "valid": all(r["unmasked_identical"] == 1.0 for r in arms.values()),
        "arms": arms,
    }
    (RESULTS / "clsprobe_h65.json").write_text(json.dumps(out, indent=1))
    return out


def report_h39() -> dict:
    """Write results/clsprobe_h39.json: four label orders, ends64 against reserved label experts."""
    orders = (0, 1, 2, 3)
    models: dict[str, Any] = {}
    collapse: dict[str, dict[str, dict[str, float]]] = {r: {} for r in ("ends", "labels1", "labels2")}
    for m, (k, label) in COLLAPSE.items():
        base = {o: _read(cell_path(m, 0, "", "", o)) for o in orders}
        row: dict[str, Any] = {"unmasked_accuracy": [_stats(base[o], base[o])["accuracy"] for o in orders]}
        for lab in (label, COMPARE[m][1]):
            for rule in collapse:
                cells = [_stats(_read(cell_path(m, k, lab, rule, o)), base[o]) for o in orders]
                kept = [c["kept"] for c in cells]
                row[f"{lab}_{rule}"] = {**order_summary(kept), "kept": kept, "shares": [c["share"] for c in cells]}
                if lab == label:
                    collapse[rule][m] = row[f"{lab}_{rule}"]
        models[m] = row
    out = {
        "collapse": {r: h39_collapse(collapse[r], collapse["ends"]) for r in ("labels1", "labels2")},
        "gap": {r: h39_gap(collapse[r], collapse["ends"]) for r in ("labels1", "labels2")},
        "models": models,
    }
    (RESULTS / "clsprobe_h39.json").write_text(json.dumps(out, indent=1))
    return out


def report_h40() -> dict:
    """Write results/clsprobe_h40.json: untrained frontier budgets in four label orders, emotion in two."""
    orders = (0, 1, 2, 3)
    models: dict[str, Any] = {}
    for m, k in FRONTIER.items():
        base = {o: _read(cell_path(m, 0, "", "", o)) for o in orders}
        recs = {o: _read(cell_path(m, k, "untrained", "ends", o)) for o in orders}
        cells = [_kept(recs[o], base[o], LABELS) for o in orders]
        emotion = [
            _kept(
                _read(cell_path(m, k, "untrained", "ends", o, "emotion")),
                _read(cell_path(m, 0, "", "", o, "emotion")),
                EMOTIONS,
            )
            for o in (0, 1)
        ]
        models[m] = {
            "k": k,
            "orders": cells,
            **order_summary([d["mean"] for d in cells]),
            "shares": [_stats(recs[o], base[o])["share"] for o in orders],
            "emotion": emotion,
        }
    out = {**h40_verdict({m: r["orders"] for m, r in models.items()}), "models": models}
    (RESULTS / "clsprobe_h40.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model-tag", required=True)
    r.add_argument("--k", type=int, required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--delta")
    r.add_argument("--rules", default="ends,article")
    r.add_argument("--orders", default="0,1")
    r.add_argument("--suite", default="", choices=["", "emotion"])
    r.add_argument("--start", type=int, default=START, help="first item of each source; 300 or 600")
    sub.add_parser("report")
    sub.add_parser("report-h38")
    sub.add_parser("report-h63")
    sub.add_parser("report-h64")
    sub.add_parser("report-h65")
    sub.add_parser("report-h68")
    sub.add_parser("report-h68-paired")
    sub.add_parser("report-h39")
    sub.add_parser("report-h40")
    a = p.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        reports = {
            "report": report,
            "report-h38": report_h38,
            "report-h63": report_h63,
            "report-h64": report_h64,
            "report-h65": report_h65,
            "report-h68": report_h68,
            "report-h68-paired": report_h68_paired,
            "report-h39": report_h39,
            "report-h40": report_h40,
        }
        print(json.dumps(reports[a.cmd](), indent=1))


if __name__ == "__main__":
    main()
