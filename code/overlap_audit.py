"""Audit: does any training prompt share text with a test item?

Reads the exact training prompts (token ids saved in distill_data/*teacher*.jsonl), decodes them, and looks for
shared word windows with the test texts of the short suite and with the whole test splits.

    .venv/bin/python overlap_audit.py            # writes results/overlap_audit.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
WINDOW = 12


def body(prompt: str) -> str:
    """The item text of a prompt: no chat template, no instruction line (the part before the first blank line)."""
    text = prompt.split("<|im_end|>")[0].removeprefix("<|im_start|>user\n")
    return text.split("\n\n", 1)[1] if "\n\n" in text else text


def shingles(text: str, n: int) -> set[tuple[str, ...]]:
    """All windows of n consecutive lower-case words."""
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def index(texts: list[str], n: int) -> set[tuple[str, ...]]:
    """The set of windows of all training texts."""
    out: set[tuple[str, ...]] = set()
    for t in texts:
        out |= shingles(t, n)
    return out


def hits(tests: list[str], train: set[tuple[str, ...]], n: int) -> list[int]:
    """Positions of the test texts that share at least one window with the training index."""
    return [i for i, t in enumerate(tests) if shingles(t, n) & train]


def training_texts(by_file: bool = False) -> dict[str, list[str]]:
    """Decoded training prompts of every recipe's teacher file, by task (or by file)."""
    from transformers import AutoTokenizer

    import c3b

    tok = AutoTokenizer.from_pretrained(c3b.MODELS["qwen36"])
    out: dict[str, list[str]] = {}
    seen: set[str] = set()
    for f in sorted((HERE / "distill_data").glob("qwen36_c3b_teacher*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            text = body(tok.decode(r["prompt"]))
            if by_file:
                out.setdefault(f.stem, []).append(text)
                continue
            if text in seen:
                continue
            seen.add(text)
            out.setdefault(r["task"], []).append(text)
    return out


def used_items(train_by_file: dict[str, list[str]]) -> dict:
    """Overlap of the scored items (300 to 599 and 600 to 899, every task) with each teacher file separately."""
    import short

    out: dict = {}
    for start in (300, 600):
        its = short.items(300, start)
        for name, texts in train_by_file.items():
            idx = index(texts, WINDOW)
            for task in ("summarize", "qa", "classify", "rewrite"):
                rows = [body(p) for t, _, p, _ in its if t == task]
                h = hits(rows, idx, WINDOW)
                out.setdefault(f"items {start} to {start + 299}", {}).setdefault(name, {})[task] = {
                    "items": len(rows),
                    "sharing_a_window": len(h),
                }
    return out


def main() -> None:
    from datasets import load_dataset

    train = training_texts()
    pooled = [t for rows in train.values() for t in rows]
    idx = index(pooled, WINDOW)
    tests = {
        "summarize/rewrite, CNN/DailyMail test (all)": [
            r["article"][:3000] for r in load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
        ],
        "qa, SQuAD validation (all), context and question": [
            r["context"] + " " + r["question"] for r in load_dataset("rajpurkar/squad", split="validation")
        ],
        "classify, AG News test (all)": list(load_dataset("fancyzhx/ag_news", split="test")["text"]),
        "emotion test (all)": list(load_dataset("SetFit/emotion", split="test")["text"]),
    }
    out: dict = {"window_words": WINDOW, "training_prompts": {k: len(v) for k, v in train.items()}, "tests": {}}
    for name, texts in tests.items():
        h = hits(texts, idx, WINDOW)
        out["tests"][name] = {"items": len(texts), "sharing_a_window": len(h), "first_positions": h[:10]}
        print(name, len(texts), len(h), flush=True)
    out["scored_items_by_teacher_file"] = used_items(training_texts(by_file=True))
    print(json.dumps(out["scored_items_by_teacher_file"], indent=1), flush=True)
    (HERE / "results" / "overlap_audit.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
