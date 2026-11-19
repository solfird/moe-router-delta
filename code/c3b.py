"""C3b, mode-isolated masked distillation: a router delta that is active only under the mask.

The original router is kept and called as is; DeltaProj adds x @ delta.T to its logits only when the controller is in
"mask" or "train" mode, so unmasked outputs are bit-identical to the base model. Training is teacher-forced in one
parallel pass, with the segment sets of segroute.Controller (groups below).
"""

import argparse
import json
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from adapters import SIGMOID_BIAS, SOFTMAX_LINEAR, GateTap, find_routers
from models import MODELS
from segroute import Controller

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "distill_data"  # teacher answers; not in this repository
CKPT = HERE / "distill_ckpt"  # trained deltas; not in this repository


def groups(n_tok, n_prompt, s):
    """Token ranges sharing one expert set, and the range each set is chosen from.

    Matches segroute.Controller: the prompt set serves the prompt and the first S generated tokens;
    set g >= 1 serves [n_prompt + gS, n_prompt + (g+1)S) and is chosen from the S tokens before it.
    """
    out = [((0, min(n_tok, n_prompt + s)), (0, n_prompt))]
    g = 1
    while n_prompt + g * s < n_tok:
        a = n_prompt + g * s
        out.append(((a, min(n_tok, a + s)), (a - s, a)))
        g += 1
    return out


ACTIVE = ("mask", "train")


def _shape(inner: nn.Module) -> tuple[int, int]:
    """(experts, hidden) of a router projection, quantized or not."""
    if hasattr(inner, "scales"):
        return inner.weight.shape[0], inner.scales.shape[-1] * inner.group_size
    return inner.weight.shape[0], inner.weight.shape[1]


def segment_mask(probs: mx.array, n_prompt: int, k: int, s: int, ends: int | None = None) -> mx.array:
    """Additive (T, E) mask: 0 on the K' experts of each token's segment set, -1e9 elsewhere.

    ends: the prompt set comes from this many prompt tokens, half from each end (selrule's "ends" rule). None: all.
    """
    n_tok, n_exp = probs.shape
    rows = []
    for (a, b), (c, d) in groups(n_tok, n_prompt, s):
        source = probs[c:d]
        if a == 0 and ends is not None and n_prompt > ends:
            source = mx.concatenate([probs[: ends // 2], probs[n_prompt - (ends - ends // 2) : n_prompt]], axis=0)
        top = mx.argsort(-source.sum(axis=0))[:k]
        keep = (mx.arange(n_exp)[None, :] == top[:, None]).any(axis=0)
        rows.append(mx.broadcast_to(mx.where(keep, 0.0, -1e9), (b - a, n_exp)))
    return mx.concatenate(rows, axis=0)


class DeltaProj(nn.Module):
    """Router projection plus a trainable delta that is applied only in mask and train modes."""

    def __init__(self, inner: nn.Module, ctl: Any) -> None:
        super().__init__()
        inner.freeze()
        self.inner = inner
        self.delta = mx.zeros(_shape(inner), dtype=mx.float32)
        self._ctl = [ctl]  # list keeps it out of the parameter tree

    def __call__(self, x: mx.array) -> mx.array:
        out = self.inner(x)
        ctl = self._ctl[0]
        if ctl.mode not in ACTIVE:
            return out
        base = out.astype(mx.float32)
        logits = base + x.astype(mx.float32) @ self.delta.T
        ctx = getattr(ctl, "c3b", None)
        if ctl.mode == "train" and ctx is not None:  # parallel teacher-forced pass: mask and prompt anchor here
            n_prompt = ctx["n_prompt"]
            p = mx.softmax(logits[0], axis=-1)
            p0 = mx.softmax(base[0, :n_prompt], axis=-1)
            ctx["drift"].append((p0 * (mx.log(p0 + 1e-9) - mx.log(p[:n_prompt] + 1e-9))).sum(axis=-1).mean())
            logits = logits + segment_mask(mx.stop_gradient(p), n_prompt, ctx["k"], ctx["s"], ctx.get("ends"))[None]
        return logits.astype(out.dtype)


class DeltaGate(GateTap):
    """Nemotron-H gate tap plus a trainable delta on the pre-sigmoid logits, applied only in mask and train modes."""

    def __init__(self, gate: nn.Module, layer: int, ctl: Any) -> None:
        super().__init__(gate, layer, ctl)
        gate.freeze()
        self.delta = mx.zeros(gate.weight.shape, dtype=mx.float32)
        self._base: list[mx.array] = []  # base scores of the last call, train mode only

    def logits(self, x: mx.array) -> mx.array:
        base = super().logits(x)
        if self._ctl[0].mode not in ACTIVE:
            return base
        self._base = [mx.sigmoid(base[0])]
        return base + x.astype(mx.float32) @ self.delta.T

    def bias(self, weights: mx.array, scores: mx.array) -> mx.array | None:
        ctl = self._ctl[0]
        ctx = getattr(ctl, "c3b", None)
        if ctl.mode != "train" or ctx is None:
            return super().bias(weights, scores)
        n_prompt = ctx["n_prompt"]  # prompt anchor on the sigmoid scores, normalized over the experts
        p = scores[:n_prompt] / scores[:n_prompt].sum(axis=-1, keepdims=True)
        p0 = self._base[0][:n_prompt] / self._base[0][:n_prompt].sum(axis=-1, keepdims=True)
        ctx["drift"].append((p0 * (mx.log(p0 + 1e-9) - mx.log(p + 1e-9))).sum(axis=-1).mean())
        return segment_mask(mx.stop_gradient(weights), n_prompt, ctx["k"], ctx["s"], ctx.get("ends"))[None]


def attach(model: nn.Module, ctl: Controller) -> list[DeltaProj | DeltaGate]:
    """Insert a delta in every segroute tap (call after segroute.install): DeltaProj in a softmax-linear Tap,
    DeltaGate in place of a Nemotron GateTap."""
    out: list[DeltaProj | DeltaGate] = []
    for owner, name, kind in find_routers(model):
        tap = getattr(owner, name)
        if kind == SOFTMAX_LINEAR:
            tap.inner = DeltaProj(tap.inner, ctl)
            out.append(tap.inner)
        elif kind == SIGMOID_BIAS:
            setattr(owner, name, DeltaGate(tap.gate, tap.layer, ctl))
            out.append(getattr(owner, name))
        else:
            raise NotImplementedError(f"C3b has no delta for {kind} routers")
    return out


def prepare_training(model: nn.Module, deltas: list[DeltaProj | DeltaGate]) -> None:
    """Freeze all but the deltas; train mode, which sends Qwen3.6 gated-delta layers off the kernel that has no vjp."""
    model.freeze()
    for d in deltas:
        d.unfreeze(keys=["delta"])
    model.train()


@contextmanager
def approx_linear_attention() -> Iterator[None]:
    """Qwen3.6 gated-delta layers on the Metal kernel, with no gradient through the recurrence (the kernel has no vjp).

    The forward is unchanged. The gradient still reaches each delta through the residual stream, the z gate and
    projections of the linear-attention blocks, the MoE layers and the full-attention layers.
    """
    import mlx_lm.models.qwen3_5 as q35

    orig = q35.gated_delta_update

    def cut(*args: Any, **kw: Any) -> tuple[mx.array, mx.array]:
        args = tuple(mx.stop_gradient(x) if isinstance(x, mx.array) else x for x in args)
        kw = {k: mx.stop_gradient(v) if isinstance(v, mx.array) else v for k, v in kw.items()}
        return orig(*args, **{**kw, "use_kernel": True})

    q35.gated_delta_update = cut
    try:
        yield
    finally:
        q35.gated_delta_update = orig


def _answer_logits(model: nn.Module, ids: list[int], gen: list[int]) -> mx.array:
    """Float32 log-probs of each answer token, from one pass over prompt + answer[:-1]."""
    logits = model(mx.array(ids + gen[:-1])[None])[0, len(ids) - 1 :].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _masked(
    model: nn.Module, ctl: Controller, ids: list[int], gen: list[int], k: int, s: int, ends: int | None = None
) -> tuple[mx.array, mx.array]:
    ctl.reset()
    ctl.mode, ctl.c3b = "train", {"n_prompt": len(ids), "k": k, "s": s, "ends": ends, "drift": []}
    try:
        lp = _answer_logits(model, ids, gen)
        drift = mx.stack(ctl.c3b["drift"]).mean()
    finally:
        ctl.mode, ctl.c3b = "off", None
    return lp, drift


def masked_logprobs(model: nn.Module, ctl: Controller, ids: list[int], gen: list[int], k: int, s: int) -> mx.array:
    """Masked log-probs of each answer token, teacher-forced, in one parallel pass."""
    return _masked(model, ctl, ids, gen, k, s)[0]


def teacher_logprobs(model: nn.Module, ctl: Controller, ids: list[int], gen: list[int]) -> mx.array:
    """Unmasked log-probs of each answer token: the base model, since the delta is off."""
    ctl.reset()
    ctl.mode = "off"
    return mx.stop_gradient(_answer_logits(model, ids, gen))


def loss_fn(
    model: nn.Module,
    ctl: Controller,
    ids: list[int],
    gen: list[int],
    t_lp: mx.array,
    k: int,
    s: int,
    mu: float,
    ends: int | None = None,
) -> mx.array:
    """KL(teacher || masked student) on answer tokens plus mu x KL(original || corrected) router on prompt tokens."""
    s_lp, drift = _masked(model, ctl, ids, gen, k, s, ends)
    kl = (mx.exp(t_lp) * (t_lp - s_lp)).sum(axis=-1).mean()
    return kl + mu * drift


def with_stop(recs: list[dict], eos: int, limits: dict[str, int]) -> list[dict]:
    """Teacher answers with the end token added where the teacher stopped before its token limit.

    _answer_logits never feeds the last token, so this only adds the stop position to the KL.
    """
    return [{**r, "answer": r["answer"] + [eos]} if len(r["answer"]) < limits[r["task"]] else r for r in recs]


RECIPES = {  # classify items per source, teacher data of H38
    "known": {"dbpedia": 500, "ag_news": 500},
    "zero": {"dbpedia": 334, "newsgroups": 333, "amazon": 333},
    "dbrand": {"dbpedia": 1000},  # H68: the DBpedia-only items with the random label order of the mixed arm
}


def mix_prompts(sources: list[tuple[list[str], list[str]]], counts: list[int], seed: int = 0) -> list[str]:
    """Classify prompts from each (labels, texts) source, with the label list in a new random order on every item."""
    rng = np.random.default_rng(seed)
    out = []
    for (labels, texts), n in zip(sources, counts, strict=True):
        for text in texts[:n]:
            order = ", ".join(labels[j] for j in rng.permutation(len(labels)))
            out.append(f"Classify this item into one of: {order}. Reply with the label only.\n\n{text}")
    return out


def recipe_prompts(recipe: str, seed: int = 0) -> list[str]:
    """Classify training prompts of one H38 recipe, from sets and splits the suite does not use."""
    from datasets import load_dataset

    def source(name: str, n: int) -> tuple[list[str], list[str]]:
        if name == "dbpedia":
            d = load_dataset("fancyzhx/dbpedia_14", split="train").shuffle(seed=seed).select(range(n))
            return d.features["label"].names, list(d["content"])
        if name == "ag_news":
            d = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=seed).select(range(n))
            return d.features["label"].names, list(d["text"])
        if name == "newsgroups":
            d = load_dataset("SetFit/20_newsgroups", split="train").shuffle(seed=seed)
            d = d.filter(lambda r: len(r["text"]) > 40).select(range(n))
            return sorted(set(d["label_text"])), [x[:1500] for x in d["text"]]
        d = load_dataset("fancyzhx/amazon_polarity", split="train").shuffle(seed=seed).select(range(n))
        return d.features["label"].names, list(d["content"])

    plan = RECIPES[recipe]
    return mix_prompts([source(name, n) for name, n in plan.items()], list(plan.values()), seed)


def replace_classify(rows: list[dict], classify: list[dict]) -> list[dict]:
    """Teacher rows with the classify task replaced."""
    return [r for r in rows if r["task"] != "classify"] + classify


def teacher_path(model_tag: str, recipe: str) -> Path:
    return DATA / f"{model_tag}_c3b_teacher{'_' + recipe if recipe else ''}.jsonl"


def run_name(a: argparse.Namespace) -> str:
    """Checkpoint folder of one training run."""
    suffix = "_approx" if a.approx_la else "_chunked" if a.chunked_la else ""
    ends = getattr(a, "ends", None)
    rule = f"_ends{ends}" if ends else ""
    stop = "_stop" if getattr(a, "stop", False) else ""
    recipe = f"_{a.recipe}" if getattr(a, "recipe", "") else ""
    seed = f"_seed{a.seed}" if getattr(a, "seed", 0) else ""
    return f"{a.model_tag}_c3b_k{a.k}_s{a.s}_mu{a.mu}{rule}{stop}{recipe}{seed}{suffix}"


def keep_mrqa(rows: Iterable[dict]) -> list[dict]:
    """MRQA rows outside its SQuAD subset (the short suite evaluates on SQuAD)."""
    return [r for r in rows if r["subset"] != "SQuAD"]


def training_items(n_per_task: int, seed: int = 0) -> list[tuple[str, str]]:
    """(task, prompt) in the short-suite formats, from sets the suite does not use: XSum, MRQA minus SQuAD, DBpedia."""
    from datasets import load_dataset

    out: list[tuple[str, str]] = []
    xsum = load_dataset("EdinburghNLP/xsum", split="train").shuffle(seed=seed)
    for r in xsum.select(range(n_per_task)):
        out.append(("summarize", "Summarize this news article in 2 or 3 sentences.\n\n" + r["document"][:3000]))
    for r in xsum.select(range(n_per_task, 2 * n_per_task)):
        para = r["document"][:700].rsplit(".", 1)[0] + "."
        out.append(("rewrite", "Rewrite this paragraph in plain, simple English. Keep all the facts.\n\n" + para))
    mrqa = load_dataset("mrqa-workshop/mrqa", split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    qa = keep_mrqa(r for r, _ in zip(mrqa.filter(lambda r: r["subset"] != "SQuAD"), range(n_per_task), strict=False))
    for r in qa:
        text = f"Answer the question with a short span copied from the text.\n\nText: {r['context'][:3000]}\n\nQuestion: {r['question']}"  # noqa: E501 prompt kept in the short-suite format
        out.append(("qa", text))
    db = load_dataset("fancyzhx/dbpedia_14", split="train").shuffle(seed=seed)
    labels = db.features["label"].names
    head = f"Classify this item into one of: {', '.join(labels)}. Reply with the label only.\n\n"
    for r in db.select(range(n_per_task)):
        out.append(("classify", head + r["content"]))
    return out


def save_delta(deltas: list[DeltaProj | DeltaGate], path: Path) -> None:
    """Save the deltas, one tensor per MoE layer in router order."""
    mx.save_safetensors(str(path), {f"{i:03d}.delta": d.delta for i, d in enumerate(deltas)})


def load_delta(deltas: list[DeltaProj | DeltaGate], path: str | Path) -> int:
    """Load a save_delta checkpoint into deltas in router order; returns the number of layers loaded."""
    w = mx.load(str(path))
    if len(w) != len(deltas):
        raise ValueError(f"checkpoint has {len(w)} layers, the model has {len(deltas)} routers")
    for i, d in enumerate(deltas):
        d.delta = w[f"{i:03d}.delta"]
    return len(w)


def task_batches(items: list[tuple[str, str]], size: int) -> Iterator[tuple[str, list[tuple[int, str]]]]:
    """Batches of (index, prompt) sharing one task, so a batch shares one max_tokens; tasks in first-seen order."""
    by_task: dict[str, list[tuple[int, str]]] = {}
    for i, (task, prompt) in enumerate(items):
        by_task.setdefault(task, []).append((i, prompt))
    for task, rows in by_task.items():
        for j in range(0, len(rows), size):
            yield task, rows[j : j + size]


def cmd_teach(a: argparse.Namespace) -> None:
    """Teacher answers: the base model unmasked, greedy, batched, short-suite max tokens; resumable by index."""
    from mlx_lm import batch_generate, load

    import segroute as sr
    from short import MAX

    model, tok = load(MODELS[a.model_tag])
    items = training_items(a.n_per_task)
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{a.model_tag}_c3b_teacher.jsonl"
    done = {json.loads(line)["i"] for line in path.open()} if path.exists() else set()
    t0 = time.time()
    with open(path, "a") as f:
        for task, rows in task_batches(items, a.batch):
            rows = [(i, p) for i, p in rows if i not in done]
            if not rows:
                continue
            ids = [sr.chat(tok, p) for _, p in rows]
            res = batch_generate(model, tok, ids, max_tokens=MAX[task], return_token_ids=True)
            for (i, _), p, g in zip(rows, ids, res.token_ids, strict=True):
                f.write(json.dumps({"i": i, "task": task, "prompt": p, "answer": list(g)}) + "\n")
            f.flush()
            done |= {i for i, _ in rows}
            print(f"{len(done)}/{len(items)} {task} {time.time() - t0:.0f}s", flush=True)


def cmd_teach_cls(a: argparse.Namespace) -> None:
    """Teacher file of one H38 recipe: the base teacher's other tasks, new classify items, unmasked and greedy."""
    from mlx_lm import batch_generate, load

    import segroute as sr
    from short import MAX

    model, tok = load(MODELS[a.model_tag])
    prompts = recipe_prompts(a.recipe)
    rows = []
    for j in range(0, len(prompts), a.batch):
        ids = [sr.chat(tok, p) for p in prompts[j : j + a.batch]]
        res = batch_generate(model, tok, ids, max_tokens=MAX["classify"], return_token_ids=True)
        for n, (p, g) in enumerate(zip(ids, res.token_ids, strict=True)):
            rows.append({"i": 100000 + j + n, "task": "classify", "prompt": p, "answer": list(g)})
        print(f"{len(rows)}/{len(prompts)}", flush=True)
    base = [json.loads(line) for line in teacher_path(a.model_tag, "").open()]
    with teacher_path(a.model_tag, a.recipe).open("w") as f:
        for r in replace_classify(base, rows):
            f.write(json.dumps(r) + "\n")


def cmd_train(a: argparse.Namespace) -> None:
    """One pass over the teacher answers, gradient accumulation, AdamW with cosine decay; deltas saved every N steps."""
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    from mlx_lm import load

    import segroute as sr
    from short import MAX

    model, tok = load(MODELS[a.model_tag])
    ctl = sr.install(model)
    deltas = attach(model, ctl)
    prepare_training(model, deltas)
    recs = [json.loads(line) for line in teacher_path(a.model_tag, getattr(a, "recipe", "")).open()]
    if getattr(a, "stop", False):
        recs = with_stop(recs, tok.eos_token_id, MAX)
    recs = [r for r in recs if len(r["answer"]) >= 2]
    order = np.random.default_rng(getattr(a, "seed", 0)).permutation(len(recs))
    ends = getattr(a, "ends", None)
    run = CKPT / run_name(a)
    run.mkdir(parents=True, exist_ok=True)
    steps = len(order) // a.accum
    opt = optim.AdamW(learning_rate=optim.cosine_decay(a.lr, steps, a.lr * 0.1), weight_decay=0.0)
    vg = nn.value_and_grad(model, loss_fn)
    t0, acc, total = time.time(), None, 0.0
    if a.approx_la and a.chunked_la:
        raise SystemExit("--approx-la and --chunked-la are exclusive")
    if a.chunked_la:
        from gdn_chunked import chunked_linear_attention

        linear_attention = chunked_linear_attention()
    else:
        linear_attention = approx_linear_attention() if a.approx_la else nullcontext()
    with linear_attention, open(run / "train.jsonl", "a") as log:
        for n, idx in enumerate(order[: steps * a.accum]):
            r = recs[int(idx)]
            t_lp = teacher_logprobs(model, ctl, r["prompt"], r["answer"])
            loss, g = vg(model, ctl, r["prompt"], r["answer"], t_lp, a.k, a.s, a.mu, ends)
            acc = g if acc is None else tree_map(lambda x, y: x + y, acc, g)
            mx.eval(acc, loss)
            total += loss.item()
            if (n + 1) % a.accum:
                continue
            step = (n + 1) // a.accum
            opt.update(model, tree_map(lambda x: x / a.accum, acc))
            mx.eval(model.parameters(), opt.state)
            acc = None
            rec = {"step": step, "loss": total / a.accum, "s": round(time.time() - t0)}
            total = 0.0
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
            if step % a.every == 0 or step == steps:
                save_delta(deltas, run / f"delta_{step:04d}.safetensors")


def parser() -> argparse.ArgumentParser:
    """Command line: teach or train."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teach")
    t.add_argument("--model-tag", required=True, choices=sorted(MODELS))
    t.add_argument("--n-per-task", type=int, default=1000)
    t.add_argument("--batch", type=int, default=16)
    tc = sub.add_parser("teach-cls")
    tc.add_argument("--model-tag", required=True, choices=sorted(MODELS))
    tc.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    tc.add_argument("--batch", type=int, default=16)
    tr = sub.add_parser("train")
    tr.add_argument("--model-tag", required=True, choices=sorted(MODELS))
    tr.add_argument("--k", type=int, default=16)
    tr.add_argument("--s", type=int, default=64)
    tr.add_argument("--mu", type=float, default=0.01)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--accum", type=int, default=8)
    tr.add_argument("--every", type=int, default=100)
    tr.add_argument("--ends", type=int, help="prompt set from this many prompt tokens, half from each end")
    tr.add_argument("--recipe", default="", choices=["", *sorted(RECIPES)], help="teacher file of an H38 recipe")
    tr.add_argument("--seed", type=int, default=0, help="training order seed")
    tr.add_argument("--stop", action="store_true", help="add the end token to teacher answers that stopped (H35)")
    tr.add_argument("--approx-la", action="store_true", help="Qwen3.6: no gradient through the gated-delta recurrence")
    tr.add_argument(
        "--chunked-la", action="store_true", help="Qwen3.6: chunked, differentiable gated delta (gdn_chunked)"
    )
    return p


def main() -> None:
    a = parser().parse_args()
    {"teach": cmd_teach, "teach-cls": cmd_teach_cls, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
