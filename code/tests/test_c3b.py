"""c3b.py: masked distillation of a router delta that is active only under the mask."""

import gc

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm import load

import c3b
import segroute
from c3b import groups
from models import MODELS


class Ctl:
    mode = "off"


def proj(d=6, e=4, seed=0):
    mx.random.seed(seed)
    lin = nn.Linear(d, e, bias=False)
    lin.freeze()
    return lin


def test_delta_proj_is_the_router_when_off_even_with_a_delta():
    ctl, lin = Ctl(), proj()
    dp = c3b.DeltaProj(lin, ctl)
    dp.delta = mx.ones_like(dp.delta)
    x = mx.random.normal((1, 3, 6)).astype(mx.bfloat16)
    assert mx.array_equal(dp(x), lin(x))


@pytest.mark.parametrize("mode", ["mask", "train"])
def test_delta_proj_adds_the_delta_when_active(mode):
    ctl, lin = Ctl(), proj()
    ctl.mode = mode
    dp = c3b.DeltaProj(lin, ctl)
    x = mx.random.normal((1, 3, 6))
    assert mx.array_equal(dp(x), lin(x))  # zero delta at start
    dp.delta = mx.ones_like(dp.delta)
    assert mx.allclose(dp(x), lin(x) + x.sum(axis=-1, keepdims=True))


def test_only_the_delta_is_trainable():
    dp = c3b.DeltaProj(proj(), Ctl())
    names = [k for k, _ in tree_flatten(dp.trainable_parameters())]
    assert names == ["delta"]


class FakeGate(nn.Module):
    """The attributes of a Nemotron-H MoEGate that adapters.GateTap reads."""

    def __init__(self, d=6, e=8, seed=0):
        super().__init__()
        mx.random.seed(seed)
        self.weight = mx.random.normal((e, d))
        self.e_score_correction_bias = mx.random.normal((e,)) * 0.1
        self.top_k, self.n_group, self.norm_topk_prob, self.routed_scaling_factor = 2, 1, True, 2.5
        self.n_routed_experts = e


def ctl_for(e=8):
    c = segroute.Controller(n_layers=1, n_experts=e)
    c.k, c.s = 4, 2
    return c


@pytest.mark.parametrize("mode", ["off", "log"])
def test_delta_gate_is_the_gate_tap_when_inactive_even_with_a_delta(mode):
    ctl, gate = ctl_for(), FakeGate()
    ctl.mode = mode
    ref = c3b.GateTap(gate, 0, ctl)
    dg = c3b.DeltaGate(gate, 0, ctl)
    dg.delta = mx.ones_like(dg.delta)
    x = mx.random.normal((1, 3, 6))
    (i0, w0), (i1, w1) = ref(x), dg(x)
    assert mx.array_equal(i0, i1) and mx.array_equal(w0, w1)


def test_delta_gate_with_zero_delta_matches_the_gate_tap_under_the_mask():
    ctl_a, ctl_b, gate = ctl_for(), ctl_for(), FakeGate()
    ctl_a.mode = ctl_b.mode = "mask"
    ref, dg = c3b.GateTap(gate, 0, ctl_a), c3b.DeltaGate(gate, 0, ctl_b)
    for x in (mx.random.normal((1, 3, 6)), mx.random.normal((1, 1, 6)), mx.random.normal((1, 1, 6))):
        (i0, w0), (i1, w1) = ref(x), dg(x)
        assert mx.array_equal(i0, i1) and mx.allclose(w0, w1)


def test_delta_gate_train_mode_selects_inside_the_segment_set_and_trains_the_delta_only():
    ctl, gate = ctl_for(), FakeGate()
    dg = c3b.DeltaGate(gate, 0, ctl)
    assert [k for k, _ in tree_flatten(dg.trainable_parameters())] == ["delta"]
    ctl.mode, ctl.c3b = "train", {"n_prompt": 2, "k": 2, "s": 2, "drift": []}
    x = mx.random.normal((1, 7, 6))
    inds, _ = dg(x)
    scores = mx.sigmoid(x[0] @ gate.weight.T)
    r = scores + gate.e_score_correction_bias
    free = mx.argpartition(-r, kth=1, axis=-1)[:, :2]
    w = mx.put_along_axis(mx.zeros_like(scores), free, mx.take_along_axis(scores, free, -1), axis=-1)
    w = w / w.sum(axis=-1, keepdims=True)
    allowed = np.array(c3b.segment_mask(w, n_prompt=2, k=2, s=2)) == 0.0
    assert all(allowed[t, int(j)] for t, row in enumerate(np.array(inds[0])) for j in row)
    assert len(ctl.c3b["drift"]) == 1 and float(ctl.c3b["drift"][0]) == pytest.approx(0.0, abs=1e-6)


def test_segment_mask_matches_the_groups_of_the_controller():
    # 2 prompt tokens, S = 2, 7 tokens: prompt set serves 0-3, then sets from 2-3 and 4-5
    probs = np.zeros((7, 4), dtype=np.float32)
    probs[0:2, 0] = probs[2:4, 1] = probs[4:6, 2] = 1.0
    mask = np.array(c3b.segment_mask(mx.array(probs), n_prompt=2, k=1, s=2))
    allowed = [int(np.argmax(row)) for row in mask]
    assert allowed == [0, 0, 0, 0, 1, 1, 2]
    assert ((mask == 0.0) | (mask <= -1e8)).all()
    assert [tuple(g) for g in groups(7, 2, 2)][0] == ((0, 4), (0, 2))


def test_training_prompts_avoid_the_short_suite_and_squad():
    items = c3b.training_items(n_per_task=5)
    assert {t for t, _ in items} == {"summarize", "qa", "classify", "rewrite"}
    assert len(items) == 20
    import short

    suite = {p for _, _, p, _ in short.items(300)}
    assert not suite & {p for _, p in items}
    assert all("SQuAD" not in p for _, p in items)


def test_mrqa_rows_from_squad_are_dropped():
    rows = [{"subset": "SQuAD", "context": "a", "question": "q"}, {"subset": "NewsQA", "context": "b", "question": "r"}]
    assert [r["subset"] for r in c3b.keep_mrqa(rows)] == ["NewsQA"]


@pytest.fixture(scope="module", params=["gemma", "qwen36", "nemotron"])
def gemma(request):
    model, tok, config = load(MODELS[request.param], return_config=True)
    ctl = segroute.install(model)
    deltas = c3b.attach(model, ctl)
    ids = segroute.chat(tok, "Summarize in one sentence: the cat sat on the mat while the dog slept in the sun.")
    ctl.mode = "off"
    gen = segroute.generate(model, tok, ids, 24, config)
    yield model, ctl, deltas, ids, gen
    del model, tok
    mx.clear_cache()
    gc.collect()


@pytest.mark.slow
def test_unmasked_logits_are_bit_identical_with_a_trained_delta(gemma):
    model, ctl, deltas, ids, gen = gemma
    seq = mx.array(ids + gen)[None]
    ctl.reset()
    ctl.mode = "off"
    before = model(seq)
    for d in deltas:
        d.delta = mx.random.normal(d.delta.shape) * 0.1
    after = model(seq)
    for d in deltas:
        d.delta = mx.zeros_like(d.delta)
    assert mx.array_equal(before, after)


@pytest.mark.slow
def test_gradient_reaches_the_delta_only(gemma):
    model, ctl, deltas, ids, gen = gemma
    c3b.prepare_training(model, deltas)  # train mode: Qwen3.6 gated-delta layers use the differentiable ops path
    try:
        t_lp = c3b.teacher_logprobs(model, ctl, ids, gen)
        loss, grads = nn.value_and_grad(model, c3b.loss_fn)(model, ctl, ids, gen, t_lp, 16, 8, 0.01)
    finally:
        model.eval()
    flat = tree_flatten(grads)
    assert flat and all(k.endswith("delta") for k, _ in flat)
    assert any(float(mx.abs(g).sum()) > 0 for _, g in flat)
    assert np.isfinite(loss.item())


def test_delta_checkpoint_round_trips(tmp_path):
    ctl = Ctl()
    a = [c3b.DeltaProj(proj(seed=i), ctl) for i in range(3)]
    for i, d in enumerate(a):
        d.delta = mx.full(d.delta.shape, float(i + 1))
    path = tmp_path / "delta.safetensors"
    c3b.save_delta(a, path)
    b = [c3b.DeltaProj(proj(seed=i), ctl) for i in range(3)]
    assert c3b.load_delta(b, path) == 3
    assert all(mx.array_equal(x.delta, y.delta) for x, y in zip(a, b, strict=True))


def test_load_delta_rejects_a_layer_count_mismatch(tmp_path):
    ctl = Ctl()
    path = tmp_path / "delta.safetensors"
    c3b.save_delta([c3b.DeltaProj(proj(), ctl)], path)
    with pytest.raises(ValueError):
        c3b.load_delta([c3b.DeltaProj(proj(), ctl), c3b.DeltaProj(proj(), ctl)], path)


def test_teacher_batches_group_items_by_task():
    items = [("qa", "a"), ("classify", "b"), ("qa", "c"), ("qa", "d")]
    batches = list(c3b.task_batches(items, size=2))
    assert batches == [("qa", [(0, "a"), (2, "c")]), ("qa", [(3, "d")]), ("classify", [(1, "b")])]


def test_short_run_accepts_a_delta_checkpoint():
    import short

    a = short.parser().parse_args(["run", "--model-tag", "gemma", "--frac", "0.125", "--delta", "x.safetensors"])
    assert a.delta == "x.safetensors"


def _is_qwen36(model):
    return type(model).__module__.endswith("qwen3_5_moe") or "qwen3_5" in type(model).__module__


@pytest.mark.slow
def test_approx_linear_attention_keeps_the_forward_and_trains_the_delta_only(gemma):
    model, ctl, deltas, ids, gen = gemma
    if not _is_qwen36(model):
        pytest.skip("gated-delta layers are Qwen3.6 only")
    ref = c3b.masked_logprobs(model, ctl, ids, gen, 16, 8)  # eval mode: the Metal kernel
    c3b.prepare_training(model, deltas)
    try:
        with c3b.approx_linear_attention():
            lp = c3b.masked_logprobs(model, ctl, ids, gen, 16, 8)
            t_lp = c3b.teacher_logprobs(model, ctl, ids, gen)
            loss, grads = nn.value_and_grad(model, c3b.loss_fn)(model, ctl, ids, gen, t_lp, 16, 8, 0.01)
    finally:
        model.eval()
    assert mx.array_equal(lp, ref)
    flat = tree_flatten(grads)
    assert flat and all(k.endswith("delta") for k, _ in flat)
    assert sum(float(mx.abs(g).sum()) > 0 for _, g in flat) == len(flat)
    assert np.isfinite(loss.item())


def test_approx_linear_attention_restores_the_recurrence():
    import mlx_lm.models.qwen3_5 as q35

    orig = q35.gated_delta_update
    with c3b.approx_linear_attention():
        assert q35.gated_delta_update is not orig
    assert q35.gated_delta_update is orig


def test_train_accepts_the_approx_flag():
    a = c3b.parser().parse_args(["train", "--model-tag", "qwen36", "--approx-la"])
    assert a.approx_la


def test_train_accepts_the_chunked_flag():
    a = c3b.parser().parse_args(["train", "--model-tag", "qwen36", "--chunked-la"])
    assert a.chunked_la and not a.approx_la


@pytest.mark.slow
def test_chunked_linear_attention_trains_every_delta_on_qwen36(gemma):
    import gdn_chunked

    model, ctl, deltas, ids, gen = gemma
    if not _is_qwen36(model):
        pytest.skip("gated-delta layers are Qwen3.6 only")
    c3b.prepare_training(model, deltas)
    try:
        with gdn_chunked.chunked_linear_attention():
            t_lp = c3b.teacher_logprobs(model, ctl, ids, gen)
            loss, grads = nn.value_and_grad(model, c3b.loss_fn)(model, ctl, ids, gen, t_lp, 16, 8, 0.01)
    finally:
        model.eval()
    flat = tree_flatten(grads)
    assert flat and all(k.endswith("delta") for k, _ in flat)
    assert sum(float(mx.abs(g).sum()) > 0 for _, g in flat) == len(flat)
    assert np.isfinite(loss.item())


def test_segment_mask_with_ends_chooses_the_prompt_set_from_both_ends_only():
    # tokens 1 and 2 are the first and last prompt tokens; the middle token favors another expert by a wide margin
    probs = mx.array([[1.0, 0, 0, 0], [0, 0, 0, 9.0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
    plain = np.array(c3b.segment_mask(probs, n_prompt=3, k=1, s=2))
    ends = np.array(c3b.segment_mask(probs, n_prompt=3, k=2, s=2, ends=2))
    assert plain[0].tolist()[3] == 0.0
    assert ends[0].tolist()[3] < 0.0
    assert ends[0].tolist()[:2] == [0.0, 0.0]


def test_segment_mask_with_ends_changes_the_prompt_set_only():
    mx.random.seed(0)
    probs = mx.random.uniform(shape=(12, 8))
    plain = np.array(c3b.segment_mask(probs, n_prompt=6, k=2, s=3))
    ends = np.array(c3b.segment_mask(probs, n_prompt=6, k=2, s=3, ends=4))
    assert (plain[9:] == ends[9:]).all()


def test_segment_mask_with_ends_longer_than_the_prompt_uses_the_whole_prompt():
    mx.random.seed(1)
    probs = mx.random.uniform(shape=(8, 6))
    plain = np.array(c3b.segment_mask(probs, n_prompt=5, k=2, s=3))
    assert (plain == np.array(c3b.segment_mask(probs, n_prompt=5, k=2, s=3, ends=64))).all()


def test_training_and_serving_choose_the_same_prompt_set_under_the_ends_rule():
    import selrule as sl

    mx.random.seed(2)
    probs = mx.random.uniform(shape=(40, 8))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    c = sl.TailController(1, 8, lambda: 8, "ends")
    c.mode, c.k, c.s = "mask", 3, 32
    c.reset()
    served = np.array(c.observe(0, probs)).reshape(-1) == 0.0
    trained = np.array(c3b.segment_mask(probs, n_prompt=40, k=3, s=32, ends=8))[0] == 0.0
    assert served.tolist() == trained.tolist()


def test_train_parser_takes_the_ends_rule():
    a = c3b.parser().parse_args(["train", "--model-tag", "gemma", "--ends", "64"])
    assert a.ends == 64
    assert c3b.parser().parse_args(["train", "--model-tag", "gemma"]).ends is None


def test_with_stop_appends_the_end_token_to_answers_that_stopped():
    recs = [{"task": "qa", "answer": [5, 6]}, {"task": "classify", "answer": [7]}]
    out = c3b.with_stop(recs, eos=99, limits={"qa": 32, "classify": 8})
    assert [r["answer"] for r in out] == [[5, 6, 99], [7, 99]]


def test_with_stop_leaves_answers_cut_at_the_token_limit():
    out = c3b.with_stop([{"task": "qa", "answer": [1, 2, 3]}], eos=99, limits={"qa": 3})
    assert out[0]["answer"] == [1, 2, 3]


def test_with_stop_does_not_change_its_input():
    recs = [{"task": "qa", "answer": [5]}]
    c3b.with_stop(recs, eos=99, limits={"qa": 32})
    assert recs[0]["answer"] == [5]


def test_train_parser_takes_the_stop_flag():
    assert c3b.parser().parse_args(["train", "--model-tag", "qwen36", "--stop"]).stop
    assert not c3b.parser().parse_args(["train", "--model-tag", "qwen36"]).stop


def test_run_name_keeps_stop_runs_apart_from_h32_runs():
    p = c3b.parser()
    base = ["train", "--model-tag", "qwen36", "--k", "16", "--s", "32", "--ends", "64", "--chunked-la"]
    assert c3b.run_name(p.parse_args(base)) == "qwen36_c3b_k16_s32_mu0.01_ends64_chunked"
    assert c3b.run_name(p.parse_args([*base, "--stop"])) == "qwen36_c3b_k16_s32_mu0.01_ends64_stop_chunked"


def test_mix_prompts_takes_the_counts_and_shuffles_each_label_list():
    sources = [(["a", "b", "c", "d"], [f"x{i}" for i in range(50)]), (["e", "f", "g"], [f"y{i}" for i in range(50)])]
    out = c3b.mix_prompts(sources, [30, 20], seed=0)
    assert len(out) == 50
    assert sum(p.endswith(tuple(f"\n\nx{i}" for i in range(50))) for p in out) == 30
    heads = {p.split("\n\n")[0] for p in out if "a" in p.split(":")[1]}
    assert len(heads) > 1
    assert all(p.startswith("Classify this item into one of: ") and "Reply with the label only." in p for p in out)


def test_mix_prompts_is_deterministic():
    sources = [(["a", "b", "c"], ["t1", "t2", "t3"])]
    assert c3b.mix_prompts(sources, [3], seed=1) == c3b.mix_prompts(sources, [3], seed=1)


def test_recipe_counts_sum_to_one_thousand_and_known_alone_holds_ag_news():
    assert sum(c3b.RECIPES["known"].values()) == 1000
    assert sum(c3b.RECIPES["zero"].values()) == 1000
    assert "ag_news" in c3b.RECIPES["known"] and "ag_news" not in c3b.RECIPES["zero"]
    assert "emotion" not in c3b.RECIPES["known"] and "emotion" not in c3b.RECIPES["zero"]


def test_replace_classify_keeps_the_other_tasks():
    rows = [{"i": 1, "task": "qa"}, {"i": 2, "task": "classify"}]
    out = c3b.replace_classify(rows, [{"i": 100000, "task": "classify"}])
    assert [r["i"] for r in out] == [1, 100000]


def test_run_name_separates_teacher_recipes_and_seeds():
    p = c3b.parser()
    base = ["train", "--model-tag", "qwen36", "--k", "12", "--s", "32", "--ends", "64", "--chunked-la", "--stop"]
    assert c3b.run_name(p.parse_args(base)) == "qwen36_c3b_k12_s32_mu0.01_ends64_stop_chunked"
    assert c3b.run_name(p.parse_args([*base, "--seed", "1"])) == "qwen36_c3b_k12_s32_mu0.01_ends64_stop_seed1_chunked"
    assert (
        c3b.run_name(p.parse_args([*base, "--recipe", "known"]))
        == "qwen36_c3b_k12_s32_mu0.01_ends64_stop_known_chunked"
    )


def test_teacher_path_follows_the_recipe():
    assert c3b.teacher_path("gemma", "").name == "gemma_c3b_teacher.jsonl"
    assert c3b.teacher_path("gemma", "zero").name == "gemma_c3b_teacher_zero.jsonl"


def test_dbrand_recipe_is_the_dbpedia_only_items_with_the_mixed_arms_random_label_order():
    assert c3b.RECIPES["dbrand"] == {"dbpedia": 1000}
    assert "ag_news" not in c3b.RECIPES["dbrand"]


def test_dbrand_run_name_and_teacher_file_carry_the_recipe():
    p = c3b.parser()
    base = ["train", "--model-tag", "qwen36", "--k", "12", "--s", "32", "--ends", "64", "--stop", "--chunked-la"]
    assert (
        c3b.run_name(p.parse_args([*base, "--recipe", "dbrand", "--seed", "3"]))
        == "qwen36_c3b_k12_s32_mu0.01_ends64_stop_dbrand_seed3_chunked"
    )
    assert c3b.teacher_path("qwen36", "dbrand").name == "qwen36_c3b_teacher_dbrand.jsonl"
