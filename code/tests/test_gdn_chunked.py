"""gdn_chunked.py: chunked gated-delta rule, differentiable, against mlx_lm's per-token ops reference."""

import mlx.core as mx
import mlx_lm.models.gated_delta as gd
import numpy as np
import pytest

import gdn_chunked as gc


def inputs(t=37, hk=2, hv=4, dk=32, dv=16, seed=0, dtype=mx.float32):
    mx.random.seed(seed)
    q = mx.random.normal((1, t, hk, dk))
    k = mx.random.normal((1, t, hk, dk))
    q, k = gd.normalize_qk(q, k, inv_scale=dk**-0.5, eps=1e-6)
    v = mx.random.normal((1, t, hv, dv))
    log_g = -mx.random.uniform(0.0, 0.5, (1, t, hv))
    beta = mx.sigmoid(mx.random.normal((1, t, hv)))
    state = mx.random.normal((1, hv, dv, dk)) * 0.1
    return q.astype(dtype), k.astype(dtype), v.astype(dtype), log_g, beta.astype(dtype), state


def test_unit_lower_inverse_matches_a_solve():
    mx.random.seed(1)
    n = mx.tril(mx.random.normal((3, 8, 8)) * 0.3, k=-1)
    inv = gc.unit_lower_inverse(n)
    eye = mx.eye(8)
    assert mx.allclose(inv @ (eye + n), mx.broadcast_to(eye, (3, 8, 8)), atol=1e-5)


def test_unit_lower_inverse_is_stable_when_entries_are_near_one():
    """Real Qwen3.6 chunks: unit-norm, nearly parallel keys and little decay give strictly-lower entries near 1."""
    mx.random.seed(2)
    base = mx.random.normal((1, 64))
    k = base + 0.05 * mx.random.normal((64, 64))
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    n = mx.tril(0.9 * (k @ k.T), k=-1)
    inv = gc.unit_lower_inverse(n)
    exact = np.linalg.inv(np.eye(64) + np.array(n, dtype=np.float64))
    assert np.abs(np.array(inv) - exact).max() <= 1e-3 * max(1.0, np.abs(exact).max())


@pytest.mark.parametrize(("t", "chunk"), [(37, 16), (64, 16), (5, 64), (130, 64)])
def test_chunked_matches_the_ops_reference(t, chunk):
    q, k, v, log_g, beta, state = inputs(t=t)
    y_ref, s_ref = gd.gated_delta_ops(q, k, v, mx.exp(log_g), beta, state)
    y, s = gc.chunk_gated_delta(q, k, v, log_g, beta, state, chunk=chunk)
    assert y.shape == y_ref.shape and s.shape == s_ref.shape
    assert mx.allclose(y, y_ref, atol=1e-4, rtol=1e-4)
    assert mx.allclose(s, s_ref, atol=1e-4, rtol=1e-4)


def test_chunked_without_a_state_starts_from_zero():
    q, k, v, log_g, beta, _ = inputs(t=20)
    y_ref, _ = gd.gated_delta_ops(q, k, v, mx.exp(log_g), beta, None)
    y, _ = gc.chunk_gated_delta(q, k, v, log_g, beta, None, chunk=8)
    assert mx.allclose(y, y_ref, atol=1e-4, rtol=1e-4)


def test_chunked_gradients_match_the_ops_reference():
    q, k, v, log_g, beta, state = inputs(t=24)
    w = mx.random.normal((1, 24, 4, 16))

    def loss(fn, q, k, v, log_g, beta):
        return (fn(q, k, v, log_g, beta) * w).sum()

    def ops(q, k, v, log_g, beta):
        return gd.gated_delta_ops(q, k, v, mx.exp(log_g), beta, state)[0]

    def chunked(q, k, v, log_g, beta):
        return gc.chunk_gated_delta(q, k, v, log_g, beta, state, chunk=8)[0]

    args = (q, k, v, log_g, beta)
    g_ref = mx.grad(lambda *a: loss(ops, *a), argnums=(0, 1, 2, 3, 4))(*args)
    g = mx.grad(lambda *a: loss(chunked, *a), argnums=(0, 1, 2, 3, 4))(*args)
    for a, b in zip(g, g_ref, strict=True):
        assert mx.allclose(a, b, atol=1e-3, rtol=1e-3)


def test_bfloat16_inputs_return_bfloat16_outputs_and_a_float32_state():
    q, k, v, log_g, beta, state = inputs(t=20, dtype=mx.bfloat16)
    y, s = gc.chunk_gated_delta(q, k, v, log_g, beta, state, chunk=8)
    assert y.dtype == mx.bfloat16 and s.dtype == mx.float32


def test_update_matches_mlx_lm_gated_delta_update():
    q, k, v, _, _, state = inputs(t=30)
    mx.random.seed(3)
    a, b = mx.random.normal((1, 30, 4)), mx.random.normal((1, 30, 4))
    a_log, dt_bias = mx.random.normal((4,)) * 0.3, mx.random.normal((4,)) * 0.3
    y_ref, s_ref = gd.gated_delta_update(q, k, v, a, b, a_log, dt_bias, state, None, use_kernel=False)
    y, s = gc.gated_delta_update(q, k, v, a, b, a_log, dt_bias, state, None, use_kernel=True)
    assert mx.allclose(y, y_ref, atol=1e-4, rtol=1e-4) and mx.allclose(s, s_ref, atol=1e-4, rtol=1e-4)


def test_update_refuses_what_it_does_not_implement():
    q, k, v, _, _, state = inputs(t=4)
    a = b = mx.zeros((1, 4, 4))
    with pytest.raises(NotImplementedError):
        gc.gated_delta_update(q, k, v, a, b, mx.zeros((4,)), mx.zeros((4,)), state, mx.ones((1, 4), dtype=mx.bool_))
    with pytest.raises(NotImplementedError):
        gc.gated_delta_update(q, k, v, a, b, mx.zeros((4,)), mx.zeros((4,)), state, None, lower_bound=-1.0)


def test_context_manager_patches_and_restores_qwen3_5():
    import mlx_lm.models.qwen3_5 as q35

    orig = q35.gated_delta_update
    with gc.chunked_linear_attention():
        assert q35.gated_delta_update is gc.gated_delta_update
    assert q35.gated_delta_update is orig


def test_chunked_memory_grows_by_chunk_not_by_token():
    """Gradient of a 256-token pass: the chunked path stays well under the per-token ops path."""
    q, k, v, log_g, beta, state = inputs(t=256, hk=2, hv=8, dk=128, dv=128)

    def peak(fn):
        mx.clear_cache()
        mx.reset_peak_memory()
        g = mx.grad(lambda v: fn(q, k, v, log_g, beta).sum())(v)
        mx.eval(g)
        return mx.get_peak_memory()

    ops = peak(lambda q, k, v, lg, b: gd.gated_delta_ops(q, k, v, mx.exp(lg), b, state)[0])
    chunked = peak(lambda q, k, v, lg, b: gc.chunk_gated_delta(q, k, v, lg, b, state, chunk=64)[0])
    assert chunked < ops / 4, (chunked, ops)
    assert np.isfinite(chunked)


def _ref64(q, k, v, log_g, beta):
    q, k, v, lg, beta = (np.array(x.astype(mx.float32), dtype=np.float64) for x in (q, k, v, log_g, beta))
    hv = v.shape[-2]
    q, k = np.repeat(q, hv // q.shape[-2], -2), np.repeat(k, hv // k.shape[-2], -2)
    s = np.zeros((q.shape[0], hv, v.shape[-1], q.shape[-1]))
    ys = []
    for t in range(q.shape[1]):
        s = s * np.exp(lg[:, t])[..., None, None]
        d = (v[:, t] - (s * k[:, t][:, :, None, :]).sum(-1)) * beta[:, t][..., None]
        s = s + k[:, t][:, :, None, :] * d[..., None]
        ys.append((s * q[:, t][:, :, None, :]).sum(-1))
    return np.stack(ys, 1)


@pytest.mark.slow
def test_chunked_is_as_close_to_float64_as_the_kernel_on_qwen36_calls():
    import json

    import mlx_lm.models.qwen3_5 as q35
    from mlx_lm import load

    import c3b
    from models import MODELS

    calls = []
    orig = q35.gated_delta_update

    def record(q, k, v, a, b, a_log, dt_bias, state=None, mask=None, **kw):
        calls.append((q, k, v, a, b, a_log, dt_bias))
        return orig(q, k, v, a, b, a_log, dt_bias, state, mask, **kw)

    r = next(json.loads(x) for x in (c3b.DATA / "qwen36_c3b_teacher.jsonl").open() if json.loads(x)["i"] == 2036)
    model, _ = load(MODELS["qwen36"])
    q35.gated_delta_update = record
    try:
        mx.eval(model(mx.array(r["prompt"] + r["answer"])[None]))
    finally:
        q35.gated_delta_update = orig
    for q, k, v, a, b, a_log, dt_bias in calls[:: len(calls) // 5]:
        log_g = -mx.exp(a_log.astype(mx.float32)) * mx.logaddexp(a + dt_bias, 0.0)
        ref = _ref64(q, k, v, log_g, mx.sigmoid(b))
        y_kernel = gd.gated_delta_update(q, k, v, a, b, a_log, dt_bias)[0]
        y_chunk = gc.gated_delta_update(q, k, v, a, b, a_log, dt_bias)[0]
        err_kernel = np.abs(np.array(y_kernel.astype(mx.float32)) - ref).max()
        err_chunk = np.abs(np.array(y_chunk.astype(mx.float32)) - ref).max()
        assert err_chunk <= 2 * err_kernel, (err_chunk, err_kernel)
