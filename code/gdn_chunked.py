"""Chunked gated-delta rule for Qwen3.6 training (C3b): differentiable, memory per chunk instead of per token.

mlx_lm runs the gated-delta recurrence either on a Metal kernel (no vjp) or as a per-token ops loop whose backward keeps
one (Dv, Dk) state per token (57.7 GB for a 309-token item). This is the chunkwise form of the same recurrence (the
WY / UT form used by flash-linear-attention and transformers' torch_chunk_gated_delta_rule):

    S_t = g_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T,    y_t = S_t q_t

Inside a chunk of C tokens the intra-chunk part is a few (C, C) and (C, D) matrix products; one sequential step per
chunk carries the state. The unit lower-triangular inverse (I + N)^-1 is
computed by block inversion in log2(C) levels of matrix products, with no row loop.

Per call it matches a float64 reference within bfloat16 output rounding, like the kernel and the ops path. The model
is sensitive to rounding-level differences, so whole-model logits are not bit-identical to
the kernel.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def unit_lower_inverse(n: mx.array) -> mx.array:
    """(I + n)^-1 for strictly lower-triangular n of shape (..., C, C), C a power of two.

    Block inversion, bottom up: [[A, 0], [C, D]]^-1 = [[A^-1, 0], [-D^-1 C A^-1, D^-1]]. As stable as forward
    substitution; the power series (I - N)(I + N^2)(I + N^4)... is not (its terms reach 1e9 on real chunks).
    """
    c = n.shape[-1]
    if c & (c - 1):
        raise ValueError(f"chunk size must be a power of two, not {c}")
    lead = n.shape[:-2]
    inv = mx.ones((*lead, c, 1, 1), dtype=n.dtype)  # inverses of the 1x1 diagonal blocks
    s = 1
    while s < c:
        m = c // (2 * s)
        blocks = n.reshape(*lead, m, 2 * s, m, 2 * s)
        blocks = mx.moveaxis(mx.diagonal(blocks, axis1=-4, axis2=-2), -1, -3)  # (..., m, 2s, 2s) diagonal blocks
        a_inv, d_inv = inv[..., 0::2, :, :], inv[..., 1::2, :, :]
        low = -(d_inv @ blocks[..., s:, :s] @ a_inv)
        top = mx.concatenate([a_inv, mx.zeros_like(a_inv)], axis=-1)
        bottom = mx.concatenate([low, d_inv], axis=-1)
        inv = mx.concatenate([top, bottom], axis=-2)
        s *= 2
    return inv[..., 0, :, :]


def chunk_gated_delta(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    log_g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    chunk: int = 64,
) -> tuple[mx.array, mx.array]:
    """Gated-delta recurrence, chunkwise. Shapes as mlx_lm's gated_delta_ops, scalar gating only.

    q, k: (B, T, Hk, Dk), already normalized; v: (B, T, Hv, Dv); log_g, beta: (B, T, Hv); state: (B, Hv, Dv, Dk).
    Returns y (B, T, Hv, Dv) in q's dtype and the final state in float32.
    """
    b, t, hk, dk = q.shape
    hv, dv = v.shape[-2:]
    out_dtype = q.dtype
    if hv // hk > 1:
        q = mx.repeat(q, hv // hk, -2)
        k = mx.repeat(k, hv // hk, -2)
    pad = (-t) % chunk
    n_chunks = (t + pad) // chunk

    def heads_first(x: mx.array) -> mx.array:  # (B, T, H, ...) -> (B, H, N, C, ...), padded with zeros
        x = x.astype(mx.float32)
        if pad:
            x = mx.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        x = mx.moveaxis(x, 2, 1)
        return x.reshape(b, hv, n_chunks, chunk, *x.shape[3:])

    # Padding: k = v = beta = 0 and log_g = 0 leave the state unchanged, and their outputs are cut off.
    q, k, v, lg, bt = (heads_first(x) for x in (q, k, v, log_g, beta))
    g_cum = mx.cumsum(lg, axis=-1)  # (B, H, N, C)
    rows = mx.arange(chunk)
    lower = rows[:, None] >= rows[None, :]
    diff = g_cum[..., :, None] - g_cum[..., None, :]
    decay = mx.where(lower, mx.exp(mx.where(lower, diff, 0.0)), 0.0)  # exp(G_i - G_j) for i >= j

    k_beta = k * bt[..., None]
    n = mx.where(rows[:, None] > rows[None, :], (k_beta @ mx.swapaxes(k, -1, -2)) * decay, 0.0)
    tri = unit_lower_inverse(n)  # (B, H, N, C, C)
    u = tri @ (v * bt[..., None])  # (B, H, N, C, Dv)
    w = tri @ (k_beta * mx.exp(g_cum)[..., None])  # (B, H, N, C, Dk)
    attn = (q @ mx.swapaxes(k, -1, -2)) * decay  # (B, H, N, C, C), lower triangle incl. diagonal

    s = mx.zeros((b, hv, dk, dv), dtype=mx.float32) if state is None else mx.swapaxes(state.astype(mx.float32), -1, -2)
    ys = []
    for i in range(n_chunks):
        g_i = g_cum[:, :, i]
        v_new = u[:, :, i] - w[:, :, i] @ s
        ys.append((q[:, :, i] * mx.exp(g_i)[..., None]) @ s + attn[:, :, i] @ v_new)
        g_last = g_i[..., -1:]
        s = s * mx.exp(g_last)[..., None] + mx.swapaxes(k[:, :, i] * mx.exp(g_last - g_i)[..., None], -1, -2) @ v_new
    y = mx.concatenate(ys, axis=2).reshape(b, hv, n_chunks * chunk, dv)[:, :, :t]
    return mx.moveaxis(y, 1, 2).astype(out_dtype), mx.swapaxes(s, -1, -2)


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,  # noqa: N803 (mlx_lm's name)
    dt_bias: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
    *,
    use_kernel: bool = True,
    lower_bound: float | None = None,
    allow_neg_eigval: bool = False,
) -> tuple[mx.array, mx.array]:
    """Drop-in for mlx_lm.models.gated_delta.gated_delta_update (no mask, no lower bound), chunked."""
    del use_kernel
    if mask is not None or lower_bound is not None:
        raise NotImplementedError("chunked gated delta: no padding mask and no lower-bound gating")
    beta = mx.sigmoid(b)
    if allow_neg_eigval:
        beta = beta * 2.0
    log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)  # log of mlx_lm's compute_g, without exp/log
    return chunk_gated_delta(q, k, v, log_g, beta, state)


@contextmanager
def chunked_linear_attention() -> Iterator[None]:
    """Route Qwen3.6 (mlx_lm qwen3_5) gated-delta layers through the chunked form for the duration."""
    import mlx_lm.models.qwen3_5 as q35

    orig: Any = q35.gated_delta_update
    q35.gated_delta_update = gated_delta_update
    try:
        yield
    finally:
        q35.gated_delta_update = orig
