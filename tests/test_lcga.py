"""
Design doc A, sec. 9: LCGA tests (pytest, CPU, tiny config).

    python -m pytest tests -q
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import loop_vit_reference as ref
from loop_vit import LoopViT, LoopViTConfig, graph_delta

TINY = dict(image_size=32, patch_size=8, dim=64, num_heads=4, core_depth=2,
            loop_steps=3, num_classes=5)
CARRY = ["logprob", "qk", "realformer"]


def make(**kw) -> LoopViT:
    torch.manual_seed(0)
    return LoopViT(LoopViTConfig(**{**TINY, **kw})).eval()


def set_gamma(model: LoopViT, value=None, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for blk in model.core:
            gm = blk.attn.gamma
            gm.copy_(torch.full_like(gm, value) if value is not None
                     else torch.randn(gm.shape, generator=g) * 0.7)


def x_batch(n=6, seed=0):
    return torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def qkv_of(attn, x):
    return attn._qkv(x)


# --------------------------------------------------------------------------- #
def test_t1_baseline_equivalence():
    torch.manual_seed(0)
    old = ref.LoopViT(ref.LoopViTConfig(**TINY)).eval()
    new = LoopViT(LoopViTConfig(**TINY, attn_carry="none")).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    x = x_batch()
    with torch.no_grad():
        assert torch.equal(old(x), new(x))
        for a, b in zip(old.logits_per_step(x, 5), new.logits_per_step(x, 5)):
            assert torch.equal(a, b)
        do, dn = old.dynamic_forward(x, tau=0.5), new.dynamic_forward(x, tau=0.5)
        assert torch.equal(do["exit_steps"], dn["exit_steps"])
        assert torch.allclose(do["logits"], dn["logits"], atol=1e-6)


@pytest.mark.parametrize("mode", CARRY)
def test_t2_gamma_zero_equivalence(mode):
    torch.manual_seed(0)
    old = ref.LoopViT(ref.LoopViTConfig(**TINY)).eval()
    new = LoopViT(LoopViTConfig(**TINY, attn_carry=mode, carry_gamma_init=0.0)).eval()
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and all(k.endswith("attn.gamma") for k in missing)
    x = x_batch()
    with torch.no_grad():
        assert torch.allclose(old(x), new(x), atol=1e-5)
        assert torch.allclose(old(x, 6), new(x, 6), atol=1e-5)


def test_t3_prior_is_used():
    m = make(attn_carry="logprob")
    set_gamma(m, 1.0)
    x = x_batch()
    with torch.no_grad():
        h0 = m.embed(x)
        h1, st1, _ = m.core_step(h0, 0)
        blk = m.core[0]
        inp = blk.norm1(h1 + m._step_vector(1))
        q, k, v = qkv_of(blk.attn, inp)
        L = (q @ k.transpose(-2, -1)) * blk.attn.scale
        expected = (L + st1[0].clamp_min(-10.0)).log_softmax(-1)
        a = blk.attn(inp, state=st1[0], t=1)
        assert torch.allclose(a.state, expected, atol=1e-5)
        out = (expected.exp() @ v).transpose(1, 2).reshape(x.shape[0], -1, TINY["dim"])
        assert torch.allclose(a.out, blk.attn.proj(out), atol=1e-5)
        # the same thing through core_step
        _, st2, _ = m.core_step(h1, 1, states=st1)
        assert torch.allclose(st2[0], expected, atol=1e-5)
        # and the prior actually changes the graph
        assert not torch.allclose(expected, L.log_softmax(-1), atol=1e-3)


def test_t4_qk_matches_dense():
    m = make(attn_carry="qk")
    set_gamma(m)
    attn = m.core[0].attn
    g = torch.Generator().manual_seed(3)
    x = torch.randn(4, 17, TINY["dim"], generator=g)
    x_prev = torch.randn(4, 17, TINY["dim"], generator=g)
    with torch.no_grad():
        prev = attn(x_prev, t=0)
        a = attn(x, state=prev.state, t=1, need_cls=True)
        q, k, v = qkv_of(attn, x)
        qp, kp = prev.state
        gm = attn._gamma(1)
        L = (q @ k.transpose(-2, -1) + gm * (qp @ kp.transpose(-2, -1))) * attn.scale  # eq. A3
        P = L.softmax(-1)
        out = attn.proj((P @ v).transpose(1, 2).reshape(4, 17, -1))
        assert torch.allclose(a.out, out, atol=1e-5)
        assert torch.allclose(a.cls_probs, P[:, :, 0, :], atol=1e-5)


@pytest.mark.parametrize("mode", CARRY)
def test_t5_gradients(mode):
    # gamma = 0 at init still gets a gradient: the prior is non-trivial
    m = make(attn_carry=mode).train()
    with torch.no_grad():   # the head is zero-initialised, which blocks all core gradients
        m.head.weight.normal_(0, 0.1)
    F.cross_entropy(m(x_batch()), torch.arange(6) % 5).backward()
    # realformer carries block b-1 -> b within a step, so block 0's gamma is unused
    for blk in (m.core[1:] if mode == "realformer" else m.core):
        assert blk.attn.gamma.grad is not None and blk.attn.gamma.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["logprob", "qk"])
@pytest.mark.parametrize("detach", [False, True])
def test_t5_detach_blocks_state_gradient(mode, detach):
    m = make(attn_carry=mode, carry_detach=detach)
    set_gamma(m, 0.5)
    attn = m.core[0].attn
    g = torch.Generator().manual_seed(5)
    x1 = torch.randn(2, 17, TINY["dim"], generator=g, requires_grad=True)
    x2 = torch.randn(2, 17, TINY["dim"], generator=g, requires_grad=True)
    a1 = attn(x1, t=0)
    a2 = attn(x2, state=a1.state, t=1)
    a2.out.pow(2).sum().backward()
    assert attn.gamma.grad is not None and attn.gamma.grad.abs().sum() > 0
    assert x2.grad is not None
    if detach:
        assert x1.grad is None
    else:
        assert x1.grad is not None and x1.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["none"] + CARRY)
@pytest.mark.parametrize("per_step", [False, True])
def test_t6_extrapolation(mode, per_step):
    m = make(attn_carry=mode, carry_gamma_per_step=per_step)
    if mode != "none":
        set_gamma(m)
        assert m.core[0].attn.gamma.shape == ((3, 4) if per_step else (1, 4))
    with torch.no_grad():
        out = m(x_batch(), num_steps=2 * TINY["loop_steps"])
        logits, graphs = m.logits_and_graphs(x_batch(), 2 * TINY["loop_steps"])
    assert out.shape == (6, 5) and torch.isfinite(out).all()
    assert len(logits) == len(graphs) == 6
    assert graphs[0].shape == (6, 2, 4, 17)
    assert torch.allclose(graphs[-1].sum(-1), torch.ones(6, 2, 4), atol=1e-4)


@pytest.mark.parametrize("mode", ["none"] + CARRY)
@pytest.mark.parametrize("exit_mode", ["entropy", "graph", "both", "either"])
def test_t7_never_exit_equals_forward(mode, exit_mode):
    m = make(attn_carry=mode, exit_mode=exit_mode)
    if mode != "none":
        set_gamma(m)
    x = x_batch()
    with torch.no_grad():
        d = m.dynamic_forward(x, tau=-1.0, graph_eps=-1.0, max_steps=5)
        ref_logits = m(x, num_steps=5)
    assert torch.allclose(d["logits"], ref_logits, atol=1e-6)
    assert d["steps_run"] == 5 and (d["exit_steps"] == 5).all()
    assert d["block_apps"] == 6 * 5 * TINY["core_depth"]
    assert torch.isnan(d["graph_delta"][:, 0]).all()
    assert torch.isfinite(d["graph_delta"][:, 1:]).all()


@pytest.mark.parametrize("mode", ["none"] + CARRY)
def test_t8_compaction(mode):
    m = make(attn_carry=mode)
    if mode != "none":
        set_gamma(m)
    x = x_batch(8)
    # half the samples exit at step 1, one at step 3, the rest run the full budget
    tau = torch.tensor([1e9, -1, 1e9, -1, 1e9, -1, 1e9, -1.0])
    with torch.no_grad():
        d = m.dynamic_forward(x, tau=tau, max_steps=4)
        expect = torch.tensor([1, 4, 1, 4, 1, 4, 1, 4])
        assert torch.equal(d["exit_steps"], expect)
        for i in range(8):
            one = m(x[i:i + 1], num_steps=int(expect[i]))
            assert torch.allclose(d["logits"][i:i + 1], one, atol=1e-5), i
    assert d["block_apps"] == int(expect.sum()) * TINY["core_depth"]
    # frozen samples keep NaN graph deltas after exiting
    assert torch.isnan(d["graph_delta"][0, 1:]).all()


def test_t8_graph_exit_uses_delta():
    m = make(attn_carry="logprob", exit_mode="graph")
    set_gamma(m)
    x = x_batch()
    with torch.no_grad():
        _, graphs = m.logits_and_graphs(x, 4)
        deltas = torch.stack([graph_delta(graphs[t], graphs[t - 1]) for t in range(1, 4)], 1)
        eps = float(deltas[:, 0].median())
        d = m.dynamic_forward(x, graph_eps=eps, max_steps=4)
    # never exits at step 1 (Delta_0 undefined); exits at step 2 iff Delta_1 < eps
    assert (d["exit_steps"] >= 2).all()
    assert torch.equal(d["exit_steps"] == 2, deltas[:, 0] < eps)
    assert torch.allclose(d["graph_delta"][:, 1], deltas[:, 0], atol=1e-5)


@pytest.mark.parametrize("mode", ["none"] + CARRY)
def test_t9_low_precision(mode):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.bfloat16
    m = make(attn_carry=mode).to(dev)
    if mode != "none":
        set_gamma(m, 2.0)
    x = x_batch().to(dev) * 50          # large activations stress the logits
    with torch.no_grad(), torch.autocast(device_type=dev, dtype=dtype):
        h, states = m.embed(x), None
        for t in range(6):
            h, states, cls = m.core_step(h, t, states=states, need_cls=True)
            for s in states:
                for part in (s if isinstance(s, tuple) else (s,)):
                    if part is not None:
                        assert torch.isfinite(part.float().clamp_min(-1e30)).all()
            assert torch.isfinite(cls).all()
        logits = m.classify(h)
        d = m.dynamic_forward(x, max_steps=6)
    assert torch.isfinite(logits).all() and torch.isfinite(d["logits"]).all()


def test_shifted_carry_and_param_report():
    m = make(attn_carry="logprob", carry_source="shifted", carry_gamma_per_step=True)
    rep = m.param_report()
    assert rep["lcga_params"] == TINY["core_depth"] * TINY["num_heads"] * TINY["loop_steps"]
    assert make().param_report()["lcga_params"] == 0
    set_gamma(m)
    with torch.no_grad():
        assert torch.isfinite(m(x_batch(), 5)).all()


def test_fixed_gamma_not_learnable():
    m = make(attn_carry="logprob", carry_gamma_init=1.0, carry_gamma_learnable=False)
    assert all(not blk.attn.gamma.requires_grad for blk in m.core)
    assert torch.equal(m.gamma_values(), torch.ones(2, 1, 4))


def test_record_entropy():
    for mode in ["none"] + CARRY:
        m = make(attn_carry=mode)
        m.set_record_entropy(True)
        with torch.no_grad():
            m(x_batch())
        e = m.core[0].attn.last_row_entropy
        assert e.shape == (6, 4) and (e > 0).all() and (e <= torch.log(torch.tensor(17.0)) + 1e-4).all()
