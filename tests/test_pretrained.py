"""
pretrained.py: a LoopViT built from a timm ViT reproduces it exactly.

    python -m pytest tests -q
"""
from __future__ import annotations

import pytest
import torch

timm = pytest.importorskip("timm")
from timm.models.vision_transformer import VisionTransformer  # noqa: E402

from loop_vit import LoopViT, LoopViTConfig  # noqa: E402
from pretrained import block_sources, load_matching, load_timm_vit, timm_arch  # noqa: E402

CARRY = ["none", "logprob", "qk", "realformer"]


def tiny_vit(init_values=None, img=32):
    torch.manual_seed(0)
    vit = VisionTransformer(img_size=img, patch_size=8, embed_dim=64, depth=4, num_heads=4,
                            num_classes=0, init_values=init_values).eval()
    with torch.no_grad():   # non-trivial norms / LayerScale / pos so a wrong mapping shows
        for n, p in vit.named_parameters():
            if "norm" in n or "ls" in n or "pos_embed" in n or "cls_token" in n:
                p.add_(torch.randn_like(p) * 0.1)
    return vit


def retrofit(vit, image_size=32, spec="stepwise", **kw):
    arch = timm_arch(vit)
    arch.pop("_depth")
    cfg = LoopViTConfig(image_size=image_size, num_classes=5, drop_path=0.0,
                        **{**arch, **kw})
    m = LoopViT(cfg).eval()
    load_timm_vit(m, vit, spec)
    return m


def features(m, x, steps=None):
    with torch.no_grad():
        return m.norm(m.loop(m.embed(x), steps or m.cfg.loop_steps))[:, 0]


@pytest.mark.parametrize("init_values", [None, 1e-1])
@pytest.mark.parametrize("mode", CARRY)
def test_unlooped_equals_timm(mode, init_values):
    """core_depth = L, loop_steps = 1: the model *is* the pretrained ViT (any carry mode,
    since nothing is carried within a single step except RealFormer, whose gamma is 0)."""
    vit = tiny_vit(init_values)
    m = retrofit(vit, core_depth=4, loop_steps=1, spec="lower", attn_carry=mode)
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        ref = vit.forward_features(x)[:, 0]
    torch.testing.assert_close(features(m, x), ref, atol=1e-5, rtol=1e-5)


def test_looped_first_pass_is_the_chosen_blocks():
    """B=2 stepwise from L=4 -> layers 0, 2; one step equals running those two timm blocks."""
    vit = tiny_vit(1e-1)
    m = retrofit(vit, core_depth=2, loop_steps=3, attn_carry="logprob")
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        h = vit._pos_embed(vit.patch_embed(x))
        for i in (0, 2):
            h = vit.blocks[i](h)
        ref = vit.norm(h)[:, 0]
    torch.testing.assert_close(features(m, x, steps=1), ref, atol=1e-5, rtol=1e-5)


def test_pos_embed_resampled_for_higher_resolution():
    vit = tiny_vit()
    m = retrofit(vit, image_size=64, core_depth=2, loop_steps=2)
    assert m.pos_embed.shape == (1, 1 + 8 * 8, 64)
    torch.testing.assert_close(m.pos_embed[:, :1], vit.pos_embed[:, :1])
    with torch.no_grad():
        assert torch.isfinite(m(torch.randn(2, 3, 64, 64))).all()


def test_block_sources():
    assert block_sources("stepwise", 4, 12) == [[0], [3], [6], [9]]
    assert block_sources("lower", 4, 12) == [[0], [1], [2], [3]]
    assert block_sources("average", 4, 12) == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
    assert block_sources("0,4,8,11", 4, 12) == [[0], [4], [8], [11]]
    with pytest.raises(SystemExit):
        block_sources("0,4", 4, 12)


def test_average_init_is_mean_of_layers():
    vit = tiny_vit()
    m = retrofit(vit, core_depth=2, loop_steps=2, spec="average")
    want = (vit.blocks[0].mlp.fc1.weight + vit.blocks[2].mlp.fc1.weight) / 2
    torch.testing.assert_close(m.core[0].ffn.fc1.weight, want)


def test_rejects_register_tokens():
    vit = VisionTransformer(img_size=32, patch_size=8, embed_dim=64, depth=2, num_heads=4,
                            num_classes=0, reg_tokens=4)
    with pytest.raises(SystemExit):
        timm_arch(vit)


def test_init_from_skips_head(tmp_path):
    vit = tiny_vit()
    src = retrofit(vit, core_depth=2, loop_steps=2, attn_carry="logprob")
    path = tmp_path / "ck.pt"
    torch.save({"model": src.state_dict()}, path)
    arch = timm_arch(vit)
    arch.pop("_depth")
    dst = LoopViT(LoopViTConfig(image_size=32, num_classes=7, core_depth=2, loop_steps=2,
                                attn_carry="logprob", **arch))
    loaded, skipped = load_matching(dst, str(path))
    assert set(skipped) == {"head.weight", "head.bias"}
    torch.testing.assert_close(dst.core[1].attn.qkv.weight, src.core[1].attn.qkv.weight)
