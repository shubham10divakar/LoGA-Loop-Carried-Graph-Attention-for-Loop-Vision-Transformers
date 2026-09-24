"""
Initialise a LoopViT / LCGA model from pretrained weights (design doc A, sec. 10).

Two sources:

1. A timm plain ViT (`--pretrained deit_small_patch16_224.fb_in1k`,
   `vit_small_patch14_dinov2.lvd142m`, `vit_base_patch16_224.augreg2_in21k_ft_in1k`, ...).
   The loop core is built from L pretrained blocks, which gives a "retrofitted" LoopViT:

       z_0 = emb(x)                      <- pretrained patch/pos/CLS embedding
       z_t+1 = M(z_t + e_t)              <- M = B blocks picked from the L pretrained ones
       y = head(norm(z_T))               <- pretrained final norm, new head

   This model has to match the ViT block exactly, so the architecture args are
   taken from the timm model: LayerNorm instead of RMSNorm, a plain GELU MLP
   instead of ConvGLU, no RoPE (the weights were trained with absolute position
   embeddings), and LayerScale when the ViT has it (DINOv2). e_t is zeroed and
   gamma starts at 0, so the first pass through the core is exactly the chosen
   pretrained blocks and LCGA / vanilla / RealFormer all start from the same function.

   How the B core blocks are chosen from the L pretrained layers (`--pretrained-blocks`),
   following the looped-init schemes of Relaxed Recursive Transformers (Bae et al., 2024):
       stepwise  block b <- layer round(b * L / B)          (12 -> 4: 0, 3, 6, 9)
       lower     block b <- layer b                         (12 -> 4: 0, 1, 2, 3)
       average   block b <- mean of layers b, b+B, b+2B...  (the layers it replaces when unrolled)
       "0,4,8,11" an explicit list of B layer indices
   With --core-depth L --loop-steps 1 the model is exactly the pretrained ViT
   (the non-looped reference baseline).

2. One of this repo's checkpoints (`--init-from runs/loopvit_plantvillage/best.pt`),
   e.g. PlantVillage -> PlantDoc. Every tensor whose name and shape match is copied;
   the head (different class count) and anything else that does not match stays
   freshly initialised. Optimizer / schedule / epoch are NOT restored (use --resume for that).
"""
from __future__ import annotations

import torch
import torch.nn as nn

PRETRAINED_BLOCK_MODES = ("stepwise", "lower", "average")


def _timm():
    try:
        import timm
    except ImportError as e:
        raise SystemExit("--pretrained needs timm: pip install timm") from e
    return timm


def create_timm_vit(name: str, pretrained: bool = True):
    """The timm ViT, without its classifier. Downloads weights on first use."""
    return _timm().create_model(name, pretrained=pretrained, num_classes=0)


def _check_supported(vit):
    from timm.models.vision_transformer import Mlp, VisionTransformer
    if not isinstance(vit, VisionTransformer):
        raise SystemExit(f"--pretrained must be a timm VisionTransformer, got {type(vit).__name__}")
    problems = []
    if not getattr(vit, "has_class_token", True) or vit.num_prefix_tokens != 1:
        problems.append(f"needs exactly 1 CLS token and no register tokens "
                        f"(has {vit.num_prefix_tokens} prefix tokens)")
    if getattr(vit, "no_embed_class", False):
        problems.append("no_embed_class=True is not supported")
    if not isinstance(vit.norm_pre, nn.Identity):
        problems.append("pre-norm (norm_pre, e.g. CLIP) is not supported")
    if not isinstance(getattr(vit, "fc_norm", nn.Identity()), nn.Identity):
        problems.append("fc_norm (avg-pool ViTs) is not supported")
    if vit.global_pool != "token":
        problems.append(f"global_pool={vit.global_pool!r}; only 'token' (CLS) is supported")
    b0 = vit.blocks[0]
    if not isinstance(b0.mlp, Mlp) or not isinstance(b0.mlp.act, nn.GELU) \
            or b0.mlp.act.approximate != "none":
        problems.append(f"MLP must be timm Mlp with exact GELU, got {type(b0.mlp).__name__}")
    if not isinstance(b0.attn.q_norm, nn.Identity) or not isinstance(b0.attn.k_norm, nn.Identity):
        problems.append("qk-norm is not supported")
    if not isinstance(b0.norm1, nn.LayerNorm):
        problems.append(f"block norm must be LayerNorm, got {type(b0.norm1).__name__}")
    if problems:
        raise SystemExit("unsupported --pretrained model:\n  " + "\n  ".join(problems))


def timm_arch(vit) -> dict:
    """LoopViT architecture args that make the core blocks identical to the ViT's."""
    _check_supported(vit)
    b0 = vit.blocks[0]
    ls = b0.ls1.gamma.detach() if hasattr(b0.ls1, "gamma") else None
    pc = getattr(vit, "pretrained_cfg", {}) or {}
    return {
        "dim": vit.embed_dim,
        "num_heads": b0.attn.num_heads,
        "patch_size": vit.patch_embed.patch_size[0],
        "mlp_ratio": b0.mlp.fc1.out_features / vit.embed_dim,
        "qkv_bias": b0.attn.qkv.bias is not None,
        "ffn": "vanilla",
        "rope": False,
        "norm": "layer",
        "num_cls_tokens": 1,
        "pool": "cls",
        # any positive value creates the parameters; the real values are copied in
        "layer_scale": float(ls.abs().mean()) if ls is not None else 0.0,
        "input_mean": tuple(float(v) for v in pc.get("mean", (0.485, 0.456, 0.406))),
        "input_std": tuple(float(v) for v in pc.get("std", (0.229, 0.224, 0.225))),
        "_depth": len(vit.blocks),
    }


def block_sources(spec: str, B: int, L: int) -> list[list[int]]:
    """For each of the B core blocks, the pretrained layer indices it is built from."""
    if spec == "stepwise":
        return [[min(L - 1, round(b * L / B))] for b in range(B)]
    if spec == "lower":
        if B > L:
            raise SystemExit(f"--pretrained-blocks lower needs core_depth <= {L}")
        return [[b] for b in range(B)]
    if spec == "average":
        if B > L:
            raise SystemExit(f"--pretrained-blocks average needs core_depth <= {L}")
        return [list(range(b, L, B)) for b in range(B)]
    try:
        idx = [int(v) for v in spec.split(",")]
    except ValueError:
        raise SystemExit(f"--pretrained-blocks must be one of {PRETRAINED_BLOCK_MODES} "
                         f"or a comma list of layer indices, got {spec!r}") from None
    if len(idx) != B or any(not 0 <= i < L for i in idx):
        raise SystemExit(f"--pretrained-blocks {spec!r}: need {B} indices in [0, {L - 1}]")
    return [[i] for i in idx]


@torch.no_grad()
def load_timm_vit(model, vit, spec: str = "stepwise") -> list[list[int]]:
    """Copy the timm ViT's weights into `model` (a LoopViT built with `timm_arch(vit)`).
    Returns the layer indices used for each core block."""
    from timm.layers import resample_abs_pos_embed
    _check_supported(vit)
    cfg = model.cfg
    L, B = len(vit.blocks), len(model.core)
    src = block_sources(spec, B, L)

    model.patch_embed.proj.weight.copy_(vit.patch_embed.proj.weight)
    model.patch_embed.proj.bias.copy_(vit.patch_embed.proj.bias)
    model.cls_token.copy_(vit.cls_token)
    g = model.patch_embed.grid
    model.pos_embed.copy_(resample_abs_pos_embed(vit.pos_embed, new_size=(g, g),
                                                 num_prefix_tokens=1, verbose=False))

    def mean_of(get):
        return torch.stack([get(vit.blocks[i]).float() for i in layers]).mean(0)

    for blk, layers in zip(model.core, src):
        pairs = [(blk.norm1.weight, lambda v: v.norm1.weight), (blk.norm1.bias, lambda v: v.norm1.bias),
                 (blk.attn.qkv.weight, lambda v: v.attn.qkv.weight),
                 (blk.attn.proj.weight, lambda v: v.attn.proj.weight),
                 (blk.attn.proj.bias, lambda v: v.attn.proj.bias),
                 (blk.norm2.weight, lambda v: v.norm2.weight), (blk.norm2.bias, lambda v: v.norm2.bias),
                 (blk.ffn.fc1.weight, lambda v: v.mlp.fc1.weight), (blk.ffn.fc1.bias, lambda v: v.mlp.fc1.bias),
                 (blk.ffn.fc2.weight, lambda v: v.mlp.fc2.weight), (blk.ffn.fc2.bias, lambda v: v.mlp.fc2.bias)]
        if blk.attn.qkv.bias is not None:
            pairs.append((blk.attn.qkv.bias, lambda v: v.attn.qkv.bias))
        if blk.ls1 is not None:
            pairs += [(blk.ls1, lambda v: v.ls1.gamma), (blk.ls2, lambda v: v.ls2.gamma)]
        for dst, get in pairs:
            dst.copy_(mean_of(get))

    model.norm.weight.copy_(vit.norm.weight)
    model.norm.bias.copy_(vit.norm.bias)
    if model.step_embed is not None:
        model.step_embed.zero_()        # first pass through the core = the pretrained blocks
    # the head stays zero-initialised (LoopViT._init_weights)
    assert cfg.norm == "layer" and cfg.ffn == "vanilla" and not cfg.rope
    return src


@torch.no_grad()
def load_matching(model, ckpt_path: str) -> tuple[list[str], list[str]]:
    """Copy every tensor of a repo checkpoint whose name and shape match.
    Returns (loaded keys, skipped keys)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = ckpt.get("model", ckpt)
    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in own.items():
        if k in src and src[k].shape == v.shape:
            v.copy_(src[k])
            loaded.append(k)
        else:
            skipped.append(k)
    return loaded, skipped
