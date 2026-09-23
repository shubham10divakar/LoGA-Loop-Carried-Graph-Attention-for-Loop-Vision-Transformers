"""
LoopViT + LCGA (Loop-Carried Graph Attention) for image classification.

Base architecture: "LoopViT: Scaling Visual ARC with Looped Transformers"
(arXiv:2602.02156), ported to whole-image classification (see the original
`loopvit_paper/loop_vit.py`). Everything below the LCGA additions is the same
model; with `attn_carry="none"` and `exit_mode="entropy"` the outputs are
bit-identical to the original (tests/test_lcga.py::test_t1_baseline_equivalence).

LoopViT recap
-------------
        z_0   = emb(x)
        z_t+1 = M_theta(z_t + e_t)          eq. (2) in the paper
        y     = head(z_T)
B distinct Hybrid blocks (2D-RoPE MHSA + Heterogeneous ConvGLU) are applied T
times; e_t is a learned step embedding with identity extrapolation past T.

LCGA (design doc A, "Crystallizing Attention Graphs")
-----------------------------------------------------
Block b at step t is the *same function* as block b at step t-1, so its
attention graph from step t-1 is a semantically consistent prior for step t:

  logprob (eq. A2):  L~_t = Q_t K_t^T / sqrt(d) + gamma_{b,h}(t) * clamp(log A_{t-1}, min=c)
                     A_t  = softmax(L~_t)       -- recursive "posterior" over edges
  qk      (eq. A3):  L~_t = [Q_t K_t^T + gamma_{b,h}(t) * Q_{t-1} K_{t-1}^T] / sqrt(d)
                     -- Markov-1 carry, computed flash-style by concatenating
                        [q, gamma*q_prev] and [k, k_prev] along the head dim.

Controls / ablations (design doc sec. 10):
  realformer         residual scores across *blocks within one step*
                     (L~_b = QK^T/sqrt(d) + gamma * L~_{b-1}); no carry across steps.
  carry_source       "same" (b -> b, proposed) | "shifted" (b -> b+1).
  carry_gamma_learnable=False with carry_gamma_init=1 gives "gamma fixed at 1".

gamma is initialised to 0, so at init LCGA is exactly vanilla LoopViT.

Dynamic exit (eq. A4): besides the paper's entropy criterion, a sample can halt
when its attention graph stops changing: Delta_t = mean_{b,h} KL(p_t || p_{t-1})
over the CLS-query attention row. `dynamic_forward` also compacts the active
batch so exited samples really stop costing compute (`block_apps`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTN_CARRY_MODES = ("none", "logprob", "qk", "realformer")
EXIT_MODES = ("entropy", "graph", "both", "either")
CARRY_SOURCES = ("same", "shifted")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class LoopViTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 10

    dim: int = 384
    core_depth: int = 4          # B: distinct hybrid blocks in the shared core
    loop_steps: int = 3          # T: iterations of the core during training
    num_heads: int = 6
    mlp_ratio: float = 4.0       # FFN hidden = dim * mlp_ratio
    qkv_bias: bool = True
    dropout: float = 0.0
    attn_dropout: float = 0.0
    drop_path: float = 0.1       # max stochastic-depth rate over the B*T applications

    ffn: str = "hybrid"          # "hybrid" (ConvGLU + DW-Conv) | "vanilla" (plain MLP)
    rope: bool = True            # 2D rotary embeddings in the attention
    step_embedding: bool = True  # e_t in eq. (2)
    num_cls_tokens: int = 1      # the paper's "task tokens"; they bypass the DW-Conv
    pool: str = "cls"            # "cls" | "mean"

    # dynamic exit (inference only)
    exit_tau: float = 0.05       # halt a sample once its entropy (nats) drops below this
    min_loop_steps: int = 1
    max_loop_steps: int = 0      # 0 = use loop_steps

    # --- LCGA: loop-carried graph attention ---
    attn_carry: str = "none"          # "none" | "logprob" | "qk" | "realformer" (control)
    carry_gamma_init: float = 0.0     # gamma init; 0 => identical to vanilla at init
    carry_gamma_per_step: bool = False
    carry_gamma_learnable: bool = True  # False + init 1.0 = "gamma fixed at 1" ablation
    carry_detach: bool = False
    carry_clamp: float = -10.0        # floor on log A_{t-1} in logprob mode
    carry_source: str = "same"        # "same" (b -> b) | "shifted" (b -> b+1) ablation

    # --- dynamic exit criteria (extends exit_tau etc.) ---
    exit_mode: str = "entropy"        # "entropy" | "graph" | "both" | "either"
    exit_graph_eps: float = 0.01      # nats; CLS-row KL(A_t || A_{t-1}) threshold

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Stochastic depth. The rate is passed per call because a shared block is
    applied at several depths and each application gets its own rate."""
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * mask / keep


class RMSNorm(nn.Module):
    """Pre-norm of choice in the paper (eq. 6/7): cheaper than LayerNorm and
    numerically better behaved when the same weights run many times."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return x32.to(dtype) * self.weight


class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, dim):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                      # (N, C, H, W)
        x = self.proj(x)                       # (N, D, g, g)
        return x.flatten(2).transpose(1, 2)    # (N, P, D)


class Rotary2D(nn.Module):
    """Axial 2D RoPE: half of each head's channels encode the patch row, the
    other half the column. The first `n_prefix` tokens (CLS / task tokens) are
    left un-rotated, matching `no_rope=num_task_tokens` in the reference code."""

    def __init__(self, head_dim: int, grid: int, n_prefix: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"2D RoPE needs head_dim divisible by 4, got {head_dim}")
        q = head_dim // 4                                   # frequencies per axis
        freqs = 1.0 / (theta ** (torch.arange(q).float() / q))
        ang = torch.outer(torch.arange(grid).float(), freqs)         # (grid, q)
        row = ang[:, None, :].expand(grid, grid, q)
        col = ang[None, :, :].expand(grid, grid, q)
        a = torch.cat([row, col], dim=-1).reshape(grid * grid, 2 * q)  # (P, head_dim/2)
        self.register_buffer("cos", a.cos()[None, None], persistent=False)
        self.register_buffer("sin", a.sin()[None, None], persistent=False)
        self.n_prefix = n_prefix

    def forward(self, t):                      # (N, h, T, d)
        p = self.n_prefix
        prefix, x = t[:, :, :p, :], t[:, :, p:, :]
        x1, x2 = x.chunk(2, dim=-1)
        cos, sin = self.cos.to(x.dtype), self.sin.to(x.dtype)
        x = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.cat([prefix, x], dim=2) if p else x


@dataclass
class AttnOut:
    out: torch.Tensor                  # (N, T_tok, D)
    state: object | None               # logprob: (N,h,T_tok,T_tok) fp32 log-probs
                                       # realformer: (N,h,T_tok,T_tok) fp32 carried logits
                                       # qk: tuple(q, k), each (N,h,T_tok,d), post-RoPE
                                       # none: None
    cls_probs: torch.Tensor | None     # (N, h, T_tok) attention row of CLS query 0, fp32


class Attention(nn.Module):
    """MHSA with optional 2D rotary embeddings (eq. 3-5) and an optional
    loop-carried attention graph (LCGA, eq. A2 / A3)."""

    def __init__(self, dim, num_heads, grid, n_prefix, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0, rope=True, carry="none", gamma_init=0.0,
                 gamma_steps=1, detach=False, clamp=-10.0, gamma_learnable=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        if carry not in ATTN_CARRY_MODES:
            raise ValueError(f"attn_carry must be one of {ATTN_CARRY_MODES}, got {carry!r}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rotary = Rotary2D(self.head_dim, grid, n_prefix) if rope else None

        self.carry, self.detach, self.clamp = carry, detach, clamp
        # gamma only exists when there is something to carry, so a "none" model
        # has exactly the original state-dict keys.
        self.gamma = (nn.Parameter(torch.full((gamma_steps, num_heads), float(gamma_init)),
                                   requires_grad=gamma_learnable)
                      if carry != "none" else None)
        # analysis hook: when True, forward stores the mean per-query attention
        # entropy (N, h) of the dense graph in `last_row_entropy`.
        self.record_entropy = False
        self.last_row_entropy: torch.Tensor | None = None

    def _gamma(self, t: int) -> torch.Tensor:
        """gamma_{b,h}(t) (sec. 2.3), identity-extrapolated past the last trained step."""
        return self.gamma[min(t, self.gamma.shape[0] - 1)].view(1, -1, 1, 1)

    def _qkv(self, x):
        N, T, _ = x.shape
        q, k, v = (self.qkv(x)
                   .reshape(N, T, 3, self.num_heads, self.head_dim)
                   .permute(2, 0, 3, 1, 4))            # 3 x (N, h, T, d)
        if self.rotary is not None:
            q, k = self.rotary(q), self.rotary(k)
        return q, k, v

    def _qk_logits(self, q, k, state, t: int, rows: slice = slice(None)) -> torch.Tensor:
        """Dense fp32 logits of eq. A1 (no state) or eq. A3 (qk state) for query rows `rows`."""
        with torch.autocast(device_type=q.device.type, enabled=False):
            lg = q[:, :, rows].float() @ k.float().transpose(-2, -1)
            if state is not None:
                qp, kp = state
                lg = lg + self._gamma(t) * (qp[:, :, rows].float() @ kp.float().transpose(-2, -1))
            return lg * self.scale

    def forward(self, x, state=None, t: int = 0, need_cls: bool = False) -> AttnOut:
        N, T, D = x.shape
        q, k, v = self._qkv(x)
        drop = self.attn_drop if self.training else 0.0
        cls_probs, new_state, dense_logp = None, None, None

        if self.carry == "none":
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
            if need_cls:
                cls_probs = self._qk_logits(q, k, None, t, slice(0, 1)).softmax(-1).squeeze(2)
            if self.record_entropy:
                dense_logp = self._qk_logits(q, k, None, t).log_softmax(-1)

        elif self.carry in ("logprob", "realformer"):
            # eq. A2 (logprob) / RealFormer-style residual scores (realformer)
            with torch.autocast(device_type=x.device.type, enabled=False):
                logits = (q.float() @ k.float().transpose(-2, -1)) * self.scale   # (N,h,T,T)
                if state is not None:
                    prior = state.clamp_min(self.clamp) if self.carry == "logprob" else state
                    logits = logits + self._gamma(t) * prior
                logA = logits.log_softmax(-1)
            probs = logA.exp()
            probs_d = F.dropout(probs, drop, self.training)     # dropout AFTER carry capture
            out = probs_d.to(v.dtype) @ v
            new_state = logA if self.carry == "logprob" else logits
            if self.detach:
                new_state = new_state.detach()
            if need_cls:
                cls_probs = probs[:, :, 0, :]
            dense_logp = logA if self.record_entropy else None

        else:  # "qk", eq. A3 via the concatenation trick
            if state is None:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, scale=self.scale)
            else:
                qp, kp = state
                g = self._gamma(t)
                q_cat = torch.cat([q, g.to(q.dtype) * qp.to(q.dtype)], dim=-1)   # (N,h,T,2d)
                k_cat = torch.cat([k, kp.to(k.dtype)], dim=-1)
                out = F.scaled_dot_product_attention(q_cat, k_cat, v, dropout_p=drop,
                                                     scale=self.scale)
            new_state = (q.detach(), k.detach()) if self.detach else (q, k)
            if need_cls:
                cls_probs = self._qk_logits(q, k, state, t, slice(0, 1)).softmax(-1).squeeze(2)
            if self.record_entropy:
                dense_logp = self._qk_logits(q, k, state, t).log_softmax(-1)

        if dense_logp is not None:
            with torch.no_grad():
                lp = dense_logp.detach().float()
                self.last_row_entropy = -(lp.exp() * lp).sum(-1).mean(-1)   # (N, h)

        out = out.transpose(1, 2).reshape(N, T, D)
        return AttnOut(self.proj_drop(self.proj(out)), new_state, cls_probs)


class HeteroConvGLU(nn.Module):
    """The paper's Heterogeneous ConvGLU FFN (eq. 8-10, Fig. 4D).

    The gate branch is split by token type: CLS / task tokens bypass the
    depth-wise convolution so the abstract rule token is never smeared across
    the grid, while image tokens are folded back into (g, g) and convolved."""

    def __init__(self, dim, hidden, grid, n_prefix, drop=0.0):
        super().__init__()
        self.grid, self.n_prefix = grid, n_prefix
        self.fc1 = nn.Linear(dim, hidden * 2)          # -> [X_gate, X_val]
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):                              # (N, T, D)
        gate, val = self.fc1(x).chunk(2, dim=-1)
        p, g = self.n_prefix, self.grid
        task, img = gate[:, :p], gate[:, p:]
        N, P, H = img.shape
        img = self.dw(img.transpose(1, 2).reshape(N, H, g, g)).flatten(2).transpose(1, 2)
        gate = torch.cat([task, img], dim=1) if p else img
        return self.drop(self.fc2(self.act(gate) * val))


class MLP(nn.Module):
    """Vanilla FFN, for the Hybrid-vs-Vanilla ablation (`ffn: vanilla`)."""

    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class HybridBlock(nn.Module):
    """One layer of the recurrent core: pre-norm MHSA + ConvGLU (Fig. 4C)."""

    def __init__(self, cfg: LoopViTConfig, grid: int):
        super().__init__()
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg.dim, cfg.num_heads, grid, cfg.num_cls_tokens,
                              cfg.qkv_bias, cfg.attn_dropout, cfg.dropout, cfg.rope,
                              carry=cfg.attn_carry, gamma_init=cfg.carry_gamma_init,
                              gamma_steps=cfg.loop_steps if cfg.carry_gamma_per_step else 1,
                              detach=cfg.carry_detach, clamp=cfg.carry_clamp,
                              gamma_learnable=cfg.carry_gamma_learnable)
        self.norm2 = RMSNorm(cfg.dim)
        if cfg.ffn == "hybrid":
            self.ffn = HeteroConvGLU(cfg.dim, hidden, grid, cfg.num_cls_tokens, cfg.dropout)
        elif cfg.ffn == "vanilla":
            self.ffn = MLP(cfg.dim, hidden, cfg.dropout)
        else:
            raise ValueError(f"ffn must be 'hybrid' or 'vanilla', got {cfg.ffn!r}")

    def forward(self, x, dp: float = 0.0, state=None, t: int = 0, need_cls: bool = False):
        # When drop_path drops the attention branch for a sample, its graph state
        # is still computed and carried, so the graph stays well-defined (sec. 4.2).
        a = self.attn(self.norm1(x), state=state, t=t, need_cls=need_cls)
        x = x + drop_path(a.out, dp, self.training)
        x = x + drop_path(self.ffn(self.norm2(x)), dp, self.training)
        return x, a.state, a.cls_probs


# --------------------------------------------------------------------------- #
# per-sample state helpers (batch compaction in dynamic_forward)
# --------------------------------------------------------------------------- #
def gather_state(s, idx: torch.Tensor):
    """Index the batch dim of a carried state (tensor, (q, k) tuple or None)."""
    if s is None:
        return None
    if isinstance(s, tuple):
        return tuple(x[idx] for x in s)
    return s[idx]


def scatter_state(full, new, idx: torch.Tensor):
    """Write the active samples' new state back into the full-batch buffer.
    Frozen samples keep their last graph (S_k = S_t, sec. 5.3)."""
    if new is None:
        return full
    if isinstance(new, tuple):
        return tuple(scatter_state(f, n, idx) for f, n in zip(full, new))
    full = full.clone()
    full[idx] = new.to(full.dtype)
    return full


def graph_delta(p: torch.Tensor, p_prev: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Delta_t = mean_{b,h} KL(p_t || p_{t-1}) of the CLS attention rows (eq. A4).
    p, p_prev: (N, B, h, T_tok). Returns (N,) in nats, computed in fp32."""
    p, q = p.float(), p_prev.float()
    kl = (p * ((p + eps).log() - (q + eps).log())).sum(-1)      # (N, B, h)
    return kl.mean(dim=(1, 2))


# --------------------------------------------------------------------------- #
# LoopViT
# --------------------------------------------------------------------------- #
class LoopViT(nn.Module):
    def __init__(self, cfg: LoopViTConfig):
        super().__init__()
        if cfg.pool not in ("cls", "mean"):
            raise ValueError("pool must be 'cls' or 'mean'")
        if cfg.pool == "cls" and cfg.num_cls_tokens < 1:
            raise ValueError("pool='cls' needs num_cls_tokens >= 1")
        if cfg.min_loop_steps < 1:
            raise ValueError("min_loop_steps must be >= 1")
        if cfg.attn_carry not in ATTN_CARRY_MODES:
            raise ValueError(f"attn_carry must be one of {ATTN_CARRY_MODES}")
        if cfg.exit_mode not in EXIT_MODES:
            raise ValueError(f"exit_mode must be one of {EXIT_MODES}")
        if cfg.carry_source not in CARRY_SOURCES:
            raise ValueError(f"carry_source must be one of {CARRY_SOURCES}")
        if cfg.num_cls_tokens < 1:
            # the graph signal (eq. A4) and cls_probs use the CLS query row
            if cfg.exit_mode != "entropy":
                raise ValueError(f"exit_mode={cfg.exit_mode!r} needs num_cls_tokens >= 1")
        self.cfg = cfg
        D, n_cls = cfg.dim, cfg.num_cls_tokens

        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_chans, D)
        n_tok = self.patch_embed.num_patches + n_cls
        self.cls_token = nn.Parameter(torch.zeros(1, n_cls, D)) if n_cls else None
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tok, D))
        self.pos_drop = nn.Dropout(cfg.dropout)

        # M_theta: B distinct blocks, reused on every iteration.
        self.core = nn.ModuleList([HybridBlock(cfg, self.patch_embed.grid)
                                   for _ in range(cfg.core_depth)])

        # e_t: one learned vector per iteration; steps past T reuse the last one.
        self.step_embed = (nn.Parameter(torch.zeros(cfg.loop_steps, 1, 1, D))
                           if cfg.step_embedding else None)

        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, cfg.num_classes)
        self._init_weights()

    # ---- init --------------------------------------------------------------
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.step_embed is not None:
            nn.init.trunc_normal_(self.step_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                w = m.weight.data
                nn.init.xavier_uniform_(w.view(w.shape[0], -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ---- pieces ------------------------------------------------------------
    def _drop_path_rates(self, total: int):
        """Linearly increasing rate over the UNROLLED depth (1 ... B*T)."""
        if total <= 1:
            return [0.0] * total
        return [self.cfg.drop_path * i / (total - 1) for i in range(total)]

    def _step_vector(self, t: int):
        """e_t, with identity extrapolation past the training budget."""
        if self.step_embed is None:
            return None
        return self.step_embed[min(t, self.step_embed.shape[0] - 1)]

    def embed(self, x):
        """z_0 = emb(x)."""
        h = self.patch_embed(x)
        if self.cls_token is not None:
            h = torch.cat([self.cls_token.expand(h.shape[0], -1, -1), h], dim=1)
        return self.pos_drop(h + self.pos_embed)

    def core_step(self, h, t: int, rates=None, states=None, need_cls: bool = False):
        """z_{t+1} = M_theta(z_t + e_t), carrying one graph state per block (eq. A2/A3).

        Returns (h, new_states, cls) with new_states a list of B per-block states
        (to pass to the next step) and cls the (N, B, h, T_tok) CLS attention
        rows when `need_cls`, else None."""
        e = self._step_vector(t)
        if e is not None:
            h = h + e
        B = len(self.core)
        states = states if states is not None else [None] * B
        if self.cfg.carry_source == "shifted":
            states = [states[(b - 1) % B] for b in range(B)]      # block b-1 -> block b
        chain = self.cfg.attn_carry == "realformer"
        new_states, cls, s_chain = [], [], None
        for b, blk in enumerate(self.core):
            s_in = s_chain if chain else states[b]
            h, s, c = blk(h, rates[t * B + b] if rates else 0.0, state=s_in, t=t,
                          need_cls=need_cls)
            if chain:
                s_chain, s = s, None          # RealFormer: carried across blocks, not steps
            new_states.append(s)
            cls.append(c)
        return h, new_states, (torch.stack(cls, 1) if need_cls else None)

    def loop(self, h, num_steps: int, return_all: bool = False, return_graphs: bool = False):
        """Run the core `num_steps` times, threading the per-block graph states.

        Returns h (or the list of h after every step with `return_all`); with
        `return_graphs` returns (that, list of per-step (N, B, h, T_tok) CLS rows)."""
        rates = self._drop_path_rates(len(self.core) * num_steps)
        hs, graphs, states = [], [], None
        for t in range(num_steps):
            h, states, cls = self.core_step(h, t, rates, states, need_cls=return_graphs)
            if return_all:
                hs.append(h)
            if return_graphs:
                graphs.append(cls)
        out = hs if return_all else h
        return (out, graphs) if return_graphs else out

    def classify(self, h):
        h = self.norm(h)
        pooled = (h[:, :self.cfg.num_cls_tokens].mean(dim=1) if self.cfg.pool == "cls"
                  else h[:, self.cfg.num_cls_tokens:].mean(dim=1))
        return self.head(pooled)

    # ---- forward -----------------------------------------------------------
    def forward(self, x, num_steps: int | None = None):
        return self.classify(self.loop(self.embed(x), num_steps or self.cfg.loop_steps))

    def logits_per_step(self, x, num_steps: int | None = None):
        """Logits read out after every iteration: list of (N, num_classes).
        Shows the 'crystallization' of the prediction across the loop."""
        states = self.loop(self.embed(x), num_steps or self.cfg.loop_steps, return_all=True)
        return [self.classify(s) for s in states]

    def logits_and_graphs(self, x, num_steps: int | None = None):
        """(list of per-step logits, list of per-step (N, B, h, T_tok) CLS attention rows)."""
        hs, graphs = self.loop(self.embed(x), num_steps or self.cfg.loop_steps,
                               return_all=True, return_graphs=True)
        return [self.classify(s) for s in hs], graphs

    @staticmethod
    def entropy(logits):
        """H_t = -sum_c p log p, in nats (eq. 11, over classes instead of pixels)."""
        logp = F.log_softmax(logits.float(), dim=-1)
        return -(logp.exp() * logp).sum(-1)

    @staticmethod
    def exit_decision(ent, delta, t: int, tau, graph_eps: float, mode: str):
        """Exit rule of sec. 5.2 (before the min_steps check). ent/delta: (n,)."""
        ent_ok = ent < tau
        graph_ok = (delta < graph_eps) if (t >= 1 and delta is not None) \
            else torch.zeros_like(ent_ok)
        if mode == "entropy":
            return ent_ok
        if mode == "graph":
            return graph_ok
        if mode == "both":
            return ent_ok & graph_ok
        return ent_ok | graph_ok

    @torch.no_grad()
    def dynamic_forward(self, x, tau=None, min_steps: int | None = None,
                        max_steps: int | None = None, exit_mode: str | None = None,
                        graph_eps: float | None = None):
        """Dynamic Exit (sec. 3.3 of the paper, extended by sec. 5 of design A).

        Per sample: iterate until the exit rule fires, then freeze that sample's
        state (z_k = z_t and S_k = S_t) and drop it from the active batch, so
        further iterations cost it nothing. `tau` may be a float or an (N,) tensor.

        Returns a dict with the final `logits`, per-sample `exit_steps`, the
        (N, steps_run) `entropy` and `graph_delta` traces (graph_delta is NaN at
        t=0 and for samples that had already exited), `steps_run` (the deepest
        sample) and `block_apps` (block applications actually executed, summed
        over samples -- the real compute metric)."""
        cfg = self.cfg
        tau = cfg.exit_tau if tau is None else tau
        min_steps = cfg.min_loop_steps if min_steps is None else min_steps
        max_steps = max_steps or cfg.max_loop_steps or cfg.loop_steps
        mode = exit_mode or cfg.exit_mode
        graph_eps = cfg.exit_graph_eps if graph_eps is None else graph_eps
        if mode not in EXIT_MODES:
            raise ValueError(f"exit_mode must be one of {EXIT_MODES}")
        if mode != "entropy" and cfg.num_cls_tokens < 1:
            raise ValueError("graph-based exit needs num_cls_tokens >= 1")
        need_cls = cfg.num_cls_tokens >= 1
        tau_t = tau if torch.is_tensor(tau) else None

        h = self.embed(x).clone()
        N, B = x.shape[0], len(self.core)
        dev = x.device
        done = torch.zeros(N, dtype=torch.bool, device=dev)
        exit_steps = torch.full((N,), max_steps, dtype=torch.long, device=dev)
        states = [None] * B
        prev_cls = None
        logits = ent = None
        ent_trace, delta_trace, block_apps, t = [], [], 0, 0
        for t in range(max_steps):
            active = (~done).nonzero(as_tuple=True)[0]
            h_a = h[active]
            st_a = [gather_state(s, active) for s in states]
            h_new, st_new, cls_new = self.core_step(h_a, t, states=st_a, need_cls=need_cls)
            block_apps += active.numel() * B
            h[active] = h_new.to(h.dtype)
            states = [n if s is None else scatter_state(s, n, active)
                      for s, n in zip(states, st_new)]

            lg_a = self.classify(h_new)
            if logits is None:
                logits = torch.zeros(N, lg_a.shape[-1], dtype=lg_a.dtype, device=dev)
                ent = torch.zeros(N, device=dev)
            logits[active] = lg_a.to(logits.dtype)
            ent_a = self.entropy(lg_a)
            ent[active] = ent_a

            delta = torch.full((N,), float("nan"), device=dev)
            delta_a = None
            if need_cls:
                cls_new = cls_new.float()
                if prev_cls is None:
                    prev_cls = cls_new.clone()
                else:
                    delta_a = graph_delta(cls_new, prev_cls[active])
                    delta[active] = delta_a
                    prev_cls[active] = cls_new
            ent_trace.append(ent.clone())
            delta_trace.append(delta)

            if (t + 1) >= min_steps:
                tau_a = tau_t[active] if tau_t is not None else tau
                newly_a = self.exit_decision(ent_a, delta_a, t, tau_a, graph_eps, mode)
                newly = active[newly_a]
                exit_steps[newly] = t + 1
                done[newly] = True
            if bool(done.all()):
                break
        return {"logits": logits, "exit_steps": exit_steps,
                "entropy": torch.stack(ent_trace, dim=1),
                "graph_delta": torch.stack(delta_trace, dim=1),
                "steps_run": t + 1, "block_apps": block_apps}

    # ---- LCGA bookkeeping --------------------------------------------------
    def gamma_values(self) -> torch.Tensor | None:
        """Learned gamma as (B, gamma_steps, h), or None for attn_carry='none'."""
        if self.cfg.attn_carry == "none":
            return None
        return torch.stack([blk.attn.gamma.detach().float().cpu() for blk in self.core])

    def set_record_entropy(self, on: bool = True):
        for blk in self.core:
            blk.attn.record_entropy = on
            blk.attn.last_row_entropy = None

    def param_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        per_block = sum(p.numel() for p in self.core[0].parameters())
        core = per_block * len(self.core)
        T = self.cfg.loop_steps
        lcga = sum(blk.attn.gamma.numel() for blk in self.core
                   if blk.attn.gamma is not None)
        return {
            "total_params": total,
            "params_per_block": per_block,
            "core_params (shared)": core,
            "other_params (embed/norm/head)": total - core,
            "lcga_params": lcga,
            "block_applications": len(self.core) * T,
            "untied_equivalent_params": total + core * (T - 1),
        }


def build_model(**kwargs) -> LoopViT:
    return LoopViT(LoopViTConfig(**kwargs))


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _fmt(n: int) -> str:
    return f"{n:,} ({n / 1e6:.2f}M)"


def variant_name(cfg: LoopViTConfig) -> str:
    return {"none": "LoopViT (vanilla)", "logprob": "LCGA-logprob", "qk": "LCGA-qk",
            "realformer": "LoopViT + RealFormer residual scores (control)"}[cfg.attn_carry]


def print_model_summary(model: LoopViT, batch_size: int = 1, device="cpu"):
    cfg = model.cfg
    line = "=" * 78
    print(line)
    print(f" {variant_name(cfg)} -- LoopViT (arXiv:2602.02156) for classification")
    print(line)
    print(f" image {cfg.image_size}x{cfg.image_size}, patch {cfg.patch_size} -> "
          f"{model.patch_embed.num_patches} patches + {cfg.num_cls_tokens} CLS token(s)")
    print(f" dim {cfg.dim}, heads {cfg.num_heads}, mlp_ratio {cfg.mlp_ratio}, "
          f"classes {cfg.num_classes}, pool '{cfg.pool}'")
    print(f" recurrent core: B={cfg.core_depth} hybrid blocks x T={cfg.loop_steps} "
          f"iterations = {cfg.core_depth * cfg.loop_steps} block applications")
    unrolled = " | ".join(
        f"t={t + 1}: apps {t * cfg.core_depth + 1}-{(t + 1) * cfg.core_depth}"
        for t in range(cfg.loop_steps))
    print(f" unrolled: {unrolled}  (every iteration reuses blocks 1-{cfg.core_depth})")
    print(f" ffn={cfg.ffn}, rope={cfg.rope}, step_embedding={cfg.step_embedding}, "
          f"drop_path={cfg.drop_path}")
    if cfg.attn_carry != "none":
        print(f" LCGA: attn_carry={cfg.attn_carry}, gamma_init={cfg.carry_gamma_init}, "
              f"per_step={cfg.carry_gamma_per_step}, learnable={cfg.carry_gamma_learnable}, "
              f"detach={cfg.carry_detach}, clamp={cfg.carry_clamp}, source={cfg.carry_source}")
    print(f" dynamic exit: mode={cfg.exit_mode}, tau={cfg.exit_tau} nats, "
          f"graph_eps={cfg.exit_graph_eps}, min_steps={cfg.min_loop_steps}, "
          f"max_steps={cfg.max_loop_steps or cfg.loop_steps}")
    print(line)
    for k, v in model.param_report().items():
        print(f" {k:<32} {_fmt(v) if 'params' in k and 'lcga' not in k else v}")
    print(line)

    try:
        from torchinfo import summary
        summary(model,
                input_size=(batch_size, cfg.in_chans, cfg.image_size, cfg.image_size),
                depth=2, device=device,
                col_names=("input_size", "output_size", "num_params"),
                row_settings=("var_names",))
        print(" Note: '(recursive)' rows are the same block run again on a later\n"
              " iteration; its parameters are counted once, its compute every time.")
    except ImportError:
        print(" (pip install torchinfo for the layer-by-layer table)")
        print(model)
    print(line)
