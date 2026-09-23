# Design Doc A — Loop-Carried Graph Attention (LCGA) for LoopViT

> **Paper working title:** *Crystallizing Attention Graphs: Loop-Carried Graph Attention in Looped Vision Transformers*
> **Method name:** LCGA (Loop-Carried Graph Attention)
> **Base code:** `loopvit.py` (LoopViT port of arXiv:2602.02156 for image classification) and `train.py`
> **Audience:** a coding agent (CLI) implementing this, plus the author writing the paper.

---

## 0. Instructions for the coding agent (read first)

1. **Do not change baseline behaviour.** With the new config fields at their defaults (`attn_carry="none"`, `exit_mode="entropy"`), the model must produce **exactly** the same outputs as the current `loopvit.py`, given the same weights. Section 9 includes a test for this.
2. Put all new behaviour behind config fields in `LoopViTConfig`. Don't fork the model class.
3. Keep `forward`, `logits_per_step`, `dynamic_forward`, `param_report` and `print_model_summary` working for every mode.
4. Compute any attention logits you materialise in **fp32** under autocast. Cast back to the value dtype only for the `probs @ v` matmul. The author has hit fp16 overflow NaNs in custom attention before.
5. Work in the phases in Section 8, in order, and run the tests after each phase.
6. Type-hint new functions and give them a docstring that cites the equation number in this doc.

---

## 1. Motivation

LoopViT applies the same `B` blocks `T` times (`z_{t+1} = M_θ(z_t + e_t)`). At present, each iteration builds **every attention graph from scratch**: block `b` at step `t` computes `softmax(QKᵀ/√d)` without seeing the graph that the *same block, with the same weights* produced at step `t−1`.

That wastes structure the model has already worked out. The LoopViT paper describes predictions *crystallizing* over iterations, with predictive entropy falling. LCGA makes the **token graph** crystallize too. Each block's attention graph at step `t` gets the graph from step `t−1` as a prior, so the loop refines one graph over time instead of rebuilding it.

Why this suits LoopViT specifically, and not a standard ViT:

- In a standard ViT, layer `l` and layer `l+1` have **different** weights and roles, so carrying scores across layers (RealFormer-style) mixes graphs that mean different things.
- In LoopViT, block `b` at step `t` and block `b` at step `t+1` are **the same function**. Carrying the graph from block `b` to itself is semantically consistent. It turns the loop into an iterative estimate of one graph per block.
- It costs almost no parameters: `B × h` scalars (24 for the default config).
- It gives a new, parameter-free **stop signal**: halt when the graph stops changing (Section 5).

---

## 2. Formulation

Notation: `N` batch, `h` heads, `T_tok = P + n_prefix` tokens (197 for 224/16 with 1 CLS), `d` head dim, `t ∈ {0..T−1}` loop step, `b ∈ {0..B−1}` block index.

The current (vanilla) attention of block `b` at step `t` is:

```
L_t^b = Q_t^b (K_t^b)ᵀ / √d                                  (A1)
A_t^b = softmax(L_t^b)
```

### 2.1 Mode `logprob` (primary, exact graph prior)

```
S_t^b   = log A_t^b                         (the carried edge state, a log-prob graph)
L̃_t^b  = Q_t^b (K_t^b)ᵀ / √d + γ_{b,h}(t) · clamp(S_{t−1}^b, min=c)      (A2)
A_t^b   = softmax(L̃_t^b),     S_0^b uses no prior
```

Interpretation: `A_t ∝ exp(QKᵀ/√d) · (A_{t−1})^γ`. The previous graph is a **tempered prior** and the current query–key compatibility is the **likelihood**. This is a Bayesian-style posterior update over edges. Because `S_t` itself contains the earlier prior, the memory is **recursive**: step `t` depends on every earlier step, weighted geometrically by γ.

- `c` (`carry_clamp`, default −10) puts a floor on the prior. Without it, an edge that reached ~0 probability could never come back (`log 0 = −∞`).
- Adding a row-constant to logits doesn't change softmax, so the prior needs no centering.
- γ is **unconstrained**. γ > 0 means persistence and γ < 0 means anti-persistence (exploration). Report the learned sign as a finding.

### 2.2 Mode `qk` (flash-friendly, Markov-1 carry of raw logits)

```
L̃_t^b = [ Q_t^b (K_t^b)ᵀ + γ_{b,h}(t) · Q_{t−1}^b (K_{t−1}^b)ᵀ ] / √d    (A3)
```

This never materialises the `T_tok×T_tok` matrix. Concatenating along the head dimension gives exactly the sum of the two dot products:

```
q_cat = cat([q, γ·q_prev], -1)      # (N,h,T_tok,2d)
k_cat = cat([k, k_prev],   -1)      # (N,h,T_tok,2d)
out   = SDPA(q_cat, k_cat, v, scale=d**-0.5)   # v keeps head dim d
```

- `F.scaled_dot_product_attention` allows `v` to have a different last dim from `q`/`k`. Pass `scale` explicitly (torch ≥ 2.1). Otherwise SDPA scales by `1/√(2d)`, which is wrong here.
- The flash kernel may not accept `E_qk ≠ E_v`. SDPA then falls back to the memory-efficient kernel, which is fine.
- The state is `(q_t, k_t)` **after RoPE**. The memory is **one step only**: raw logits from `t−1`, not accumulated. This is a real semantic difference from `logprob` and an ablation in its own right.

### 2.3 Per-step γ (optional)

`carry_gamma_per_step=False` (default) gives one γ per (block, head): `gamma ∈ R^{h}` inside each block's attention module.
`carry_gamma_per_step=True` gives `gamma ∈ R^{T_train × h}`, indexed with `min(t, T_train−1)`. This is the same identity extrapolation that `_step_vector` uses, so a model trained at T=3 still runs at T=6.
In both cases γ is **initialised to `carry_gamma_init` (default 0.0)**, so at initialisation the model is exactly vanilla LoopViT.

### 2.4 Gradient through the carry

`carry_detach=False` (default) lets gradients flow into the previous step's graph. That is full BPTT over the graph, which is what makes γ learn a useful prior.
`carry_detach=True` applies stop-gradient to the state. It saves memory and is an ablation ("does the model need to learn *to produce* graphs that are good priors?").

---

## 3. Config additions (`LoopViTConfig`)

```python
# --- LCGA: loop-carried graph attention ---
attn_carry: str = "none"          # "none" | "logprob" | "qk"
carry_gamma_init: float = 0.0     # gamma init; 0 => identical to vanilla at init
carry_gamma_per_step: bool = False
carry_detach: bool = False
carry_clamp: float = -10.0        # floor on log A_{t-1} in logprob mode

# --- dynamic exit criteria (extends existing exit_tau etc.) ---
exit_mode: str = "entropy"        # "entropy" | "graph" | "both" | "either"
exit_graph_eps: float = 0.01      # nats; CLS-row KL(A_t || A_{t-1}) threshold
```

Validation in `LoopViT.__init__`:
- `attn_carry in {"none","logprob","qk"}`.
- `exit_mode in {"entropy","graph","both","either"}`. Any mode that uses `graph` requires `num_cls_tokens >= 1` (Section 5 uses the CLS row).
- `attn_carry == "none"` with `exit_mode` using `graph` is **allowed**: vanilla attention also changes its graph between steps, and that is a useful baseline for the exit criterion.

---

## 4. Code changes

### 4.1 Attention module

Replace `Attention` with a stateful version, keeping the old one as the `attn_carry="none"` path or folding it in. New signature:

```python
@dataclass
class AttnOut:
    out: torch.Tensor                  # (N, T_tok, D)
    state: object | None               # logprob: Tensor (N,h,T_tok,T_tok) fp32 log-probs
                                       # qk: tuple(q, k), each (N,h,T_tok,d), post-RoPE
                                       # none: None
    cls_probs: torch.Tensor | None     # (N, h, T_tok) attention distribution of CLS query 0, fp32

class Attention(nn.Module):
    def __init__(self, dim, num_heads, grid, n_prefix, qkv_bias=True, attn_drop=0.0,
                 proj_drop=0.0, rope=True, carry="none", gamma_init=0.0,
                 gamma_steps=1, detach=False, clamp=-10.0): ...

    def forward(self, x, state=None, t: int = 0, need_cls: bool = False) -> AttnOut: ...
```

Parameters: `self.gamma = nn.Parameter(torch.full((gamma_steps, num_heads), gamma_init))`. Use `gamma[min(t, gamma_steps-1)].view(1,h,1,1)`.

**Forward, `none` mode:** unchanged SDPA path. If `need_cls`, also compute
`cls_probs = softmax(q[:,:,:1].float() @ k.float().transpose(-2,-1) * scale).squeeze(2)`. This costs `O(T_tok)` per head.

**Forward, `logprob` mode:**
```python
with torch.autocast(device_type=x.device.type, enabled=False):
    logits = (q.float() @ k.float().transpose(-2, -1)) * scale        # (N,h,T,T)
    if state is not None:
        logits = logits + g * state.clamp_min(self.clamp)            # eq. A2
    logA = logits.log_softmax(-1)
probs = logA.exp()
probs_d = F.dropout(probs, self.attn_drop, self.training)            # dropout AFTER carry capture
out = (probs_d.to(v.dtype) @ v)
new_state = logA.detach() if self.detach else logA
cls_probs = probs[:, :, 0, :] if need_cls else None
```

**Forward, `qk` mode:**
```python
if state is None:
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=..., scale=scale)
else:
    qp, kp = state
    q_cat = torch.cat([q, g.to(q.dtype) * qp], dim=-1)
    k_cat = torch.cat([k, kp], dim=-1)
    out = F.scaled_dot_product_attention(q_cat, k_cat, v, dropout_p=..., scale=scale)
new_state = (q.detach(), k.detach()) if self.detach else (q, k)
if need_cls:   # small explicit computation of the CLS row, eq. A3 restricted to query 0
    lc = q[:, :, :1].float() @ k.float().transpose(-2, -1)
    if state is not None:
        lc = lc + g * (qp[:, :, :1].float() @ kp.float().transpose(-2, -1))
    cls_probs = (lc * scale).softmax(-1).squeeze(2)
```

### 4.2 HybridBlock

```python
def forward(self, x, dp: float = 0.0, state=None, t: int = 0, need_cls: bool = False):
    a = self.attn(self.norm1(x), state=state, t=t, need_cls=need_cls)
    x = x + drop_path(a.out, dp, self.training)
    x = x + drop_path(self.ffn(self.norm2(x)), dp, self.training)
    return x, a.state, a.cls_probs
```
When stochastic depth drops the attention branch for a sample, that sample's graph state is **still** computed and carried. This is intentional: the graph stays well-defined.

### 4.3 LoopViT: thread the state through the loop

```python
def core_step(self, h, t, rates=None, states=None, need_cls=False):
    """z_{t+1} = M_theta(z_t + e_t), carrying one graph state per block (eq. A2/A3)."""
    e = self._step_vector(t)
    if e is not None:
        h = h + e
    B = len(self.core)
    states = states if states is not None else [None] * B
    new_states, cls = [], []
    for b, blk in enumerate(self.core):
        h, s, c = blk(h, rates[t * B + b] if rates else 0.0, state=states[b], t=t, need_cls=need_cls)
        new_states.append(s); cls.append(c)
    return h, new_states, (torch.stack(cls, 1) if need_cls else None)   # cls: (N,B,h,T_tok)
```

`loop()` keeps `states` across iterations. Add `return_graphs: bool` so analysis code can collect the per-step `cls_probs`. `forward` and `logits_per_step` go through `loop()` and need no API change.

### 4.4 `param_report`

Add `"lcga_params": B*h*gamma_steps`. The untied-equivalent calculation is unchanged.

---

## 5. Dynamic exit with a graph-stability criterion

### 5.1 Graph change signal

For each sample, after step `t ≥ 1`:

```
Δ_t = mean_{b,h}  KL( p_t^{b,h} ‖ p_{t−1}^{b,h} ),   p = CLS-query attention row (N,h,T_tok)   (A4)
```

- Use the CLS row only. It is cheap in every mode (no full `T_tok²` needed in `qk`/`none`), and it is the row that feeds the classifier.
- Compute in fp32 with `eps=1e-8` inside the logs.
- `Δ_0` is undefined, so a graph criterion can't fire at `t=0`.

### 5.2 Exit rule

```
ent_ok   = H_t < tau
graph_ok = (t >= 1) & (Δ_t < exit_graph_eps)
exit_mode: entropy -> ent_ok | graph -> graph_ok | both -> ent_ok & graph_ok | either -> ent_ok | graph_ok
```
and, as now, only once `t+1 >= min_steps`.

### 5.3 Fix the known compute waste: compact the active batch

The current `dynamic_forward` still runs `core_step` on samples that have already exited and then discards the result with `torch.where`. Change it to:

```python
active = (~done).nonzero(as_tuple=True)[0]
h_a = h[active]
st_a = [gather_state(s, active) for s in states]          # index batch dim of tensor or (q,k) tuple
h_new, st_new, cls_new = self.core_step(h_a, t, states=st_a, need_cls=True)
h[active] = h_new                                         # scatter back (h cloned beforehand)
states = [scatter_state(s_full, s_new, active) for ...]
```

Keep full-batch buffers for `h`, states and the previous CLS row. A frozen sample keeps its last `h` and its last graph (`z_k = z_t` and `S_k = S_t`).
Also return `"graph_delta"` `(N, steps_run)` (NaN at t=0) and `"block_apps"` (the total block applications actually executed, summed over samples). That total is the real compute metric for the paper.

---

## 6. Training-script changes (`train.py`)

CLI flags that mirror the config: `--attn-carry`, `--carry-gamma-init`, `--carry-gamma-per-step`, `--carry-detach`, `--carry-clamp`, `--exit-mode`, `--exit-graph-eps`.

Logging, once per epoch on the validation set:
- accuracy at T = 1..T_train (from `logits_per_step`) and at T = T_train+1..2·T_train (extrapolation);
- γ values per block and head (a heatmap image, plus raw values in JSON);
- mean Δ_t per step (Section 5.1) for the validation set;
- predictive entropy H_t per step;
- peak GPU memory and throughput (img/s), because `logprob` materialises `N×h×T²` per block.

Memory estimate for `logprob` (224/16, fp32, h=6): one `(h,197,197)` map ≈ 0.93 MB per sample. At batch 128 that's ≈ 119 MB per block-step, with about 3 saved tensors per block-step for autograd over B·T = 12 block-steps, so **≈ 4–5 GB extra**. If that is too much, use `qk` mode, `carry_detach=True`, or a smaller batch with gradient accumulation.

---

## 7. Analysis script (`analyze_lcga.py`)

It takes a checkpoint and a data split and writes figures and JSON:

1. **Graph crystallization curve:** Δ_t against t for T = 1..2·T_train, comparing vanilla and LCGA (mean ± std across images).
2. **CLS attention maps over steps:** reshape `cls_probs[..., n_prefix:]` to `g×g`, average over heads, one row per step, overlaid on the image. Pick 8 fixed validation images.
3. **Per-head attention entropy per step:** does LCGA sharpen the graph?
4. **Oversmoothing diagnostics per step** (patch tokens only, after each full step):
   - Dirichlet energy on the 4-neighbour patch grid: `E = mean_{(i,j)} ‖x̂_i − x̂_j‖²` with `x̂ = x/‖x‖`.
   - Effective rank of the centred token matrix: `exp(H(σ/Σσ))` from its singular values.
5. **γ heatmap** (blocks × heads, and × steps when per-step).
6. **Exit Pareto:** accuracy against mean block applications, sweeping `tau` and `exit_graph_eps` for each exit mode.

---

## 8. Implementation phases (for the CLI)

| Phase | Work | Done when |
|---|---|---|
| 0 | Refactor so `core_step`/`loop` thread a per-block state (always `None` for now); `HybridBlock` returns a tuple | Test T1 passes (output bit-identical to the original) |
| 1 | `logprob` mode + γ params + clamp + detach | T2, T3, T5 pass |
| 2 | `qk` mode via the concatenation trick | T4 passes |
| 3 | `need_cls` / `cls_probs`, Δ_t, `exit_mode`, batch compaction in `dynamic_forward` | T6, T7, T8 pass |
| 4 | `train.py` flags + logging | one short training run per mode on a small dataset works end to end |
| 5 | `analyze_lcga.py` | produces all six outputs listed in Section 7 |
| 6 (optional) | Single-file export `kaggle_lcga.py` that inlines model + train + analysis for one-cell Kaggle runs | runs top to bottom in a fresh notebook |

---

## 9. Tests (`tests/test_lcga.py`, pytest, CPU, tiny config)

Tiny config: `image_size=32, patch_size=8, dim=64, num_heads=4, core_depth=2, loop_steps=3, num_classes=5`.

- **T1 baseline equivalence:** build the original model and the new model with `attn_carry="none"`. Copy the state dict (`strict=True`) and check `torch.allclose(out_old, out_new, atol=0)` in eval mode.
- **T2 γ=0 equivalence:** new model with `attn_carry="logprob"`, `carry_gamma_init=0`. Load the vanilla weights with `strict=False`, since the only missing keys are γ. Outputs match vanilla to `atol=1e-5` (the fp32 path vs SDPA gives tiny differences). Adding parameters shifts the init RNG order, which is why the test loads state dicts rather than relying on seeds.
- **T3 prior is used:** set γ=1 and check that the step-2 attention of block 0 equals `softmax(L + clamp(logA_prev))` computed by hand.
- **T4 `qk` correctness:** compare the concatenation-trick output with an explicit dense computation of eq. A3 (`atol=1e-5`).
- **T5 gradients:** after `loss.backward()`, `gamma.grad` is non-zero whenever γ≠0 or the prior is non-trivial. With `carry_detach=True`, gradients still reach γ, but not the previous step's q/k through the state.
- **T6 extrapolation:** `model(x, num_steps=2*loop_steps)` runs, including with `carry_gamma_per_step=True`.
- **T7 exit equivalence:** `dynamic_forward` with `tau=-1`, `exit_graph_eps=-1` (never exit) returns logits equal to `forward(x, num_steps=max_steps)` in eval mode.
- **T8 compaction:** give a batch where half the samples exit at step 1 (force it by hand with a mock or large `tau` on selected samples). Check that the compacted run matches a reference loop that runs every sample separately, and that `block_apps` is correct.
- **T9 fp16/bf16:** forward under autocast (CUDA if available, otherwise bf16 on CPU) gives no NaN or Inf in logits or states.

---

## 10. Experiments for the paper

### 10.1 Configs
- **Small (fast iteration, Kaggle-friendly):** `image_size=32, patch_size=4` (64 patches), `dim=192, heads=6` (head_dim 32, divisible by 4 for 2D RoPE), `B=4, T=3`. CIFAR-10, CIFAR-100.
- **Base:** the default 224/16 config, `dim=384, heads=6, B=4, T=3`. Tiny-ImageNet (upsampled) plus the author's plant datasets (PlantVillage, PlantDoc, FGVC Plant Pathology) for continuity with earlier work.
- Train from scratch, same augmentation and schedule for every variant. 3 seeds, report mean ± std, and use paired tests across seeds or per-image McNemar where appropriate.

### 10.2 Main comparison (equal B, T, and parameters to within 24–72)
| Model | Notes |
|---|---|
| LoopViT vanilla | baseline |
| LoopViT + RealFormer-style residual scores **across blocks within a step** | control: is it just "residual attention"? |
| LCGA-qk | Markov-1 raw-logit carry |
| LCGA-logprob | recursive posterior carry (main) |
| Untied ViT with B·T layers | reference for a parameter/compute trade-off |

### 10.3 Ablations
- `carry_detach` on/off; γ per-step vs shared; γ fixed at 1 (not learned) vs learned; `carry_clamp` ∈ {−5, −10, −20}.
- **Carry source:** same block `b→b` (proposed) vs **shifted** `b→b+1` (breaks the "same function" argument). If same-block wins, that supports the core claim.
- Exit criteria: entropy / graph / both / either, plotted on the Pareto curve.

### 10.4 Hypotheses (state them before running)
- **H1:** LCGA-logprob ≥ vanilla at T_train with ~0 extra parameters.
- **H2:** Under LCGA, Δ_t decreases monotonically with t, and accuracy degrades less than vanilla when extrapolating to T > T_train.
- **H3:** On the accuracy vs block-applications curve, the `both` exit dominates the `entropy` exit. Graph stability should catch confidently wrong early predictions.
- **H4 (open):** LCGA either reduces or increases oversmoothing (a sticky graph could accelerate collapse). Report whichever happens.

---

## 11. Related work to cite and differentiate (verify each reference before citing)

- **RealFormer** (residual attention scores across consecutive layers) and **Evolving Attention / EA-Transformer** (convolutions over the previous layer's attention maps). These are the closest work. Differentiation: they carry across *different* layers, while LCGA carries across *iterations of the same weight-tied block*, has a normalised log-prob prior with a recursive posterior reading, and uses graph stability as a halting signal.
- **Universal Transformers, ACT, PonderNet:** looped/recurrent depth and learned halting.
- **LoopViT (arXiv:2602.02156):** base architecture, entropy-based Dynamic Exit.
- The author's own **IMHA** (entropy-guided iterative attention): position LCGA as the graph-level counterpart.

## 12. Claims to avoid (reviewer-proofing)

- Say **"graph"** or **"attention graph"**, not "topology", unless you actually measure a topological invariant.
- Don't claim efficiency for `logprob` mode, which costs *more* memory. The efficiency claim belongs to the exit mechanism (`block_apps`), and only if the Pareto plot supports it.
- "Bayesian" is an **interpretation** of eq. A2 (prior × likelihood). Don't claim it is exact inference.
