# LoGA / LCGA — Loop-Carried Graph Attention for Looped Vision Transformers

> *Crystallizing Attention Graphs: Loop-Carried Graph Attention in Looped Vision Transformers*

This repo implements **LCGA (Loop-Carried Graph Attention)** on top of **LoopViT**
(Shu et al., [arXiv:2602.02156](https://arxiv.org/abs/2602.02156)), ported to image
classification. The full design is in
[`design_A_loop_carried_graph_attention.md`](design_A_loop_carried_graph_attention.md).

LoopViT applies the same `B` blocks `T` times (`z_{t+1} = M_θ(z_t + e_t)`). In vanilla
LoopViT, block `b` rebuilds its attention graph from scratch at every loop step. It never
sees the graph that the same block, with the same weights, produced one step earlier. LCGA
passes that previous graph forward as a prior, so the loop refines one attention graph per
block instead of rebuilding it each step:

```
logprob (main, eq. A2):  L̃_t = Q_t K_tᵀ/√d + γ_{b,h}(t) · clamp(log A_{t−1}, min=c)
                         A_t = softmax(L̃_t)            (A_t ∝ likelihood × A_{t−1}^γ)
qk      (eq. A3):        L̃_t = [Q_t K_tᵀ + γ_{b,h}(t) · Q_{t−1} K_{t−1}ᵀ] / √d
                         (flash-friendly: SDPA on [q, γ·q_prev] and [k, k_prev])
```

* γ is initialised to 0, so at initialisation LCGA **is** vanilla LoopViT.
* It adds only `B × h` parameters (24 for the default config).
* It adds a new stop signal that needs no extra parameters. **Graph-stability exit:** a
  sample halts once its CLS attention graph stops changing,
  `Δ_t = mean_{b,h} KL(p_t ‖ p_{t−1}) < ε` (eq. A4). This can be combined with the paper's
  entropy exit.

---

## 1. Files

| file | what |
|---|---|
| `loop_vit.py` | the model: `LoopViT` + `LoopViTConfig` with the LCGA modes, graph exit and batch-compacting `dynamic_forward` |
| `train.py` | training: YAML + CLI, variants, early stopping, checkpoint every epoch, resume (also mid-epoch), per-epoch LCGA logging |
| `analyze_lcga.py` | analysis figures + JSON (graph crystallization, CLS maps, attention entropy, oversmoothing, γ heatmap, exit Pareto) |
| `predict.py` | run a checkpoint on images, with loop-count override, dynamic exit and per-step graph change |
| `data.py` | dataset lookup (`datasets/` and `dataset/`), ImageFolder pipeline, resumable sampler |
| `downloads.py` | fetch aircraft / cub200 / flowers102 / food101 / cars into the `datasets/` layout |
| `config.yaml` | every setting in one place |
| `pretrained.py` | build the loop core from a pretrained timm ViT (`--pretrained`), or copy weights from a repo checkpoint (`--init-from`) |
| `finetune.py` + `finetune.yaml` | paper experiments: pretrained ViT → vanilla LoopViT vs LCGA on cub200 / aircraft / flowers102 / food101 |
| `tests/test_lcga.py` | tests T1–T9 from the design doc (+ extras) |
| `tests/loop_vit_reference.py` | the original, unmodified LoopViT, used to prove baseline equivalence |

## 2. Variants

Select with `--variant` (a preset) or with `--attn-carry` directly.

| `--variant` | `attn_carry` | what it is |
|---|---|---|
| `lcga` / `lcga-logprob` | `logprob` | **LCGA, main method.** Recursive log-prob graph prior (eq. A2). Default in `config.yaml` |
| `lcga-qk` | `qk` | LCGA, Markov-1 carry of raw logits (eq. A3). Never materialises `T×T`, so it uses less memory |
| `loopvit` | `none` | vanilla LoopViT baseline, bit-identical to the original code (test T1) |
| `realformer` | `realformer` | control: RealFormer-style residual scores **across blocks within one step** |

Ablation switches (design doc §10.3):

| flag | values | ablation |
|---|---|---|
| `--carry-detach` | `false` / `true` | stop-gradient through the carried graph (also saves memory) |
| `--carry-gamma-per-step` | `false` / `true` | one γ per (block, head) vs per (step, block, head) |
| `--carry-gamma-learnable false --carry-gamma-init 1` | | γ fixed at 1 |
| `--carry-clamp` | `-5`, `-10`, `-20` | floor on `log A_{t−1}` |
| `--carry-source` | `same` / `shifted` | carry block `b→b` (proposed) vs `b→b+1` |
| `--exit-mode` | `entropy` / `graph` / `both` / `either` | dynamic-exit criterion |

## 3. Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.14, torch 2.14 + CUDA 12.6 on an RTX 3060 (Windows 11). On Ampere
cards, AMP uses bf16.

## 4. Datasets

Copy the `dataset/` and `datasets/` folders into this repo's root. The loader searches both:

```
datasets/aircraft/{train,test}/<class>/*.jpg
datasets/cub200/{train,test}/<class>/*.jpg
datasets/flowers102/{train,test}/<class>/*.jpg
datasets/food101/{train,test}/<class>/*.jpg
datasets/plantodc/{train,test}/<class>/*.jpg          # PlantDoc
dataset/plantvillage/color/<class>/*.jpg              # flat: 10 % held out for validation
```

* `--dataset <name>` is looked up under `--data-root` (default `datasets`), then `datasets/`,
  then `dataset/`.
* `train/` + `test/` (or `val/`, `valid/`) are found automatically.
* A folder that only wraps the class folders in one sub-folder, such as
  `plantvillage/color/`, is descended into automatically. `--dataset plantvillage/color` also works.
* A flat folder (class folders directly inside it) gets a stratified `--val-split` hold-out.
* `--train-dir` / `--val-dir` accept any path on disk and skip the name lookup.
* On Windows, image paths longer than the 260-character `MAX_PATH` limit (common for
  PlantVillage inside a deeply nested folder) are opened through the `\\?\` long-path
  prefix, so you don't need to enable `LongPathsEnabled` or move the repo.

Fetch the public benchmarks again if needed:

```bash
python downloads.py --dataset cub200 flowers102 aircraft food101
```

## 5. Commands

All commands below run from the repo root. On Windows `cmd`, use `^` for line continuation
instead of `\` (PowerShell uses a backtick), or put each command on one line.

### 5.1 Tests (CPU, a few seconds)

```bash
python -m pytest tests -q
```

### 5.2 Model summary (no data needed)

```bash
python train.py --config config.yaml --summary-only --num-classes 38
python train.py --config config.yaml --summary-only --num-classes 38 --variant loopvit
```

### 5.3 Train the new variant (LCGA-logprob) on each dataset

```bash
python train.py --config config.yaml --dataset plantodc
python train.py --config config.yaml --dataset plantvillage
python train.py --config config.yaml --dataset cub200
python train.py --config config.yaml --dataset flowers102
python train.py --config config.yaml --dataset aircraft
python train.py --config config.yaml --dataset food101
```

Each run writes to `runs/<variant>_<dataset>/`, for example `runs/lcga_plantodc/`.

### 5.4 Main comparison (same dataset, every variant)

```bash
python train.py --config config.yaml --dataset plantodc --variant loopvit
python train.py --config config.yaml --dataset plantodc --variant lcga
python train.py --config config.yaml --dataset plantodc --variant lcga-qk
python train.py --config config.yaml --dataset plantodc --variant realformer
```

### 5.5 Ablations (give each its own `--output-dir`)

```bash
python train.py --config config.yaml --dataset plantodc --carry-detach true           --output-dir runs/abl_detach
python train.py --config config.yaml --dataset plantodc --carry-gamma-per-step true   --output-dir runs/abl_gamma_per_step
python train.py --config config.yaml --dataset plantodc --carry-gamma-learnable false --carry-gamma-init 1 --output-dir runs/abl_gamma_fixed1
python train.py --config config.yaml --dataset plantodc --carry-clamp -5               --output-dir runs/abl_clamp5
python train.py --config config.yaml --dataset plantodc --carry-clamp -20              --output-dir runs/abl_clamp20
python train.py --config config.yaml --dataset plantodc --carry-source shifted        --output-dir runs/abl_shifted
python train.py --config config.yaml --dataset plantodc --seed 1 --output-dir runs/lcga_plantodc_s1   # seeds
```

### 5.6 Small / fast config (CIFAR-like, design doc §10.1)

```bash
python train.py --config config.yaml --dataset plantodc --image-size 32 --patch-size 4 --dim 192 --num-heads 6 --core-depth 4 --loop-steps 3
```

### 5.7 Subsets, epochs, early stopping

```bash
python train.py --config config.yaml --dataset plantvillage --num-classes 10 --max-per-class 300
python train.py --config config.yaml --dataset plantvillage --classes Apple___Apple_scab Apple___Black_rot Apple___healthy
python train.py --config config.yaml --dataset cub200 --epochs 200 --early-stopping true --patience 20 --min-delta 0.001 --monitor val_acc
python train.py --config config.yaml --dataset cub200 --early-stopping false
```

### 5.8 Checkpoints and resume

Every run saves:

| file | when |
|---|---|
| `last.pt` | every epoch; every `--save-every-steps N` optimizer steps; and on Ctrl+C |
| `best.pt` | whenever the early-stopping monitor improves |
| `checkpoints/epoch_XXX.pt` | every `--save-every` epochs (default 1 = **every epoch**). `--keep-checkpoints N` keeps only the newest N |

Each checkpoint stores the model, optimizer, AMP scaler, RNG states, LR-schedule step,
early-stopping counter and the position inside the epoch. Writes are atomic, so killing
the process mid-save never corrupts `last.pt`.

```bash
# resume where you stopped: `resume: auto` is the default, so re-run the SAME command
python train.py --config config.yaml --dataset plantodc

# resume from a specific checkpoint
python train.py --config config.yaml --dataset plantodc --resume runs/lcga_plantodc/checkpoints/epoch_020.pt
python train.py --config config.yaml --dataset plantodc --resume runs/lcga_plantodc/best.pt

# also save inside long epochs (e.g. food101), so a crash loses at most 500 steps
python train.py --config config.yaml --dataset food101 --save-every-steps 500

# train longer than originally planned: raise --epochs and re-run
python train.py --config config.yaml --dataset plantodc --epochs 150

# ignore an existing last.pt and start fresh (overwrites the run folder's logs)
python train.py --config config.yaml --dataset plantodc --resume none
```

A mid-epoch resume skips exactly the batches already seen, because the shuffle order of
epoch *e* depends only on `(seed, e)`. On resume, rows in `log.csv` / `metrics.jsonl` from
after the checkpoint are dropped. A run that already early-stopped refuses to continue
unless you raise `--patience` or pass `--early-stopping false`. Resume also checks that the
classes and model architecture match the checkpoint. Inference-only knobs (`exit_*`,
`min/max_loop_steps`) may change freely.

### 5.9 Predict

```bash
python predict.py --ckpt runs/lcga_plantodc/best.pt --images some_folder/
python predict.py --ckpt runs/lcga_plantodc/best.pt --images a.jpg --per-step           # prediction, entropy, graph Δ per step
python predict.py --ckpt runs/lcga_plantodc/best.pt --images a.jpg --loop-steps 6       # extrapolate past T
python predict.py --ckpt runs/lcga_plantodc/best.pt --images some_folder/ --dynamic-exit --exit-mode entropy --exit-tau 0.05
python predict.py --ckpt runs/lcga_plantodc/best.pt --images some_folder/ --dynamic-exit --exit-mode both --exit-tau 0.1 --exit-graph-eps 0.005 --loop-steps 6
```

### 5.10 Analysis (design doc §7)

```bash
# compare vanilla vs LCGA on the same validation split
python analyze_lcga.py --ckpt runs/loopvit_plantodc/best.pt runs/lcga_plantodc/best.pt runs/lcga-qk_plantodc/best.pt \
                       --labels vanilla lcga-logprob lcga-qk --out-dir analysis/plantodc

# a single run, more images, custom exit sweep
python analyze_lcga.py --ckpt runs/lcga_cub200/best.pt --max-images 3000 --taus 0.01 0.05 0.1 0.3 --graph-eps 1e-4 1e-3 1e-2
```

Outputs:

| file | content |
|---|---|
| `crystallization.png` | Δ_t vs t = 1..2T (mean ± std), all checkpoints |
| `accuracy_per_step.png` | accuracy vs t, including extrapolation past T_train |
| `cls_maps_<label>.png` | CLS attention (last block, head mean) per step, overlaid on 8 fixed images |
| `attn_entropy_<label>.png` | per-head attention entropy per step (does LCGA sharpen the graph?) |
| `oversmoothing.png` | Dirichlet energy and effective rank of patch tokens per step |
| `gamma_<label>.png` | learned γ heatmap (blocks × heads [× steps]) |
| `exit_pareto.png` | accuracy vs mean block applications for entropy / graph / both / either exits |
| `summary.json` | every number behind the figures |

### 5.11 Replication: PlantVillage runs

The exact command used for the PlantVillage baseline (vanilla LoopViT). It sets 100 epochs
and batch size 16, saves a checkpoint for every epoch and keeps them all, early-stops when
val loss hasn't improved for 10 epochs, and resumes automatically if you re-run it:

```bash
python train.py --config config.yaml --dataset plantvillage --variant loopvit --epochs 100 --batch-size 16 --save-every 1 --keep-checkpoints 0 --early-stopping true --monitor val_loss --patience 10 --resume auto --output-dir runs/loopvit_plantvillage
```

For a fair comparison, run the other variants with the same settings. Only `--variant` and
`--output-dir` change:

```bash
python train.py --config config.yaml --dataset plantvillage --variant lcga       --epochs 100 --batch-size 16 --save-every 1 --keep-checkpoints 0 --early-stopping true --monitor val_loss --patience 10 --resume auto --output-dir runs/lcga_plantvillage
python train.py --config config.yaml --dataset plantvillage --variant lcga-qk    --epochs 100 --batch-size 16 --save-every 1 --keep-checkpoints 0 --early-stopping true --monitor val_loss --patience 10 --resume auto --output-dir runs/lcga-qk_plantvillage
python train.py --config config.yaml --dataset plantvillage --variant realformer --epochs 100 --batch-size 16 --save-every 1 --keep-checkpoints 0 --early-stopping true --monitor val_loss --patience 10 --resume auto --output-dir runs/realformer_plantvillage
```

All other settings come from `config.yaml` (seed 42, lr 5e-4, 10 % stratified val split,
T = 3, B = 4). `runs/<name>/run_history.log` records the exact command, time, git commit
and resume point of every invocation, and `config.json` stores the full resolved settings.

### 5.12 Paper experiments: fine-tuning from a pretrained ViT

Training from scratch cannot get close to published results on small fine-grained datasets,
and FGVC attention papers don't do that either. They fine-tune from ImageNet-pretrained
weights. `--pretrained <timm ViT>` builds the loop core from the ViT's blocks
(`--pretrained-blocks stepwise` = layers 0, 3, 6, 9 for 12 → 4) and sets the architecture to
match it (LayerNorm, GELU MLP, no RoPE, LayerScale for DINOv2). The step embeddings are set
to zero and γ starts at 0, so **vanilla LoopViT and LCGA start from the same function**, and
any gap between them comes from LCGA.

The paper compares two variants: `loopvit` (baseline) and `lcga` (LCGA-logprob, proposed).
`finetune.py` runs both on every dataset with the recipe in `finetune.yaml` (DeiT-S IN-1k,
AdamW lr 5e-5 with ×20 for head/γ/step embeddings, 2 warm-up epochs, TrivialAugment,
batch 16, early stopping on val acc). Per-dataset epochs are set in `RECIPES`.

```bash
python downloads.py --dataset cub200 aircraft flowers102 food101
python finetune.py                                                    # 4 datasets x {loopvit, lcga}
python finetune.py --datasets cub200 aircraft --dry-run               # only print the commands
python finetune.py --pretrained vit_small_patch14_dinov2.lvd142m      # stronger backbone
python finetune.py --datasets cub200 --image-size 448 --batch-size 8  # unknown flags go to train.py
```

Runs go to `runs/<variant>_<dataset>_<backbone>/`, and re-running continues unfinished ones.
`--core-depth 12 --loop-steps 1` gives the plain pretrained ViT as a reference number, and
`--init-from <ckpt>` fine-tunes from one of your own checkpoints (e.g. PlantVillage →
PlantDoc).

## 6. What is logged every epoch

```
epoch  12/100 | lr 4.31e-04 | train loss 1.8123 acc 0.4712 | 212 img/s, 5.84 GB
    val loss 1.6002 acc 0.5340 | acc per step [0.402 0.498 0.534] extrap [0.538 0.537 0.531]
    entropy/step [1.912 1.405 1.101 1.050 1.041 1.040] | graph delta/step [nan 0.0412 0.0087 0.0021 0.0009 0.0005]
    exit(entropy) acc 0.5310 @ 2.41 steps, 9.6 block apps/img
    gamma mean +0.1834 |max| 0.6120 | 41.3s
```

* **acc per step / extrap:** validation accuracy after each step 1..T, then T+1..2T. The
  extra steps reuse the last step embedding (identity extrapolation).
* **entropy/step:** mean predictive entropy per step.
* **graph delta/step:** mean Δ_t of the CLS attention graph (eq. A4). Hypothesis H2 says it
  should fall monotonically under LCGA.
* **exit(...):** dynamic-exit accuracy, average steps, and the block applications actually
  executed. The exit compacts the batch, so exited samples really stop costing compute.
* **gamma:** learned γ. The sign is a finding: γ > 0 means persistence, γ < 0 means exploration.
* **img/s, GB:** training throughput and peak GPU memory. `logprob` stores `N·h·T_tok²` per
  block-step, so it uses more memory than `qk` / vanilla.

Files in `runs/<name>/`: `log.csv`, `metrics.jsonl` (full per-epoch record including raw γ),
`gamma_heatmap.png` (latest epoch), `config.json`, `last.pt`, `best.pt`, `checkpoints/`,
and `run_history.log`. That last file is only ever appended to. For every invocation,
including each resume, it records the timestamp, exact command, working directory, git
commit, Python and torch versions, the resume point, and how the run ended (`END`,
`INTERRUPTED`, `FAILED` or `CRASHED`).

During an epoch a live `tqdm` bar shows batch progress, time left, and the running loss,
accuracy and lr. A second bar shows the validation pass. Turn both off with `--progress false`.

## 7. Configuration reference

Every key lives in `config.yaml` and can be overridden on the command line. YAML keys use
underscores, flags use dashes.

| group | keys (defaults) |
|---|---|
| data | `dataset`, `data_root: datasets`, `train_dir`, `val_dir`, `num_classes`, `class_selection: first`, `classes`, `max_per_class`, `val_split: 0.1`, `augment: basic` (`none`/`basic`/`trivial`), `num_workers: 4` |
| model | `image_size: 224`, `patch_size: 16`, `dim: 384`, `core_depth: 4` (B), `loop_steps: 3` (T), `num_heads: 6`, `mlp_ratio: 4.0`, `dropout`, `attn_dropout`, `drop_path: 0.1`, `ffn: hybrid`, `rope: true`, `step_embedding: true`, `num_cls_tokens: 1`, `pool: cls` |
| LCGA | `variant`, `attn_carry: logprob`, `carry_gamma_init: 0.0`, `carry_gamma_per_step: false`, `carry_gamma_learnable: true`, `carry_detach: false`, `carry_clamp: -10`, `carry_source: same` |
| exit | `exit_mode: entropy`, `exit_tau: 0.05`, `exit_graph_eps: 0.01`, `min_loop_steps: 1`, `max_loop_steps: 0` (= T), `eval_dynamic_exit: true`, `eval_extrapolate: true` |
| training | `epochs: 100`, `batch_size: 64`, `lr: 5e-4`, `min_lr: 1e-5`, `weight_decay: 0.05`, `warmup_epochs: 5`, `label_smoothing: 0.1`, `grad_clip: 1.0`, `deep_supervision: 0.0`, `amp: true`, `seed: 42`, `device: auto`, `output_dir: null` |
| checkpoints | `resume: auto`, `save_every: 1`, `keep_checkpoints: 0`, `save_every_steps: 0` |
| early stopping | `early_stopping: true`, `patience: 15`, `min_delta: 0.0`, `monitor: val_acc` |

### Memory notes

With `logprob` at 224/16 and `h = 6`, one attention map is about 0.93 MB per sample per
block-step. At batch 64 and B·T = 12, that adds roughly 2–3 GB for autograd. If you run out
of memory on a 12 GB card, lower `--batch-size`, use `--carry-detach true`, or switch to
`--variant lcga-qk`. If DataLoader workers stall on Windows, use `--num-workers 0`.

## 8. Implementation notes

* **Baseline behaviour is unchanged.** With `attn_carry=none` the model has exactly the
  original state-dict keys and produces bit-identical outputs (test T1).
* **Attention logits are fp32.** In `logprob` / `realformer` mode the logits are computed in
  fp32 under autocast. They are cast back only for the `probs @ v` matmul, which avoids
  fp16 overflow (test T9).
* **γ is identity-extrapolated.** It is indexed with `min(t, T_train−1)`, like the step
  embeddings, so a model trained at T=3 still runs at T=6 (test T6).
* **Graph state survives stochastic depth.** When drop-path drops the attention branch, the
  graph state is still computed and carried.
* **`dynamic_forward` compacts the batch.** It runs the core only on active samples. Frozen
  samples keep their last `h` and their last graph. It returns `exit_steps`, `entropy`,
  `graph_delta` and `block_apps`, and matches per-sample reference runs (tests T7, T8).
* **RealFormer control has one unused γ.** In `realformer` mode block 0 has no preceding
  block within a step, so its γ is unused.

## 9. Credits

The base architecture is LoopViT:

```bibtex
@article{shu2026loopvit,
  title={LoopViT: Scaling Visual ARC with Looped Transformers},
  author={Shu, Wen-Jie and Qiu, Xuerui and Zhu, Rui-Jie and Chen, Harold Haodong and Liu, Yexin and Yang, Harry},
  journal={arXiv preprint arXiv:2602.02156},
  year={2026}
}
```

LCGA, the graph-stability exit and this classification code are by Shubham Divakar.
