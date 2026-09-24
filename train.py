"""
Train LoopViT / LCGA (Loop-Carried Graph Attention) on an image-classification dataset.

    # LCGA-logprob (the proposed model) -- everything else from the YAML
    python train.py --config config.yaml --dataset plantodc

    # pick a variant: loopvit (vanilla baseline) | lcga (= lcga-logprob) | lcga-qk | realformer
    python train.py --config config.yaml --dataset plantodc --variant loopvit
    python train.py --config config.yaml --dataset plantodc --variant lcga-qk

    # plantvillage lives under dataset/plantvillage/color (found automatically)
    python train.py --config config.yaml --dataset plantvillage

    # resume: the default `resume: auto` continues from <output_dir>/last.pt if it
    # exists, so just re-run the same command. Or point at any checkpoint:
    python train.py --config config.yaml --dataset plantodc --resume runs/lcga_plantodc/checkpoints/epoch_010.pt

    # just build the model and print the summary
    python train.py --config config.yaml --summary-only --num-classes 10

    # fine-tune from a pretrained timm ViT (loop core built from its blocks, see pretrained.py)
    python train.py --config config.yaml --dataset cub200 --variant lcga \
        --pretrained deit_small_patch16_224.fb_in1k --lr 5e-5 --new-param-lr-mult 20

    # fine-tune from one of this repo's checkpoints (e.g. PlantVillage -> PlantDoc)
    python train.py --config config.yaml --dataset plantodc --variant lcga \
        --init-from runs/lcga_plantvillage/best.pt --lr 1e-4 --new-param-lr-mult 10

Command-line flags override the YAML, and the YAML overrides the defaults below.

Checkpoints (all in --output-dir, default runs/<variant>_<dataset>):
    last.pt                   every epoch, every --save-every-steps steps, and on Ctrl+C
    best.pt                   whenever the early-stopping monitor improves
    checkpoints/epoch_XXX.pt  every --save-every epochs (--keep-checkpoints N keeps the newest N)
Each one holds model, optimizer, AMP scaler, RNG, schedule position, early-stopping
counter and position inside the epoch, so any of them can be passed to --resume.
run_history.log (append-only) records every invocation: timestamp, exact command,
cwd, git commit, resume point, and how it ended (done / early-stopped /
interrupted / failed / crashed).

Training follows the paper's fixed-depth protocol: the core is unrolled
`loop_steps` times and only the final output is supervised. Every epoch also
logs (design A, sec. 6): validation accuracy after each step 1..2T (extrapolation
past T), predictive entropy per step, the CLS attention-graph change Delta_t per
step, the learned gamma values (JSON + heatmap), dynamic-exit accuracy / steps /
block applications, peak GPU memory and throughput.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import time

import torch
import torch.nn as nn

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from data import build_dataloaders, resolve_dataset_dirs
from loop_vit import LoopViT, LoopViTConfig, graph_delta, print_model_summary

VARIANTS = {"loopvit": "none", "lcga": "logprob", "lcga-logprob": "logprob",
            "lcga-qk": "qk", "realformer": "realformer"}
CARRY_TO_NAME = {"none": "loopvit", "logprob": "lcga", "qk": "lcga-qk", "realformer": "realformer"}


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y")


def int_or_none(v):
    return None if v is None or str(v).lower() in ("none", "null", "") else int(v)


def str_or_none(v):
    return None if v is None or str(v).lower() in ("none", "null", "") else str(v)


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="YAML config file")

    d = p.add_argument_group("data")
    d.add_argument("--dataset", type=str, default=None,
                   help="folder name under --data-root (datasets/ and dataset/ are also searched)")
    d.add_argument("--data-root", type=str, default="datasets")
    d.add_argument("--train-dir", type=str, default=None, help="overrides --dataset")
    d.add_argument("--val-dir", type=str, default=None)
    d.add_argument("--num-classes", type=int_or_none, default=None, help="use N class folders (default: all)")
    d.add_argument("--class-selection", choices=["first", "random"], default="first")
    d.add_argument("--classes", nargs="*", default=None, help="explicit class folder names")
    d.add_argument("--max-per-class", type=int_or_none, default=None)
    d.add_argument("--val-split", type=float, default=0.1)
    d.add_argument("--augment", choices=["none", "basic", "trivial", "plant"], default="basic",
                   help="plant = flips (h+v), rotation 30, colour jitter, blur (plantdoc.yaml)")
    d.add_argument("--drop-classes-without-val", type=str2bool, default=False,
                   help="drop train classes that have no validation images")
    d.add_argument("--smote", type=str2bool, default=False,
                   help="add SMOTE images (same-class nearest-neighbour blends) to minority classes")
    d.add_argument("--smote-k", type=int, default=5, help="SMOTE nearest neighbours")
    d.add_argument("--smote-target", type=str, default="max",
                   help="grow each class to: max | median | an image count")
    d.add_argument("--num-workers", type=int, default=4)

    m = p.add_argument_group("model")
    m.add_argument("--image-size", type=int, default=224)
    m.add_argument("--patch-size", type=int, default=16)
    m.add_argument("--dim", type=int, default=384)
    m.add_argument("--core-depth", type=int, default=4, help="B: distinct blocks in the recurrent core")
    m.add_argument("--loop-steps", type=int, default=3, help="T: iterations of the core during training")
    m.add_argument("--num-heads", type=int, default=6)
    m.add_argument("--mlp-ratio", type=float, default=4.0)
    m.add_argument("--dropout", type=float, default=0.0)
    m.add_argument("--attn-dropout", type=float, default=0.0)
    m.add_argument("--drop-path", type=float, default=0.1)
    m.add_argument("--ffn", choices=["hybrid", "vanilla"], default="hybrid")
    m.add_argument("--rope", type=str2bool, default=True)
    m.add_argument("--step-embedding", type=str2bool, default=True)
    m.add_argument("--num-cls-tokens", type=int, default=1)
    m.add_argument("--pool", choices=["cls", "mean"], default="cls")
    m.add_argument("--norm", choices=["rms", "layer"], default="rms")
    m.add_argument("--layer-scale", type=float, default=0.0, help="LayerScale init (0 = off)")
    m.add_argument("--exit-tau", type=float, default=0.05, help="dynamic-exit entropy threshold (nats)")
    m.add_argument("--min-loop-steps", type=int, default=1)
    m.add_argument("--max-loop-steps", type=int, default=0, help="inference budget; 0 = loop_steps")

    g = p.add_argument_group("LCGA (loop-carried graph attention)")
    g.add_argument("--variant", type=str_or_none, default=None, choices=list(VARIANTS) + [None],
                   help="preset for --attn-carry: loopvit=none, lcga/lcga-logprob=logprob, "
                        "lcga-qk=qk, realformer=realformer (control)")
    g.add_argument("--attn-carry", choices=["none", "logprob", "qk", "realformer"], default="logprob")
    g.add_argument("--carry-gamma-init", type=float, default=0.0)
    g.add_argument("--carry-gamma-per-step", type=str2bool, default=False)
    g.add_argument("--carry-gamma-learnable", type=str2bool, default=True,
                   help="false + --carry-gamma-init 1 = 'gamma fixed at 1' ablation")
    g.add_argument("--carry-detach", type=str2bool, default=False)
    g.add_argument("--carry-clamp", type=float, default=-10.0)
    g.add_argument("--carry-source", choices=["same", "shifted"], default="same")
    g.add_argument("--exit-mode", choices=["entropy", "graph", "both", "either"], default="entropy")
    g.add_argument("--exit-graph-eps", type=float, default=0.01)

    w = p.add_argument_group("pretrained initialisation (pretrained.py)")
    w.add_argument("--pretrained", type=str_or_none, default=None,
                   help="timm ViT name, e.g. deit_small_patch16_224.fb_in1k; overrides dim/heads/"
                        "patch/mlp_ratio/ffn/rope/norm/layer_scale to match it")
    w.add_argument("--pretrained-blocks", type=str, default="stepwise",
                   help="stepwise | lower | average | comma list of layer indices")
    w.add_argument("--init-from", type=str_or_none, default=None,
                   help="repo checkpoint to copy matching weights from (head re-initialised)")
    w.add_argument("--new-param-lr-mult", type=float, default=1.0,
                   help="lr multiplier for head, gamma and step embeddings (use ~10-20 when "
                        "fine-tuning with a small --lr, or gamma barely moves from 0)")

    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--min-lr", type=float, default=1e-5)
    t.add_argument("--weight-decay", type=float, default=0.05)
    t.add_argument("--warmup-epochs", type=int, default=5)
    t.add_argument("--label-smoothing", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--deep-supervision", type=float, default=0.0,
                   help="weight on the loss of intermediate iterations (0 = paper protocol)")
    t.add_argument("--amp", type=str2bool, default=True, help="mixed precision on CUDA")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", type=str, default="auto")
    t.add_argument("--output-dir", type=str_or_none, default=None,
                   help="default: runs/<variant>_<dataset>")
    t.add_argument("--summary-only", action="store_true")
    t.add_argument("--eval-dynamic-exit", type=str2bool, default=True,
                   help="also evaluate with dynamic exit each epoch")
    t.add_argument("--eval-extrapolate", type=str2bool, default=True,
                   help="also report val accuracy at T+1..2T each epoch")
    t.add_argument("--progress", type=str2bool, default=True,
                   help="live progress bar for each train epoch and val pass (needs tqdm)")

    c = p.add_argument_group("checkpoints / resume")
    c.add_argument("--resume", type=str_or_none, default=None,
                   help="'auto' (= <output_dir>/last.pt if it exists), a checkpoint path, or none")
    c.add_argument("--save-every", type=int, default=1,
                   help="write checkpoints/epoch_XXX.pt every N epochs (0 = off)")
    c.add_argument("--keep-checkpoints", type=int, default=0,
                   help="keep only the newest N epoch checkpoints (0 = keep all)")
    c.add_argument("--save-every-steps", type=int, default=0,
                   help="also refresh last.pt every N optimizer steps inside an epoch (0 = off)")

    e = p.add_argument_group("early stopping")
    e.add_argument("--early-stopping", type=str2bool, default=True)
    e.add_argument("--patience", type=int, default=15, help="epochs without improvement before stopping")
    e.add_argument("--min-delta", type=float, default=0.0, help="improvement that counts as progress")
    e.add_argument("--monitor", choices=["val_acc", "val_loss"], default="val_acc")

    # YAML -> defaults, then CLI on top
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        import yaml
        with open(pre.config) as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in p._actions}
        cfg = {k.replace("-", "_"): v for k, v in cfg.items()}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"Unknown keys in {pre.config}: {sorted(unknown)}")
        p.set_defaults(**cfg)
    args = p.parse_args(argv)
    if args.variant:
        args.attn_carry = VARIANTS[args.variant]
    return args


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def progress_bar(iterable, enabled: bool, **kw):
    """tqdm bar that clears itself when done (the per-epoch summary line stays);
    plain iterable when disabled or tqdm is not installed."""
    if enabled and tqdm is not None:
        return tqdm(iterable, dynamic_ncols=True, leave=False, **kw)
    return iterable


def cosine_lr(step, total, warmup, base, min_lr):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))


def is_new_param(name: str) -> bool:
    """Parameters that are not pretrained: the head, LCGA gamma and step embeddings."""
    return name.startswith("head.") or name.endswith(".attn.gamma") or name == "step_embed"


def param_groups(model, wd, new_lr_mult: float = 1.0):
    """AdamW groups; each carries `lr_mult`, applied on top of the scheduled lr.
    With new_lr_mult == 1 these are the original two groups, so old optimizer states load."""
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        flat = p.ndim <= 1 or n.endswith((".bias", "pos_embed", "cls_token", "step_embed", ".gamma"))
        mult = new_lr_mult if (new_lr_mult != 1.0 and is_new_param(n)) else 1.0
        groups.setdefault((not flat, mult), []).append(p)
    # order: decay groups first, as before
    return [{"params": ps, "weight_decay": wd if decay else 0.0, "lr_mult": mult}
            for (decay, mult), ps in sorted(groups.items(), key=lambda kv: (not kv[0][0], kv[0][1]))]


# Inference-only knobs: changing them must not block --resume.
RUNTIME_CFG_KEYS = ("exit_tau", "min_loop_steps", "max_loop_steps", "exit_mode", "exit_graph_eps")


def structural_cfg(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in RUNTIME_CFG_KEYS}


def default_output_dir(args) -> str:
    if args.dataset:
        tag = args.dataset
    elif args.train_dir:
        tag = os.path.basename(os.path.normpath(args.train_dir))
        if tag.lower() in ("train", "color"):
            tag = os.path.basename(os.path.dirname(os.path.normpath(args.train_dir)))
    else:
        tag = "run"
    tag = tag.replace("/", "_").replace("\\", "_").replace(" ", "_")
    if args.pretrained:
        tag += "_" + args.pretrained.split(".")[0]
    elif args.init_from:
        tag += "_ft"
    return os.path.join("runs", f"{CARRY_TO_NAME[args.attn_carry]}_{tag}")


def atomic_save(obj, path: str):
    """Write to a temp file then rename, so a kill mid-save never corrupts `path`."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


_history_dir = None     # set once the run's output dir is known, for crash logging


def log_history(event: str, detail: str = ""):
    """Append a timestamped line to <output_dir>/run_history.log. The file is only ever
    appended to, so it keeps every invocation of the run, including resumes."""
    if not _history_dir:
        return
    with open(os.path.join(_history_dir, "run_history.log"), "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {event:<11} {detail}\n")


def log_run_start(out_dir: str, resume):
    global _history_dir
    _history_dir = out_dir
    argv = [os.path.basename(sys.executable)] + sys.argv
    cmd = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
                                timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    log_history("START", f"resume from {resume}" if resume else "fresh start")
    with open(os.path.join(out_dir, "run_history.log"), "a", encoding="utf-8") as f:
        f.write(f"    cmd:    {cmd}\n    cwd:    {os.getcwd()}\n"
                f"    git:    {commit}\n    python: {sys.version.split()[0]}, "
                f"torch {torch.__version__}\n")


def fmt_list(xs, f="{:.3f}"):
    return " ".join("nan" if (x is None or (isinstance(x, float) and math.isnan(x)))
                    else f.format(x) for x in xs)


@torch.no_grad()
def evaluate(model, loader, device, criterion, use_amp=False, amp_dtype=torch.float16,
             extrap=True, dynamic_exit=False, progress=False):
    """Validation pass (design A, sec. 6): accuracy / entropy / Delta_t after every
    step 1..S (S = 2T with `extrap`, else T), plus optional dynamic exit."""
    model.eval()
    T = model.cfg.loop_steps
    S = 2 * T if extrap else T
    graphs_on = model.cfg.num_cls_tokens >= 1
    n, loss_sum = 0, 0.0
    hits, ent_sum, delta_sum = [0] * S, [0.0] * S, [0.0] * S
    dyn_correct, dyn_steps, dyn_apps = 0, 0.0, 0
    for x, y in progress_bar(loader, progress, desc="  val", unit="batch"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if graphs_on:
                outs, graphs = model.logits_and_graphs(x, S)
            else:
                outs, graphs = model.logits_per_step(x, S), None
            if dynamic_exit:
                d = model.dynamic_forward(x)
        outs = [o.float() for o in outs]
        for t, o in enumerate(outs):
            hits[t] += (o.argmax(1) == y).sum().item()
            ent_sum[t] += model.entropy(o).sum().item()
            if graphs is not None and t >= 1:
                delta_sum[t] += graph_delta(graphs[t], graphs[t - 1]).sum().item()
        loss_sum += criterion(outs[T - 1], y).item() * y.size(0)
        if dynamic_exit:
            dyn_correct += (d["logits"].argmax(1) == y).sum().item()
            dyn_steps += d["exit_steps"].sum().item()
            dyn_apps += d["block_apps"]
        n += y.size(0)
    n = max(n, 1)
    res = {"loss": loss_sum / n, "acc": hits[T - 1] / n,
           "acc_per_step": [h / n for h in hits],
           "entropy_per_step": [e / n for e in ent_sum],
           "graph_delta_per_step": ([float("nan")] + [d / n for d in delta_sum[1:]]
                                    if graphs_on else [])}
    if dynamic_exit:
        res.update(dyn_acc=dyn_correct / n, dyn_steps=dyn_steps / n, dyn_block_apps=dyn_apps / n)
    return res


def truncate_logs(out_dir: str, last_epoch: int):
    """On resume, drop log rows written after the checkpoint we resume from."""
    log_path = os.path.join(out_dir, "log.csv")
    if os.path.exists(log_path):
        with open(log_path, newline="") as f:
            rows = list(csv.reader(f))
        keep = rows[:1] + [r for r in rows[1:] if r and int(r[0]) <= last_epoch]
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerows(keep)
    jpath = os.path.join(out_dir, "metrics.jsonl")
    if os.path.exists(jpath):
        with open(jpath) as f:
            lines = [l for l in f if l.strip() and json.loads(l)["epoch"] <= last_epoch]
        with open(jpath, "w") as f:
            f.writelines(lines)


def save_gamma_plot(model, out_dir: str, epoch: int):
    gam = model.gamma_values()
    if gam is None:
        return
    try:
        from analyze_lcga import plot_gamma_heatmap
        plot_gamma_heatmap(gam, os.path.join(out_dir, "gamma_heatmap.png"),
                           title=f"learned gamma (epoch {epoch})")
    except Exception as ex:   # matplotlib missing / headless issues must not kill training
        print(f"[warn] gamma heatmap not written: {ex}")


def main():
    args = get_args()
    seed_all(args.seed)
    device = pick_device(args.device)
    print(f"[env] device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ---- pretrained architecture (must be known before the transforms) ------
    vit, in_mean, in_std, qkv_bias = None, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), True
    if args.pretrained:
        from pretrained import create_timm_vit, timm_arch
        print(f"[pretrained] loading timm model {args.pretrained}")
        vit = create_timm_vit(args.pretrained)
        arch = timm_arch(vit)
        in_mean, in_std = arch.pop("input_mean"), arch.pop("input_std")
        qkv_bias, depth = arch.pop("qkv_bias"), arch.pop("_depth")
        changed = {k: (getattr(args, k), v) for k, v in arch.items() if getattr(args, k) != v}
        for k, v in arch.items():
            setattr(args, k, v)
        print(f"[pretrained] {depth} pretrained blocks -> core of {args.core_depth} x "
              f"{args.loop_steps} steps; input mean {in_mean} std {in_std}")
        if changed:
            print("[pretrained] args set to match the ViT (was -> now): "
                  + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in changed.items()))
        if args.image_size % args.patch_size:
            raise SystemExit(f"--image-size {args.image_size} must be divisible by the "
                             f"ViT patch size {args.patch_size}")

    # ---- data ---------------------------------------------------------------
    if args.summary_only and not (args.dataset or args.train_dir):
        class_names = [f"class_{i}" for i in range(args.num_classes or 10)]
        train_loader = val_loader = None
    else:
        train_dir, val_dir = resolve_dataset_dirs(
            args.data_root, args.dataset, args.train_dir, args.val_dir)
        print(f"[data] train dir: {train_dir}")
        print(f"[data] val dir:   {val_dir or '(held out from train)'}")
        train_loader, val_loader, class_names = build_dataloaders(
            train_dir, val_dir, args.image_size, args.batch_size,
            args.num_workers, args.num_classes, args.class_selection, args.classes,
            args.max_per_class, args.val_split, args.augment, args.seed,
            pin_memory=device.type == "cuda", mean=in_mean, std=in_std,
            drop_classes_without_val=args.drop_classes_without_val, smote=args.smote,
            smote_k=args.smote_k, smote_target=args.smote_target)

    # ---- model --------------------------------------------------------------
    mcfg = LoopViTConfig(
        image_size=args.image_size, patch_size=args.patch_size, num_classes=len(class_names),
        dim=args.dim, core_depth=args.core_depth, loop_steps=args.loop_steps,
        num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        attn_dropout=args.attn_dropout, drop_path=args.drop_path, ffn=args.ffn,
        rope=args.rope, step_embedding=args.step_embedding,
        num_cls_tokens=args.num_cls_tokens, pool=args.pool, qkv_bias=qkv_bias,
        norm=args.norm, layer_scale=args.layer_scale, input_mean=in_mean, input_std=in_std,
        exit_tau=args.exit_tau, min_loop_steps=args.min_loop_steps,
        max_loop_steps=args.max_loop_steps,
        attn_carry=args.attn_carry, carry_gamma_init=args.carry_gamma_init,
        carry_gamma_per_step=args.carry_gamma_per_step,
        carry_gamma_learnable=args.carry_gamma_learnable, carry_detach=args.carry_detach,
        carry_clamp=args.carry_clamp, carry_source=args.carry_source,
        exit_mode=args.exit_mode, exit_graph_eps=args.exit_graph_eps)
    model = LoopViT(mcfg)
    if vit is not None:
        from pretrained import load_timm_vit
        src = load_timm_vit(model, vit, args.pretrained_blocks)
        print(f"[pretrained] core blocks <- pretrained layers "
              f"{[s[0] if len(s) == 1 else s for s in src]} ({args.pretrained_blocks})")
        del vit
    if args.init_from:
        from pretrained import load_matching
        loaded, skipped = load_matching(model, args.init_from)
        print(f"[init-from] {args.init_from}: copied {len(loaded)} tensors, "
              f"kept fresh init for {len(skipped)}: {skipped}")
        if not loaded:
            raise SystemExit("--init-from: no tensor matched; is the model config the same?")
    print_model_summary(model)
    model.to(device)
    if args.summary_only:
        return
    args.output_dir = args.output_dir or default_output_dir(args)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[run] output dir: {args.output_dir}")

    # ---- optimisation -------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay, args.new_param_lr_mult),
                            lr=args.lr, betas=(0.9, 0.999))
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    sampler = train_loader.sampler
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    # ---- resume -------------------------------------------------------------
    resume = args.resume
    if resume and resume.lower() == "auto":
        cand = os.path.join(args.output_dir, "last.pt")
        resume = cand if os.path.exists(cand) else None
        print(f"[resume] auto: {'found ' + cand if resume else 'no last.pt, starting fresh'}")
    log_run_start(args.output_dir, resume)

    start_epoch, step, best_acc, best_score, stale = 1, 0, -1.0, None, 0
    skip_batches, partial = 0, None
    if resume:
        print(f"[resume] loading {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        if ckpt.get("classes") != class_names:
            raise SystemExit("--resume checkpoint classes do not match the current dataset/config")
        # checkpoints written before a config field existed used its default
        ck_cfg = {**LoopViTConfig().to_dict(), **ckpt.get("model_cfg", {})}
        if structural_cfg(ck_cfg) != structural_cfg(mcfg.to_dict()):
            diff = {k: (ck_cfg.get(k), v) for k, v in structural_cfg(mcfg.to_dict()).items()
                    if ck_cfg.get(k) != v}
            raise SystemExit(f"--resume checkpoint model config does not match the current model "
                             f"args (checkpoint, now): {diff}\nUse a different --output-dir or "
                             f"--resume none for a fresh run.")
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        skip_batches = ckpt.get("batch_in_epoch", 0)
        partial = ckpt.get("partial_stats")
        best_acc = ckpt.get("best_acc", -1.0)
        best_score = ckpt.get("best_score")
        stale = ckpt.get("epochs_without_improvement", 0)
        rng = ckpt.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                # map_location may have put these on the GPU; they must be CPU ByteTensors
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        where = (f"epoch {start_epoch}, batch {skip_batches}/{steps_per_epoch}" if skip_batches
                 else f"start of epoch {start_epoch}")
        print(f"[resume] resuming at {where} (step {step}), best_acc {best_acc:.4f}, "
              f"{stale} epoch(s) without improvement")
        log_history("RESUMED", f"at {where} (step {step}), best_acc {best_acc:.4f}")
        truncate_logs(args.output_dir, start_epoch - 1)
        if args.early_stopping and stale >= args.patience and not skip_batches:
            print(f"[resume] this run already early-stopped ({stale} >= patience {args.patience}). "
                  f"Raise --patience or pass --early-stopping false to keep training.")
            log_history("END", "nothing to do: run already early-stopped")
            return
        if start_epoch > args.epochs:
            print(f"[resume] already trained for {start_epoch - 1} epochs "
                  f"(--epochs {args.epochs}). Raise --epochs to continue.")
            log_history("END", f"nothing to do: already trained {start_epoch - 1} epochs")
            return

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "model": mcfg.to_dict(), "classes": class_names}, f, indent=2)

    monitor = args.monitor if val_loader else "train_acc"
    higher_is_better = not monitor.endswith("loss")
    if args.early_stopping:
        print(f"[early stopping] monitor {monitor}, patience {args.patience}, "
              f"min_delta {args.min_delta}")

    log_path = os.path.join(args.output_dir, "log.csv")
    if not (resume and os.path.exists(log_path)):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
                                    "val_acc_per_step", "entropy_per_step", "graph_delta_per_step",
                                    "dyn_acc", "dyn_steps", "dyn_block_apps", "gamma_mean",
                                    "gamma_absmax", "img_per_s", "peak_mem_gb",
                                    "epochs_without_improvement", "sec"])

    def make_ckpt(epoch_done: int, batch_in_epoch: int = 0, stats=None, val_acc=None):
        return {"model": model.state_dict(), "model_cfg": mcfg.to_dict(),
                "classes": class_names, "args": vars(args),
                "epoch": epoch_done, "batch_in_epoch": batch_in_epoch, "partial_stats": stats,
                "val_acc": val_acc, "step": step, "best_acc": best_acc, "best_score": best_score,
                "monitor": monitor, "epochs_without_improvement": stale,
                "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                "rng_state": {"python": random.getstate(), "torch": torch.get_rng_state(),
                              "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}

    last_path = os.path.join(args.output_dir, "last.pt")
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        skip = skip_batches if epoch == start_epoch else 0
        sampler.set_epoch(epoch, min(skip * args.batch_size, len(train_loader.dataset)))
        st = dict(partial) if (skip and partial) else {"n": 0, "loss_sum": 0.0, "correct": 0}
        done_in_epoch, seen = skip, 0
        lr = cosine_lr(max(step - 1, 0), total_steps, warmup_steps, args.lr, args.min_lr)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        bar = progress_bar(train_loader, args.progress, total=steps_per_epoch, initial=skip,
                           desc=f"epoch {epoch}/{args.epochs}", unit="batch")
        try:
            for x, y in bar:
                lr = cosine_lr(step, total_steps, warmup_steps, args.lr, args.min_lr)
                for g in opt.param_groups:
                    g["lr"] = lr * g.get("lr_mult", 1.0)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    if args.deep_supervision > 0:
                        outs = model.logits_per_step(x)
                        logits = outs[-1]
                        loss = criterion(logits, y)
                        if len(outs) > 1:
                            aux = sum(criterion(o, y) for o in outs[:-1]) / (len(outs) - 1)
                            loss = loss + args.deep_supervision * aux
                    else:
                        logits = model(x)            # supervise the final output only
                        loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss {loss.item()} at step {step}; "
                                             f"resume from {last_path} with a lower --lr or --amp false")
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if args.grad_clip:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                step += 1
                done_in_epoch += 1
                seen += y.size(0)
                st["loss_sum"] += loss.item() * y.size(0)
                st["correct"] += (logits.argmax(1) == y).sum().item()
                st["n"] += y.size(0)
                if bar is not train_loader and done_in_epoch % 10 == 0:
                    bar.set_postfix(loss=f"{st['loss_sum'] / st['n']:.4f}",
                                    acc=f"{st['correct'] / st['n']:.4f}", lr=f"{lr:.2e}",
                                    refresh=False)
                if args.save_every_steps and step % args.save_every_steps == 0:
                    atomic_save(make_ckpt(epoch - 1, done_in_epoch, dict(st)), last_path)
        except KeyboardInterrupt:
            atomic_save(make_ckpt(epoch - 1, done_in_epoch, dict(st)), last_path)
            print(f"\n[interrupted] saved {last_path} at epoch {epoch}, batch "
                  f"{done_in_epoch}/{steps_per_epoch}. Re-run the same command "
                  f"(resume: auto) or pass --resume {last_path} to continue.")
            log_history("INTERRUPTED", f"Ctrl+C at epoch {epoch}, batch {done_in_epoch}/"
                                       f"{steps_per_epoch}; saved {last_path}")
            raise SystemExit(130)
        train_sec = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0
        img_s = seen / train_sec if train_sec > 0 else 0.0

        try:
            tr = {"loss": st["loss_sum"] / max(st["n"], 1), "acc": st["correct"] / max(st["n"], 1)}
            va = (evaluate(model, val_loader, device, eval_criterion, use_amp, amp_dtype,
                           extrap=args.eval_extrapolate, dynamic_exit=args.eval_dynamic_exit,
                           progress=args.progress)
                  if val_loader else {})
        except KeyboardInterrupt:
            atomic_save(make_ckpt(epoch - 1, steps_per_epoch, dict(st)), last_path)
            print(f"\n[interrupted during eval] saved {last_path}; resuming re-runs this "
                  f"epoch's evaluation only.")
            log_history("INTERRUPTED", f"Ctrl+C during eval of epoch {epoch}; saved {last_path}")
            raise SystemExit(130)
        sec = time.time() - t0

        gam = model.gamma_values()
        T = mcfg.loop_steps
        per_step = fmt_list(va.get("acc_per_step", []))
        msg = (f"epoch {epoch:3d}/{args.epochs} | lr {lr:.2e} | train loss {tr['loss']:.4f} "
               f"acc {tr['acc']:.4f} | {img_s:.0f} img/s"
               + (f", {peak_mem:.2f} GB" if device.type == "cuda" else ""))
        if va:
            msg += (f"\n    val loss {va['loss']:.4f} acc {va['acc']:.4f} | acc per step "
                    f"[{fmt_list(va['acc_per_step'][:T])}]")
            if len(va["acc_per_step"]) > T:
                msg += f" extrap [{fmt_list(va['acc_per_step'][T:])}]"
            msg += f"\n    entropy/step [{fmt_list(va['entropy_per_step'])}]"
            if va["graph_delta_per_step"]:
                msg += f" | graph delta/step [{fmt_list(va['graph_delta_per_step'], '{:.4f}')}]"
            if "dyn_acc" in va:
                msg += (f"\n    exit({mcfg.exit_mode}) acc {va['dyn_acc']:.4f} @ {va['dyn_steps']:.2f} "
                        f"steps, {va['dyn_block_apps']:.1f} block apps/img")
        if gam is not None:
            msg += f"\n    gamma mean {gam.mean():+.4f} |max| {gam.abs().max():.4f}"
        print(msg + f" | {sec:.1f}s")

        # ---- checkpoint + early stopping ------------------------------------
        score = {"val_acc": va.get("acc"), "val_loss": va.get("loss"),
                 "train_acc": tr["acc"]}[monitor]
        improved = (best_score is None or
                    (score > best_score + args.min_delta if higher_is_better
                     else score < best_score - args.min_delta))
        if improved:
            best_score, stale = score, 0
        else:
            stale += 1
        best_acc = max(best_acc, va.get("acc", tr["acc"]))

        nan = float("nan")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr:.3e}", f"{tr['loss']:.4f}", f"{tr['acc']:.4f}",
                f"{va.get('loss', nan):.4f}", f"{va.get('acc', nan):.4f}", per_step,
                fmt_list(va.get("entropy_per_step", [])),
                fmt_list(va.get("graph_delta_per_step", []), "{:.5f}"),
                f"{va.get('dyn_acc', nan):.4f}", f"{va.get('dyn_steps', nan):.2f}",
                f"{va.get('dyn_block_apps', nan):.2f}",
                f"{gam.mean().item() if gam is not None else nan:.5f}",
                f"{gam.abs().max().item() if gam is not None else nan:.5f}",
                f"{img_s:.1f}", f"{peak_mem:.3f}", stale, f"{sec:.1f}"])
        with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
            rec = {"epoch": epoch, "lr": lr, "train": tr, "val": va, "img_per_s": img_s,
                   "peak_mem_gb": peak_mem, "improved": improved, "stale": stale,
                   "gamma": gam.tolist() if gam is not None else None}
            f.write(json.dumps(rec, default=lambda o: None) + "\n")
        save_gamma_plot(model, args.output_dir, epoch)

        ckpt = make_ckpt(epoch, 0, None, va.get("acc"))
        atomic_save(ckpt, last_path)
        if improved:
            shutil.copyfile(last_path, os.path.join(args.output_dir, "best.pt"))
            print(f"    new best {monitor} {best_score:.4f} -> best.pt")
        if args.save_every and epoch % args.save_every == 0:
            shutil.copyfile(last_path, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))
            if args.keep_checkpoints > 0:
                old = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
                for pth in old[:-args.keep_checkpoints]:
                    os.remove(pth)

        if args.early_stopping and stale >= args.patience:
            print(f"early stopping: {monitor} has not improved by more than "
                  f"{args.min_delta} for {stale} epochs (best {best_score:.4f})")
            stopped_early = True
            break

    tag = "early-stopped" if stopped_early else "done"
    print(f"{tag}. best {monitor} {best_score:.4f} (best acc {best_acc:.4f}) "
          f"-> {args.output_dir}/best.pt")
    log_history("END", f"{tag} after epoch {epoch}, best {monitor} {best_score:.4f}, "
                       f"best acc {best_acc:.4f}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0, 130):         # 130 = Ctrl+C, already logged
            log_history("FAILED", str(e.code).splitlines()[0])
        raise
    except BaseException as e:
        log_history("CRASHED", f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        raise
