"""
LCGA analysis (design doc A, sec. 7). Takes one or more checkpoints (e.g. the
vanilla LoopViT and the LCGA run on the same dataset) and a data split, and
writes figures + JSON to --out-dir:

  1. crystallization.png       graph change Delta_t vs step t = 1..2T (mean +- std), all ckpts
                               (+ accuracy_per_step.png: accuracy vs t, incl. extrapolation)
  2. cls_maps_<label>.png      CLS attention map (last block, head-averaged) per step on 8 images
  3. attn_entropy_<label>.png  per-head attention entropy per step (does LCGA sharpen the graph?)
  4. oversmoothing.png         Dirichlet energy and effective rank of patch tokens per step
  5. gamma_<label>.png         learned gamma heatmap (blocks x heads [x steps])
  6. exit_pareto.png           accuracy vs mean block applications for every exit mode,
                               sweeping tau and exit_graph_eps
  summary.json                 every number behind the figures

    python analyze_lcga.py --ckpt runs/loopvit_plantodc/best.pt runs/lcga_plantodc/best.pt
    python analyze_lcga.py --ckpt runs/lcga_plantodc/best.pt --max-images 500 --out-dir analysis/plantodc

The data split is rebuilt from the arguments stored in the (first) checkpoint,
so the same classes / val split are used; --dataset / --train-dir / --val-dir /
--data-root override them.

The exit Pareto is computed offline from one full-depth pass: freezing a sample
at step t (z_k = z_t, S_k = S_t) means its final logits are exactly its step-t
logits, so every (mode, tau, eps) point is a re-read of the same trajectories.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from loop_vit import LoopViT, LoopViTConfig, graph_delta

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- #
# plotting helpers (also used by train.py)
# --------------------------------------------------------------------------- #
def plot_gamma_heatmap(gamma: torch.Tensor, path: str, title: str = "learned gamma"):
    """gamma: (B, gamma_steps, h). One panel (blocks x heads) per gamma step."""
    plt = _plt()
    g = gamma.float().numpy()
    B, S, H = g.shape
    vmax = max(float(abs(g).max()), 1e-6)
    fig, axes = plt.subplots(1, S, figsize=(1.2 + 0.7 * H * S, 0.9 + 0.55 * B), squeeze=False)
    for s in range(S):
        ax = axes[0, s]
        im = ax.imshow(g[:, s, :], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(H), [f"h{i}" for i in range(H)])
        ax.set_yticks(range(B), [f"b{i}" for i in range(B)])
        if S > 1:
            ax.set_title(f"step {s + 1}")
        if B * H <= 96:
            for b in range(B):
                for h in range(H):
                    ax.text(h, b, f"{g[b, s, h]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def pareto_front(points):
    """points: list of (cost, acc). Returns the non-dominated subset sorted by cost."""
    front, best = [], -1.0
    for c, a in sorted(points, key=lambda p: (p[0], -p[1])):
        if a > best:
            front.append((c, a))
            best = a
    return front


# --------------------------------------------------------------------------- #
# data collection
# --------------------------------------------------------------------------- #
def load_model(path: str, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LoopViT(LoopViTConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device), ckpt


def dirichlet_energy(patches: torch.Tensor, grid: int) -> torch.Tensor:
    """E = mean over 4-neighbour edges of ||x_i/|x_i| - x_j/|x_j|||^2. patches: (N, P, D) -> (N,)."""
    x = torch.nn.functional.normalize(patches.float(), dim=-1)
    x = x.view(x.shape[0], grid, grid, -1)
    dh = (x[:, :, 1:] - x[:, :, :-1]).pow(2).sum(-1).flatten(1)
    dv = (x[:, 1:, :] - x[:, :-1, :]).pow(2).sum(-1).flatten(1)
    return torch.cat([dh, dv], 1).mean(1)


def effective_rank(patches: torch.Tensor) -> torch.Tensor:
    """exp(H(sigma / sum sigma)) of the token-centred (P, D) matrix, per sample -> (N,)."""
    x = patches.float()
    x = x - x.mean(1, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / s.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.exp(-(p * (p + 1e-12).log()).sum(-1))


@torch.no_grad()
def collect(model: LoopViT, loader, device, steps: int, max_images: int, num_maps: int):
    """One full-depth pass: per-step logits, entropy, Delta_t, per-head attention
    entropy, oversmoothing stats, and CLS maps for the first `num_maps` images."""
    cfg = model.cfg
    npre, grid = cfg.num_cls_tokens, model.patch_embed.grid
    model.set_record_entropy(True)
    out = {k: [] for k in ("logits", "labels", "delta", "row_ent", "dirichlet", "erank")}
    maps, map_imgs, seen = [], [], 0
    for x, y in loader:
        if seen >= max_images:
            break
        x, y = x[: max_images - seen].to(device), y[: max_images - seen]
        h, states, prev = model.embed(x), None, None
        lg, dl, re, de, er, cl = [], [], [], [], [], []
        for t in range(steps):
            h, states, cls = model.core_step(h, t, states=states, need_cls=npre >= 1)
            lg.append(model.classify(h).float().cpu())
            re.append(torch.stack([b.attn.last_row_entropy for b in model.core], 1).cpu())
            patches = h[:, npre:]
            de.append(dirichlet_energy(patches, grid).cpu())
            er.append(effective_rank(patches).cpu())
            if cls is not None:
                cls = cls.float()
                dl.append(torch.full((x.shape[0],), float("nan")) if prev is None
                          else graph_delta(cls, prev).cpu())
                prev = cls
                cl.append(cls.cpu())
        out["logits"].append(torch.stack(lg, 1))            # (n, S, C)
        out["labels"].append(y)
        out["row_ent"].append(torch.stack(re, 1))           # (n, S, B, h)
        out["dirichlet"].append(torch.stack(de, 1))         # (n, S)
        out["erank"].append(torch.stack(er, 1))
        if dl:
            out["delta"].append(torch.stack(dl, 1))         # (n, S)
        if cl and len(map_imgs) < num_maps:
            k = min(num_maps - len(map_imgs), x.shape[0])
            maps.append(torch.stack(cl, 1)[:k])             # (k, S, B, h, T_tok)
            map_imgs += [(x[i].cpu() * STD + MEAN).clamp(0, 1) for i in range(k)]
        seen += x.shape[0]
    model.set_record_entropy(False)
    res = {k: torch.cat(v) for k, v in out.items() if v}
    res["maps"] = torch.cat(maps) if maps else None
    res["map_imgs"] = map_imgs
    return res


def simulate_exit(logits, labels, ent, delta, B, mode, tau, eps, min_steps=1):
    """Offline dynamic exit over precomputed trajectories. Returns (acc, mean block apps)."""
    N, S, _ = logits.shape
    ent_ok = ent < tau
    graph_ok = torch.zeros_like(ent_ok) if delta is None else torch.nan_to_num(delta, nan=float("inf")) < eps
    ok = {"entropy": ent_ok, "graph": graph_ok, "both": ent_ok & graph_ok,
          "either": ent_ok | graph_ok}[mode].clone()
    ok[:, : max(min_steps - 1, 0)] = False
    ok[:, S - 1] = True
    first = ok.int().argmax(1)                               # 0-based exit step
    pred = logits[torch.arange(N), first].argmax(-1)
    return (pred == labels).float().mean().item(), ((first + 1) * B).float().mean().item()


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None, help="legend names (default: variant/run dir)")
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--dataset", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--train-dir", default=None)
    p.add_argument("--val-dir", default=None)
    p.add_argument("--max-images", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--steps", type=int, default=0, help="loop steps to analyse (0 = 2 x loop_steps)")
    p.add_argument("--num-maps", type=int, default=8)
    p.add_argument("--min-steps", type=int, default=1)
    p.add_argument("--taus", type=float, nargs="*", default=None)
    p.add_argument("--graph-eps", type=float, nargs="*", default=None)
    p.add_argument("--out-dir", default=None, help="default: <first ckpt dir>/analysis")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    from data import build_dataloaders, resolve_dataset_dirs
    plt = _plt()
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.ckpt[0])), "analysis")
    os.makedirs(out_dir, exist_ok=True)

    models = [load_model(c, device) for c in args.ckpt]
    labels = args.labels or []
    for i, (m, ck) in enumerate(models[len(labels):], start=len(labels)):
        labels.append(f"{m.cfg.attn_carry}:{os.path.basename(os.path.dirname(os.path.abspath(args.ckpt[i])))}")
    classes = models[0][1]["classes"]
    for (m, ck), c in zip(models, args.ckpt):
        if ck["classes"] != classes:
            raise SystemExit(f"{c} was trained on different classes than {args.ckpt[0]}")

    # ---- data (rebuilt from the first checkpoint's training args) ----------
    targs = models[0][1].get("args") or {}
    tr_dir, va_dir = resolve_dataset_dirs(
        args.data_root or targs.get("data_root", "datasets"), args.dataset or targs.get("dataset"),
        args.train_dir or targs.get("train_dir"), args.val_dir or targs.get("val_dir"))
    img_size = models[0][0].cfg.image_size
    in_mean, in_std = models[0][0].cfg.input_mean, models[0][0].cfg.input_std
    train_loader, val_loader, names = build_dataloaders(
        tr_dir, va_dir, img_size, args.batch_size, args.num_workers,
        None, "first", classes, targs.get("max_per_class"), targs.get("val_split", 0.1),
        "none", targs.get("seed", 42), pin_memory=False, mean=in_mean, std=in_std)
    if args.split == "val":
        if val_loader is None:
            raise SystemExit("no validation split available; use --split train")
        loader = val_loader
    else:   # deterministic, un-augmented view of the train split
        from data import SampleListDataset, build_transforms
        loader = torch.utils.data.DataLoader(SampleListDataset(
            train_loader.dataset.samples, names, build_transforms(img_size, mean=in_mean, std=in_std)[1]),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    summary = {"split": args.split, "classes": len(classes), "runs": {}}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    results = []
    for (model, ck), label in zip(models, labels):
        cfg = model.cfg
        S = args.steps or 2 * cfg.loop_steps
        print(f"[analyze] {label}: {S} steps on up to {args.max_images} {args.split} images")
        r = collect(model, loader, device, S, args.max_images, args.num_maps)
        r.update(label=label, cfg=cfg, S=S, model=model)
        results.append(r)

    # ---- 1. crystallization + accuracy per step -----------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for i, r in enumerate(results):
        col = colors[i % len(colors)]
        S, T = r["S"], r["cfg"].loop_steps
        steps = list(range(1, S + 1))
        acc = (r["logits"].argmax(-1) == r["labels"][:, None]).float().mean(0).tolist()
        ent = LoopViT.entropy(r["logits"])                           # (N, S)
        run = summary["runs"].setdefault(r["label"], {})
        run.update(attn_carry=r["cfg"].attn_carry, loop_steps=T, images=len(r["labels"]),
                   acc_per_step=acc, pred_entropy_per_step=ent.mean(0).tolist())
        ax2.plot(steps, acc, "o-", color=col, label=r["label"])
        ax2.axvline(T, color=col, ls=":", lw=0.8)
        if "delta" in r:
            d = r["delta"][:, 1:]
            mu, sd = d.mean(0), d.std(0)
            ax.plot(steps[1:], mu, "o-", color=col, label=r["label"])
            ax.fill_between(steps[1:], (mu - sd).clamp_min(1e-12), mu + sd, color=col, alpha=0.2)
            ax.axvline(T, color=col, ls=":", lw=0.8)
            run["graph_delta_mean"] = [float("nan")] + mu.tolist()
            run["graph_delta_std"] = [float("nan")] + sd.tolist()
    ax.set(xlabel="loop step t", ylabel="Delta_t = KL(p_t || p_t-1)  [nats]", yscale="log",
           title="Graph crystallization (CLS attention rows)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "crystallization.png"), dpi=130, bbox_inches="tight")
    ax2.set(xlabel="loop step t", ylabel="accuracy", title="Accuracy per step (dotted = T_train)")
    ax2.legend(fontsize=8)
    fig2.savefig(os.path.join(out_dir, "accuracy_per_step.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    plt.close(fig2)

    # ---- 2. CLS attention maps over steps -----------------------------------
    for r in results:
        if r["maps"] is None:
            continue
        cfg, grid = r["cfg"], r["model"].patch_embed.grid
        maps = r["maps"][:, :, -1].mean(2)[..., cfg.num_cls_tokens:]     # (k, S, P) last block
        k, S = maps.shape[0], maps.shape[1]
        fig, axes = plt.subplots(S + 1, k, figsize=(1.6 * k, 1.6 * (S + 1)), squeeze=False)
        for j in range(k):
            img = r["map_imgs"][j].permute(1, 2, 0).numpy()
            axes[0, j].imshow(img)
            for t in range(S):
                m = maps[j, t].view(1, 1, grid, grid)
                m = torch.nn.functional.interpolate(m, size=img.shape[:2], mode="bilinear")[0, 0]
                axes[t + 1, j].imshow(img)
                axes[t + 1, j].imshow(m.numpy(), cmap="inferno", alpha=0.6)
        for a in axes.ravel():
            a.set_xticks([])
            a.set_yticks([])
        axes[0, 0].set_ylabel("image", fontsize=8)
        for t in range(S):
            axes[t + 1, 0].set_ylabel(f"t={t + 1}", fontsize=8)
        fig.suptitle(f"{r['label']}: CLS attention, last block, head mean")
        fig.savefig(os.path.join(out_dir, f"cls_maps_{_safe(r['label'])}.png"), dpi=110,
                    bbox_inches="tight")
        plt.close(fig)

    # ---- 3. per-head attention entropy per step ------------------------------
    for r in results:
        re = r["row_ent"].mean(0)                                        # (S, B, h)
        S, B, H = re.shape
        fig, axes = plt.subplots(1, B, figsize=(3.2 * B, 3), squeeze=False, sharey=True)
        for b in range(B):
            for hh in range(H):
                axes[0, b].plot(range(1, S + 1), re[:, b, hh], "o-", ms=3, label=f"h{hh}")
            axes[0, b].axvline(r["cfg"].loop_steps, color="k", ls=":", lw=0.8)
            axes[0, b].set(title=f"block {b}", xlabel="step")
        axes[0, 0].set_ylabel("mean row entropy [nats]")
        axes[0, -1].legend(fontsize=6)
        fig.suptitle(f"{r['label']}: attention entropy per head")
        fig.savefig(os.path.join(out_dir, f"attn_entropy_{_safe(r['label'])}.png"), dpi=120,
                    bbox_inches="tight")
        plt.close(fig)
        summary["runs"][r["label"]]["attn_entropy_step_block_head"] = re.tolist()

    # ---- 4. oversmoothing --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, r in enumerate(results):
        col, steps = colors[i % len(colors)], range(1, r["S"] + 1)
        de, er = r["dirichlet"].mean(0), r["erank"].mean(0)
        axes[0].plot(steps, de, "o-", color=col, label=r["label"])
        axes[1].plot(steps, er, "o-", color=col, label=r["label"])
        summary["runs"][r["label"]].update(dirichlet_energy=de.tolist(), effective_rank=er.tolist())
    axes[0].set(xlabel="step", ylabel="Dirichlet energy (normalised tokens)", title="Dirichlet energy")
    axes[1].set(xlabel="step", ylabel="effective rank", title="Effective rank of patch tokens")
    axes[0].legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "oversmoothing.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- 5. gamma ------------------------------------------------------------
    for r in results:
        gam = r["model"].gamma_values()
        if gam is not None:
            plot_gamma_heatmap(gam, os.path.join(out_dir, f"gamma_{_safe(r['label'])}.png"),
                               title=f"{r['label']}: learned gamma")
            summary["runs"][r["label"]]["gamma"] = gam.tolist()

    # ---- 6. exit Pareto --------------------------------------------------------
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    mode_col = {"entropy": "C0", "graph": "C1", "both": "C2", "either": "C3"}
    for i, r in enumerate(results):
        ax = axes[0, i]
        B, C = r["cfg"].core_depth, r["logits"].shape[-1]
        ent = LoopViT.entropy(r["logits"])
        delta = r.get("delta")
        taus = args.taus or [float(v) for v in torch.logspace(-3, math.log10(math.log(C)), 12)]
        epss = args.graph_eps or [float(v) for v in torch.logspace(-5, 0, 12)]
        grid = {"entropy": [(t, float("inf")) for t in taus],
                "graph": [(-1.0, e) for e in epss],
                "both": [(t, e) for t in taus for e in epss],
                "either": [(t, e) for t in taus for e in epss]}
        if delta is None:
            grid = {"entropy": grid["entropy"]}
        pareto = {}
        for mode, pts in grid.items():
            res = [(tau, eps) + simulate_exit(r["logits"], r["labels"], ent, delta, B, mode, tau,
                                              eps, args.min_steps) for tau, eps in pts]
            front = pareto_front([(c, a) for _, _, a, c in res])
            pareto[mode] = {"points": [{"tau": t, "eps": e, "acc": a, "block_apps": c}
                                       for t, e, a, c in res], "front": front}
            ax.scatter([c for *_, c in res], [a for _, _, a, _ in res], s=8, alpha=0.35,
                       color=mode_col[mode])
            ax.plot(*zip(*front), "-", color=mode_col[mode], label=f"exit: {mode}")
        fixed = [((t + 1) * B, (r["logits"][:, t].argmax(-1) == r["labels"]).float().mean().item())
                 for t in range(r["S"])]
        ax.plot(*zip(*fixed), "k--o", ms=3, label="fixed T")
        ax.set(xlabel="mean block applications / image", ylabel="accuracy", title=r["label"])
        ax.legend(fontsize=8)
        summary["runs"][r["label"]]["exit_pareto"] = pareto
        summary["runs"][r["label"]]["fixed_steps"] = fixed
    fig.savefig(os.path.join(out_dir, "exit_pareto.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[analyze] wrote figures and summary.json to {out_dir}")
    for label, run in summary["runs"].items():
        print(f"  {label}: acc/step {[round(a, 4) for a in run['acc_per_step']]}")
        if "graph_delta_mean" in run:
            print(f"  {' ' * len(label)}  Delta/step {[round(d, 5) for d in run['graph_delta_mean'][1:]]}")


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


if __name__ == "__main__":
    main()
