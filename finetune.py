"""
Paper experiments: fine-tune the two main variants from the same pretrained ViT on
CUB-200, FGVC-Aircraft, Flowers-102 and Food-101.

    loopvit   vanilla LoopViT (baseline)
    lcga      LCGA-logprob (proposed)

Both start from identical weights (the loop core is built from the pretrained ViT's
blocks, and gamma starts at 0 -- see pretrained.py), so any difference comes from LCGA.
The shared recipe is finetune.yaml; per-dataset settings are in RECIPES below.

    # 1. get the data (datasets/<name>/{train,test}/<class>/*.jpg)
    python downloads.py --dataset cub200 aircraft flowers102 food101

    # 2. run everything (4 datasets x 2 variants, one after another)
    python finetune.py

    # a subset, a different backbone, or a dry run that only prints the commands
    python finetune.py --datasets cub200 aircraft --variants lcga
    python finetune.py --pretrained vit_small_patch14_dinov2.lvd142m
    python finetune.py --dry-run

    # anything after the known flags is passed straight to train.py
    python finetune.py --datasets cub200 --image-size 448 --batch-size 8

Every run uses `--resume auto`, so re-running the same command continues
unfinished runs and skips finished ones. Runs go to
runs/<variant>_<dataset>_<backbone>/ (log.csv, best.pt, run_history.log, ...).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from data import FALLBACK_ROOTS

VARIANTS = ("loopvit", "lcga")
DATASETS = ("cub200", "aircraft", "flowers102", "food101")

# per-dataset settings on top of finetune.yaml
RECIPES = {
    "cub200":     {"epochs": 50},                        # 5,994 train / 200 classes
    "aircraft":   {"epochs": 50},                        # 6,667 train / 100 classes
    "flowers102": {"epochs": 50},                        # 2,040 train / 102 classes
    "food101":    {"epochs": 15, "warmup-epochs": 1,     # 75,750 train / 101 classes
                   "patience": 5, "save-every-steps": 1000},
}


def dataset_present(name: str, data_root: str) -> bool:
    roots = [data_root] + [r for r in FALLBACK_ROOTS if r != data_root]
    return any(os.path.isdir(os.path.join(r, name, "train")) for r in roots)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS),
                   help=f"default: {' '.join(DATASETS)} (any dataset folder name works)")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--pretrained", default=None, help="timm ViT; default: the one in finetune.yaml")
    p.add_argument("--config", default="finetune.yaml")
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--output-root", default="runs")
    p.add_argument("--download", action="store_true", help="run downloads.py for missing datasets")
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args, extra = p.parse_known_args()

    pretrained = args.pretrained
    if pretrained is None:
        import yaml
        with open(args.config) as f:
            pretrained = (yaml.safe_load(f) or {}).get("pretrained")
    if not pretrained:
        raise SystemExit(f"no pretrained model: set `pretrained` in {args.config} or pass --pretrained")
    backbone = pretrained.split(".")[0]

    missing = [d for d in args.datasets if not dataset_present(d, args.data_root)]
    if missing and not args.dry_run:
        if args.download:
            subprocess.run([sys.executable, "downloads.py", "--dataset", *missing,
                            "--root", args.data_root], check=False)
        still = [d for d in missing if not dataset_present(d, args.data_root)]
        if still:
            raise SystemExit(f"dataset(s) not found: {still}\n"
                             f"  python downloads.py --dataset {' '.join(still)}\n"
                             f"or re-run with --download")

    jobs = []
    for ds in args.datasets:
        for variant in args.variants:
            out = os.path.join(args.output_root, f"{variant}_{ds}_{backbone}")
            cmd = [sys.executable, "train.py", "--config", args.config, "--dataset", ds,
                   "--data-root", args.data_root, "--variant", variant,
                   "--pretrained", pretrained, "--resume", "auto", "--output-dir", out]
            for k, v in RECIPES.get(ds, {}).items():
                cmd += [f"--{k}", str(v)]
            jobs.append((f"{variant} / {ds}", cmd + extra))   # extra last: it wins

    status = []
    for i, (name, cmd) in enumerate(jobs, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(jobs)}] {name}\n  "
              + subprocess.list2cmdline(cmd) + f"\n{'=' * 78}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        status.append((name, rc, time.time() - t0))
        if rc == 130:                                  # Ctrl+C: last.pt was saved
            print("\n[finetune] interrupted; re-run the same command to continue.")
            break

    if status:
        print(f"\n{'=' * 78}\nsummary")
        for name, rc, sec in status:
            print(f"  {name:<28} {'ok' if rc == 0 else f'exit {rc}':<8} {sec / 60:6.1f} min")
        print("results: log.csv / best.pt in each run folder; compare with\n"
              "  python analyze_lcga.py --ckpt runs/loopvit_<ds>_<bb>/best.pt "
              "runs/lcga_<ds>_<bb>/best.pt --labels vanilla lcga --out-dir analysis/<ds>")


if __name__ == "__main__":
    main()
