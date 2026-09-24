"""
Classify images with a trained LoopViT / LCGA checkpoint.

    python predict.py --ckpt runs/lcga_plantodc/best.pt --images a.jpg b.png
    python predict.py --ckpt runs/lcga_plantodc/best.pt --images some_folder/ --loop-steps 6
    python predict.py --ckpt runs/lcga_plantodc/best.pt --images some_folder/ --dynamic-exit
    python predict.py --ckpt runs/lcga_plantodc/best.pt --images some_folder/ --dynamic-exit \
                      --exit-mode both --exit-tau 0.1 --exit-graph-eps 0.005
    python predict.py --ckpt runs/lcga_plantodc/best.pt --images a.jpg --per-step

`--loop-steps` runs the weight-tied core more (or fewer) times than it was
trained with; step embeddings (and per-step gammas) past the training budget
reuse the last one (identity extrapolation), exactly as in the paper.
`--per-step` prints, for every iteration, the prediction, its entropy and the
CLS attention-graph change Delta_t (eq. A4).
"""
import argparse
import os

import torch
from PIL import Image

from data import build_transforms, long_path
from loop_vit import LoopViT, LoopViTConfig, graph_delta

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", nargs="+", required=True, help="files and/or folders")
    p.add_argument("--loop-steps", type=int, default=None, help="override iterations at inference")
    p.add_argument("--dynamic-exit", action="store_true", help="halt per sample on the exit rule")
    p.add_argument("--exit-mode", choices=["entropy", "graph", "both", "either"], default=None)
    p.add_argument("--exit-tau", type=float, default=None, help="entropy threshold in nats")
    p.add_argument("--exit-graph-eps", type=float, default=None, help="graph-change threshold in nats")
    p.add_argument("--per-step", action="store_true", help="also print the prediction after every iteration")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = LoopViTConfig(**ckpt["model_cfg"])
    model = LoopViT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    classes = ckpt["classes"]
    _, tf = build_transforms(cfg.image_size)

    paths = []
    for item in args.images:
        if os.path.isdir(item):
            paths += sorted(os.path.join(item, f) for f in sorted(os.listdir(item))
                            if f.lower().endswith(EXTS))
        else:
            paths.append(item)
    if not paths:
        raise SystemExit("no images found")

    k = min(args.topk, len(classes))
    steps = args.loop_steps or cfg.loop_steps
    print(f"[predict] {len(paths)} image(s), attn_carry={cfg.attn_carry}, {steps} loop step(s)"
          f"{', dynamic exit (' + (args.exit_mode or cfg.exit_mode) + ')' if args.dynamic_exit else ''}"
          f", device {device}")

    total_apps = 0
    with torch.no_grad():
        for i in range(0, len(paths), args.batch_size):
            chunk = paths[i:i + args.batch_size]
            x = torch.stack([tf(Image.open(long_path(pth)).convert("RGB")) for pth in chunk]).to(device)

            if args.dynamic_exit:
                out = model.dynamic_forward(x, tau=args.exit_tau, max_steps=steps,
                                            exit_mode=args.exit_mode, graph_eps=args.exit_graph_eps)
                probs, extra = out["logits"].softmax(-1), out["exit_steps"].tolist()
                total_apps += out["block_apps"]
            else:
                probs, extra = model(x, num_steps=steps).softmax(-1), None

            per_step = graphs = None
            if args.per_step:
                if cfg.num_cls_tokens >= 1:
                    per_step, graphs = model.logits_and_graphs(x, num_steps=steps)
                else:
                    per_step = model.logits_per_step(x, num_steps=steps)
            for j, pth in enumerate(chunk):
                top = probs[j].topk(k)
                preds = ", ".join(f"{classes[c]} {v:.3f}"
                                  for v, c in zip(top.values.tolist(), top.indices.tolist()))
                suffix = f"  [exited at step {extra[j]}]" if extra else ""
                print(f"{pth}: {preds}{suffix}")
                if per_step:
                    for t, logits in enumerate(per_step):
                        pr = logits[j].softmax(-1)
                        ent = model.entropy(logits[j:j + 1]).item()
                        c = int(pr.argmax())
                        d = (f", graph delta {graph_delta(graphs[t][j:j + 1], graphs[t - 1][j:j + 1]).item():.5f}"
                             if graphs is not None and t >= 1 else "")
                        print(f"    step {t + 1}: {classes[c]} {pr[c]:.3f} (entropy {ent:.3f}{d})")
    if args.dynamic_exit:
        print(f"[predict] block applications: {total_apps} total, "
              f"{total_apps / len(paths):.2f} per image (fixed depth: {steps * cfg.core_depth})")


if __name__ == "__main__":
    main()
