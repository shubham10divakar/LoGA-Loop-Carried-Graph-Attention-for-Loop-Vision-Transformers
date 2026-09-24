"""
data.py: plant augmentation, SMOTE and dropping classes without validation images.

    python -m pytest tests -q
"""
from __future__ import annotations

import os
import pickle

import torch
from PIL import Image

from data import SmotePair, build_dataloaders, build_transforms, smote_samples


def make_tree(root, counts, val_counts, size=20):
    """root/train/<c>/*.png and root/test/<c>/*.png with a distinct colour per image."""
    for split, cnt in (("train", counts), ("test", val_counts)):
        for ci, (c, n) in enumerate(cnt.items()):
            d = os.path.join(root, split, c)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                Image.new("RGB", (size, size), (ci * 60 % 256, i * 10 % 256, 100)).save(
                    os.path.join(d, f"{i}.png"))
    return os.path.join(root, "train"), os.path.join(root, "test")


def test_smote_balances_to_max_and_is_deterministic(tmp_path):
    tr, _ = make_tree(str(tmp_path), {"a": 10, "b": 3, "c": 6}, {})
    by = {k: sorted(os.path.join(tr, c, f) for f in os.listdir(os.path.join(tr, c)))
          for k, c in enumerate("abc")}
    out = smote_samples(by, "max", k=2, seed=0, size=16)
    per = {k: sum(1 for _, y in out if y == k) for k in range(3)}
    assert per == {0: 0, 1: 7, 2: 4}
    again = smote_samples(by, "max", k=2, seed=0, size=16)
    assert [(p.a, p.b, p.lam) for p, _ in out] == [(p.a, p.b, p.lam) for p, _ in again]
    for p, y in out:                          # neighbours come from the same class
        assert os.path.dirname(p.a) == os.path.dirname(p.b) == os.path.join(tr, "abc"[y])
        assert p.a != p.b and 0 <= p.lam < 1
    assert sum(1 for _, y in smote_samples(by, 5, 2, 0, 16) if y == 1) == 2


def test_smote_pair_blends_and_pickles(tmp_path):
    a, b = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    Image.new("RGB", (8, 8), (0, 0, 0)).save(a)
    Image.new("RGB", (30, 10), (200, 100, 50)).save(b)
    img = pickle.loads(pickle.dumps(SmotePair(a, b, 0.5, 12))).load()
    assert img.size == (12, 12) and img.getpixel((5, 5)) == (100, 50, 25)


def test_loader_with_smote_and_class_drop(tmp_path):
    tr, va = make_tree(str(tmp_path), {"a": 8, "b": 3, "c": 4}, {"a": 2, "b": 2})
    train, val, classes = build_dataloaders(
        tr, va, image_size=16, batch_size=4, num_workers=0, augment="plant",
        pin_memory=False, drop_classes_without_val=True, smote=True, smote_k=2)
    assert classes == ["a", "b"]                   # "c" has no test images
    assert len(train.dataset) == 16 and len(val.dataset) == 4
    x, y = next(iter(train))
    assert x.shape == (4, 3, 16, 16) and torch.isfinite(x).all()


def test_plant_augment_runs():
    tf, _ = build_transforms(32, "plant")
    assert tf(Image.new("RGB", (50, 40), (10, 200, 30))).shape == (3, 32, 32)
