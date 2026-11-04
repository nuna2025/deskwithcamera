"""Compare the card models on the saved cards.

Run from the project root (the folder with main.py):

    python -m projects.mushroom.experiment              # compare all models side by side
    python -m projects.mushroom.experiment --model 1    # only model number 1, 2 or 3
    python -m projects.mushroom.experiment --list       # show the models
    python -m projects.mushroom.experiment --data path/to/sessions

How it tests (important with few cards):
  * Leave-one-out: each card is classified by a model trained on all OTHER cards.
  * Perturbations: the held-out card is also shown darker, brighter, warmer, cooler, rotated 90 deg and
    noisy. This shows how well the model survives a different lighting / placement than in training.
With only a handful of cards the percentages jump in big steps (1 card = 6-12 %); read them as a hint,
not as a measurement.
"""
import argparse
import sys

import cv2
import numpy as np

from projects.mushroom.dataset import load_cards
from projects.mushroom.models import get_model, list_models
from projects.mushroom.models.common import label_to_int


def _scale(img, b=1.0, g=1.0, r=1.0):
    return np.clip(img.astype(np.float32) * np.array([b, g, r], np.float32), 0, 255).astype(np.uint8)


PERTURBATIONS = {
    "original": lambda im: im,
    "dark": lambda im: _scale(im, 0.55, 0.55, 0.55),
    "bright": lambda im: _scale(im, 1.3, 1.3, 1.3),
    "warm": lambda im: _scale(im, 0.8, 0.95, 1.15),
    "cool": lambda im: _scale(im, 1.15, 1.0, 0.85),
    "rot90": lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
    "noise": lambda im: np.clip(im.astype(np.float32) + np.random.default_rng(0).normal(0, 12, im.shape),
                                0, 255).astype(np.uint8),
}


def evaluate(model, images, labels):
    """Leave-one-out with perturbations. Returns {perturbation: accuracy} and the per-card results."""
    y = np.array([label_to_int(l) for l in labels])
    base = [model.features(im) for im in images]
    pert = {n: [model.features(f(im)) for im in images] for n, f in PERTURBATIONS.items() if n != "original"}
    pert["original"] = base
    hits = {n: [] for n in PERTURBATIONS}
    for i in range(len(images)):
        idx = [j for j in range(len(images)) if j != i]
        if len(set(y[idx])) < 2:
            continue
        model.fit_features([base[j] for j in idx], y[idx])
        for n in PERTURBATIONS:
            hits[n].append(int(model.score_features(pert[n][i]) > 0) == y[i])
    acc = {n: float(np.mean(h)) if h else float("nan") for n, h in hits.items()}
    return acc, hits["original"]


def run_model(model_id, images, labels, names):
    model = get_model(model_id)
    print(f"\n=== Model {model.id}: {model.name} ===")
    print(model.description)
    acc, ok = evaluate(model, images, labels)
    wrong = [n for n, o in zip(names, ok) if not o]
    print("Leave-one-out accuracy (original):  %.0f %%  (%d of %d cards)" % (100 * acc["original"], sum(ok), len(ok)))
    if wrong:
        print("  wrongly classified:", ", ".join(wrong))
    # train on everything for the explanation
    model.fit(images, labels)
    text = model.explain()
    if text:
        print(text)
    return model, acc


def choose(arg):
    """No argument = compare all models. A number (1, 2, 3) runs just that one."""
    if arg in (None, "all"):
        return [m[0] for m in list_models()]
    return [int(arg)]


def main():
    ap = argparse.ArgumentParser(description="Compare mushroom card models on saved cards.")
    ap.add_argument("--model", help="1, 2 or 3 (default: all)")
    ap.add_argument("--data", help="data folder (default: projects/mushroom/data)")
    ap.add_argument("--list", action="store_true", help="list models and exit")
    args = ap.parse_args()

    if args.list:
        for i, name, desc in list_models():
            print(f"{i}  {name}  -  {desc}")
        return
    ids = choose(args.model)

    images, labels, names = load_cards(args.data)
    n_s, n_p = labels.count("safe"), labels.count("poisonous")
    print(f"Loaded {len(images)} cards: {n_s} safe (-), {n_p} poisonous (+)")
    if min(n_s, n_p) < 2:
        sys.exit("Need at least 2 cards of each kind. Save more sessions first (S key in the capture module).")

    results = {}
    for mid in ids:
        try:
            _, results[mid] = run_model(mid, images, labels, names)
        except Exception as e:                       # e.g. PyTorch missing for model 2
            print(f"\n=== Model {mid} skipped: {e}")

    if len(results) > 1:
        cols = list(PERTURBATIONS)
        print("\nSummary (leave-one-out accuracy in %)")
        print("model".ljust(8) + "".join(c.rjust(10) for c in cols))
        for mid, acc in results.items():
            print(str(mid).ljust(8) + "".join(f"{100 * acc[c]:9.0f} " for c in cols))


if __name__ == "__main__":
    main()
