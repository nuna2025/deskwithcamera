"""Compare the card models on the saved cards.

Run from the project root (the folder with main.py):

    python -m projects.mushroom.experiment --sessions            # list saved sessions (what is in each)
    python -m projects.mushroom.experiment                       # all models, leave-one-out on ALL cards
    python -m projects.mushroom.experiment --model 1             # only model 1, 2 or 3

    # Separate training and test cards (recommended when you have extra cards):
    python -m projects.mushroom.experiment --train 20261104_2105 --test 20261104_2120 20261104_2130
    python -m projects.mushroom.experiment --test 20261104_2120 20261104_2130    # train = all other sessions

A session name can be shortened to a prefix. Sessions listed under --test are NEVER used for training.

Two ways of testing:
  * Train/test split (--test given): the models learn from the training sessions only and are then
    scored on the test cards they have never seen. This is the realistic demo situation.
  * Leave-one-out (no --test): each card is classified by a model trained on all OTHER cards.
    Use this when you have just one small set.
In both cases the test cards are also shown darker, brighter, warmer, cooler, rotated 90 deg and noisy
to see how well a model survives a different lighting / placement.
"""
import argparse
import sys

import cv2
import numpy as np

from projects.mushroom.dataset import list_sessions, load_cards
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


def _features(model, images):
    """Features of the original images and of every perturbed version."""
    return {n: [model.features(f(im)) for im in images] for n, f in PERTURBATIONS.items()}


def evaluate_split(model, train_imgs, train_labels, test_imgs, test_labels):
    """Train on the training cards, score the test cards. Returns {perturbation: accuracy} and per-card results."""
    ytr = [label_to_int(l) for l in train_labels]
    yte = np.array([label_to_int(l) for l in test_labels])
    model.fit_features([model.features(im) for im in train_imgs], ytr)
    feats = _features(model, test_imgs)
    acc, scores = {}, {}
    for n, fs in feats.items():
        s = np.array([model.score_features(f) for f in fs])
        acc[n] = float(np.mean((s > 0).astype(int) == yte))
        scores[n] = s
    return acc, scores["original"]


def evaluate_loo(model, images, labels):
    """Leave-one-out: every card is judged by a model trained on all the other cards."""
    y = np.array([label_to_int(l) for l in labels])
    feats = _features(model, images)
    hits = {n: [] for n in PERTURBATIONS}
    scores = np.full(len(images), np.nan)
    for i in range(len(images)):
        idx = [j for j in range(len(images)) if j != i]
        if len(set(y[idx])) < 2:
            continue
        model.fit_features([feats["original"][j] for j in idx], y[idx])
        for n in PERTURBATIONS:
            s = model.score_features(feats[n][i])
            hits[n].append(int(s > 0) == y[i])
            if n == "original":
                scores[i] = s
    return {n: float(np.mean(h)) if h else float("nan") for n, h in hits.items()}, scores


def run_model(model_id, split, test_mode):
    tr_i, tr_l, te_i, te_l, te_n = split
    model = get_model(model_id)
    print(f"\n=== Model {model.id}: {model.name} ===")
    print(model.description)
    if test_mode:
        acc, scores = evaluate_split(model, tr_i, tr_l, te_i, te_l)
        what = "Test accuracy (unseen cards)"
    else:
        acc, scores = evaluate_loo(model, te_i, te_l)
        what = "Leave-one-out accuracy"
    pred = scores > 0
    truth = np.array([l == "poisonous" for l in te_l])
    ok = pred == truth
    print("%s (original): %.0f %%  (%d of %d cards)" % (what, 100 * acc["original"], ok.sum(), len(ok)))
    print("  safe -> wrongly called poisonous: %d of %d    poisonous -> wrongly called safe: %d of %d" % (
        (~truth & pred).sum(), (~truth).sum(), (truth & ~pred).sum(), truth.sum()))
    wrong = [f"{n} (score {s:+.2f})" for n, o, s in zip(te_n, ok, scores) if not o]
    if wrong:
        print("  wrongly classified:\n    " + "\n    ".join(wrong))
    # train on everything that is allowed (training cards, or all cards in leave-one-out) for the explanation
    if test_mode:
        model.fit(tr_i, tr_l)
    else:
        model.fit(te_i, te_l)
    text = model.explain()
    if text:
        print(text)
    return acc


def main():
    ap = argparse.ArgumentParser(description="Compare mushroom card models on saved cards.")
    ap.add_argument("--model", help="1, 2 or 3 (default: all)")
    ap.add_argument("--data", help="data folder (default: projects/mushroom/data)")
    ap.add_argument("--train", nargs="+", metavar="SESSION", help="session(s) used for training")
    ap.add_argument("--test", nargs="+", metavar="SESSION", help="session(s) used ONLY for testing")
    ap.add_argument("--sessions", action="store_true", help="list saved sessions and exit")
    ap.add_argument("--list", action="store_true", help="list models and exit")
    args = ap.parse_args()

    if args.list:
        for i, name, desc in list_models():
            print(f"{i}  {name}  -  {desc}")
        return
    if args.sessions:
        rows = list_sessions(args.data)
        print("session".ljust(24) + "safe(-)  poisonous(+)")
        for name, s, p in rows:
            print(name.ljust(24) + f"{s:6d}  {p:11d}")
        print(f"total: {sum(r[1] for r in rows)} safe, {sum(r[2] for r in rows)} poisonous")
        return

    ids = [m[0] for m in list_models()] if args.model in (None, "all") else [int(args.model)]
    test_mode = bool(args.test)
    if args.train and not args.test:
        sys.exit("--train needs --test as well (without --test the program uses leave-one-out on all cards).")

    if test_mode:
        te_i, te_l, te_n = load_cards(args.data, args.test)
        if args.train:
            tr_i, tr_l, _ = load_cards(args.data, args.train)
        else:                                                    # everything that is not a test session
            all_i, all_l, all_n = load_cards(args.data)
            keep = [k for k, n in enumerate(all_n) if not any(n.startswith(t) for t in args.test)]
            tr_i, tr_l = [all_i[k] for k in keep], [all_l[k] for k in keep]
        overlap = set(args.train or []) & set(args.test)
        if overlap:
            sys.exit(f"The same session is in --train and --test: {', '.join(sorted(overlap))}")
        print(f"Training cards: {len(tr_i)}  ({tr_l.count('safe')} safe, {tr_l.count('poisonous')} poisonous)")
        print(f"Test cards:     {len(te_i)}  ({te_l.count('safe')} safe, {te_l.count('poisonous')} poisonous)")
        if min(tr_l.count("safe"), tr_l.count("poisonous")) < 2:
            sys.exit("Need at least 2 training cards of each kind.")
        if not te_i:
            sys.exit("No test cards found. Check the session names with --sessions.")
    else:
        te_i, te_l, te_n = load_cards(args.data)
        tr_i, tr_l = [], []
        print(f"Loaded {len(te_i)} cards: {te_l.count('safe')} safe (-), {te_l.count('poisonous')} poisonous (+)  [leave-one-out]")
        if min(te_l.count("safe"), te_l.count("poisonous")) < 2:
            sys.exit("Need at least 2 cards of each kind. Save more sessions first (S key in the capture module).")

    results = {}
    for mid in ids:
        try:
            results[mid] = run_model(mid, (tr_i, tr_l, te_i, te_l, te_n), test_mode)
        except Exception as e:                       # e.g. PyTorch missing for model 2
            print(f"\n=== Model {mid} skipped: {e}")

    if len(results) > 1:
        cols = list(PERTURBATIONS)
        print("\nSummary (accuracy in %)")
        print("model".ljust(8) + "".join(c.rjust(10) for c in cols))
        for mid, acc in results.items():
            print(str(mid).ljust(8) + "".join(f"{100 * acc[c]:9.0f} " for c in cols))


if __name__ == "__main__":
    main()