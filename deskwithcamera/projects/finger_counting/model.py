"""The trained way of counting: a small classifier learns 'how many fingers' (0-5 per hand) from your own recordings.

Recordings (hand features + the number you were asked to show) are collected over time in data/samples.npz,
the model is trained on all of them and saved in data/model.joblib.
"""
from pathlib import Path

import numpy as np

from projects.finger_counting.hands import FEATURE_N

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
MIN_PER_CLASS = 16                # (every recording is stored twice: original + mirror image)


class FingerModel:
    def __init__(self, folder=DATA_DIR):
        self.folder = Path(folder)
        self.clf = None
        self.kind = "count"           # "count": old model (number 0-5) | "bits": per-finger model made by train.py
        self.info = ""                # e.g. "432 samples, accuracy 0.97"

    @property
    def ready(self):
        return self.clf is not None

    # --- recordings -----------------------------------------------------------
    def _samples_file(self):
        return self.folder / "samples.npz"

    def load_samples(self):
        f = self._samples_file()
        if not f.exists():
            return np.zeros((0, FEATURE_N), np.float32), np.zeros(0, int)
        d = np.load(f)
        if d["X"].shape[1] != FEATURE_N:                    # recordings of an older version (other features): cannot be used
            print("[fingers] old recordings (other format) are ignored: please record again")
            return np.zeros((0, FEATURE_N), np.float32), np.zeros(0, int)
        return d["X"], d["y"]

    def add_samples(self, X, y):
        if not len(X):
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        X0, y0 = self.load_samples()
        np.savez(self._samples_file(), X=np.vstack([X0, np.asarray(X, np.float32)]), y=np.concatenate([y0, np.asarray(y, int)]))

    def counts(self):
        _, y = self.load_samples()
        return {int(k): int((y == k).sum()) for k in range(6)}

    # --- model --------------------------------------------------------------------
    def load(self):
        f = self.folder / "model.joblib"
        if not f.exists():
            return False
        try:
            import joblib
            d = joblib.load(f)
            if d.get("feature_n", getattr(d["clf"], "n_features_in_", FEATURE_N)) != FEATURE_N:
                print("[fingers] the saved model is from an older version: please train again")
                return False
            self.clf, self.info, self.kind = d["clf"], d.get("info", ""), d.get("kind", "count")
            return True
        except Exception as e:
            print(f"[fingers] could not load the model: {e}")
            return False

    def train(self):
        """Train on every recording. Returns (ok, message)."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import GroupKFold, cross_val_score
            import joblib
        except ImportError:
            return False, "scikit-learn missing: pip install scikit-learn"
        X, y = self.load_samples()
        per = [int((y == k).sum()) for k in np.unique(y)]
        if len(per) < 2 or min(per) < MIN_PER_CLASS:
            return False, "not enough recordings yet"
        clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        try:
            groups = np.arange(len(y)) // 2                 # a recording and its mirror image always stay together
            acc = float(cross_val_score(clf, X, y, cv=GroupKFold(n_splits=min(5, min(per) // 2)), groups=groups).mean())
        except Exception:
            acc = float("nan")
        clf.fit(X, y)
        self.clf = clf
        self.kind = "count"
        self.info = f"{len(y)} recordings, accuracy {acc:.2f}"
        self.folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": clf, "info": self.info}, self.folder / "model.joblib")
        print(f"[fingers] model trained: {self.info}")
        return True, self.info

    def predict_bits(self, feat):
        """[thumb, index, middle, ring, pinky] as True / False (only for models made by train.py), else None."""
        if self.kind != "bits":
            return None
        row = np.asarray(self.clf.predict(np.asarray(feat, np.float32).reshape(1, -1))).reshape(-1)
        return [bool(v) for v in row]

    def predict(self, feat):
        bits = self.predict_bits(feat)
        if bits is not None:
            return int(sum(bits))
        return int(self.clf.predict(np.asarray(feat, np.float32).reshape(1, -1))[0])
