"""Shared helpers for all card models.

Convention: every image is a BGR uint8 array (what cv2.imread returns), normally 256x256.
Labels are the folder names used by the dataset: "poisonous" / "safe".
"""
import numpy as np

LABELS = ("safe", "poisonous")          # index 0 = safe (-), index 1 = poisonous (+)


def label_to_int(label):
    return LABELS.index(label)


def white_normalize(img, ring_frac=0.06, target=235):
    """Color-cast correction using the card's own paper margin.

    The outer ring of a card is paper (white in reality). The camera often shows it tinted
    (amber under warm light). The median color of that ring is scaled to a neutral gray, and the
    same scaling is applied to the whole card. Pure white padding added by the saver (255,255,255)
    is ignored. If no usable ring is found the image is returned unchanged.
    """
    h, w = img.shape[:2]
    m = max(int(ring_frac * min(h, w)), 2)
    ring = np.ones((h, w), bool)
    ring[m:h - m, m:w - m] = False
    px = img[ring].reshape(-1, 3).astype(np.float32)
    px = px[~(px >= 250).all(axis=1)]                   # drop letterbox padding
    px = px[px.max(axis=1) > 90]                        # drop dark pixels (card art touching the edge)
    if len(px) < 30:
        return img
    paper = np.median(px, axis=0)
    gain = np.clip(target / np.maximum(paper, 1.0), 0.5, 2.5)
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


class CardModel:
    """Base class. A model = feature extractor + small classifier.

    extract(img) -> 1-D feature vector       (can be slow, e.g. a neural net)
    make_classifier() -> sklearn estimator    (fast; retrained for every experiment fold)
    """
    id = 0
    key = ""
    name = ""
    description = ""
    white_balance = True            # apply white_normalize() before extracting features

    def __init__(self):
        self.clf = None

    # --- to implement -------------------------------------------------------
    def extract(self, img):
        raise NotImplementedError

    def make_classifier(self):
        raise NotImplementedError

    # --- shared ---------------------------------------------------------------
    def prepare(self, img):
        return white_normalize(img) if self.white_balance else img

    def features(self, img):
        return np.asarray(self.extract(self.prepare(img)), dtype=np.float32).ravel()

    def fit_features(self, X, y):
        self.clf = self.make_classifier().fit(np.asarray(X), np.asarray(y))
        return self

    def fit(self, images, labels):
        X = [self.features(i) for i in images]
        return self.fit_features(X, [label_to_int(l) for l in labels])

    def score_features(self, x):
        """Signed score: > 0 means poisonous. The size is the confidence (margin)."""
        return float(self.clf.decision_function(np.asarray(x).reshape(1, -1))[0])

    def predict(self, img):
        s = self.score_features(self.features(img))
        return LABELS[int(s > 0)], s

    def explain(self):
        """Optional: short text about what the trained model looks at."""
        return None
