"""Model 1: HSV color histogram + linear SVM.

12 numbers per card: how much of the card is red, orange, yellow, ... (white paper, black outline and
pale/gray pixels are ignored). The model can only learn color rules, but it is tiny, fast and
explainable: the learned weight per hue shows what it considers poisonous.
"""
import cv2
import numpy as np
from sklearn.svm import LinearSVC

from projects.mushroom.models.common import CardModel

BINS = 12
S_MIN, V_MIN = 60, 60               # ignore pale / dark pixels
HUE_NAMES = ["red", "orange", "yellow", "yellow-green", "green", "green", "cyan", "blue", "blue",
             "violet", "magenta", "red-pink"]


class HsvSvm(CardModel):
    id, key = 1, "hsv_svm"
    name = "HSV histogram + Linear SVM"
    description = "Color only. Tiny, fast, explainable (hue weights)."

    def extract(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = (hsv[..., 0].astype(int) + 8) % 180          # rotate so red sits in the middle of a bin
        mask = (hsv[..., 1] >= S_MIN) & (hsv[..., 2] >= V_MIN)
        hist = np.bincount(hue[mask] * BINS // 180, minlength=BINS)
        return hist / hue.size * 8.0

    def make_classifier(self):
        return LinearSVC(C=1.0, max_iter=20000)

    def explain(self):
        w = self.clf.coef_[0]
        lines = []
        for b in np.argsort(-np.abs(w)):
            if abs(w[b]) < 0.05:
                continue
            hue = (b * 180 / BINS + 90 / BINS - 8) % 180
            lines.append(f"  hue ~{hue:5.1f} ({HUE_NAMES[b]:>12s}): {w[b]:+.2f}  ({'poisonous' if w[b] > 0 else 'safe'})")
        return "Learned color weights (+ = poisonous):\n" + ("\n".join(lines) if lines else "  (nothing significant)")
