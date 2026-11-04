"""Model 3: HOG + linear SVM (control / baseline).

HOG describes the direction of edges (cap shape, stem, dots) and ignores color. In earlier tests on
simple cartoon cards it was weak (about chance level with 8-10 training cards) and it is NOT
rotation invariant. It is included to show what color-free, shape-only features achieve.
"""
import cv2
import numpy as np
from sklearn.svm import LinearSVC

from projects.mushroom.models.common import CardModel

_HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


class HogSvm(CardModel):
    id, key = 3, "hog_svm"
    name = "HOG + Linear SVM (control)"
    description = "Edge directions only, no color. Baseline for comparison."

    def extract(self, img):
        gray = cv2.cvtColor(cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return _HOG.compute(gray).ravel()

    def make_classifier(self):
        return LinearSVC(C=1.0, max_iter=20000)
