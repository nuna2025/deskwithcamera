"""Learning in the background + classifying single cards (used by the Classification module).

    learner = Learner()
    learner.start(images, labels, model_id)      # runs in a thread, the screen keeps animating
    learner.state                                # "idle" | "training" | "ready" | "error"
    side, confidence = learner.classify(card_image)    # side: "+" (positive) or "-" (negative), confidence 0.5 .. 1.0

Labels are the class names on disk: "poisonous" = positive (+), "safe" = negative (-).
If the chosen model cannot be loaded (e.g. PyTorch is missing for model 2) model 1 is used instead.
"""
import math
import threading
import traceback

from projects.mushroom.models import get_model

FALLBACK_MODEL_ID = 1


class Learner:
    def __init__(self):
        self.model = None
        self.model_id = None
        self.state = "idle"
        self.error = ""
        self.note = ""                  # e.g. "model 2 not available, using model 1"
        self._thread = None

    def start(self, images, labels, model_id):
        self.state, self.error, self.note = "training", "", ""
        self._thread = threading.Thread(target=self._run, args=(list(images), list(labels), model_id), daemon=True)
        self._thread.start()

    def _run(self, images, labels, model_id):
        try:
            try:
                model = get_model(model_id)
                model.fit(images, labels)
                used = model_id
            except Exception:                                  # PyTorch / weights missing ...
                if model_id == FALLBACK_MODEL_ID:
                    raise
                traceback.print_exc()
                model = get_model(FALLBACK_MODEL_ID)
                model.fit(images, labels)
                used = FALLBACK_MODEL_ID
                self.note = f"Model {model_id} is not available: using model {FALLBACK_MODEL_ID}"
            self.model, self.model_id = model, used
            self.state = "ready"
        except Exception as e:
            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"

    def classify(self, card_image):
        """-> ("+" or "-", confidence in 0.5 .. 1.0). Needs state == "ready"."""
        label, score = self.model.predict(card_image)
        return ("+" if score > 0 else "-"), 1.0 / (1.0 + math.exp(-abs(score)))
