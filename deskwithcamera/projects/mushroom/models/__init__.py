"""Card classification models, selectable by number.

    1  HSV histogram + Linear SVM          (color only, explainable)
    2  MobileNetV2 embedding + Logistic reg. (shape + color, needs PyTorch)
    3  HOG + Linear SVM                    (shape only, control)

    from projects.mushroom.models import get_model, list_models
    model = get_model(1)
    model.fit(images, labels)              # images: BGR uint8, labels: "poisonous" / "safe"
    label, score = model.predict(image)    # score > 0 = poisonous, size = confidence

Imports are lazy so that a missing PyTorch only matters when model 2 is really used.
"""
import importlib

_REGISTRY = {
    1: ("projects.mushroom.models.hsv_svm", "HsvSvm"),
    2: ("projects.mushroom.models.mobilenet_lr", "MobileNetLogReg"),
    3: ("projects.mushroom.models.hog_svm", "HogSvm"),
}


def get_model(model_id):
    if model_id not in _REGISTRY:
        raise ValueError(f"Unknown model {model_id}. Available: {sorted(_REGISTRY)}")
    mod, cls = _REGISTRY[model_id]
    return getattr(importlib.import_module(mod), cls)()


def list_models():
    """[(id, name, description)] without loading heavy libraries."""
    out = []
    for i, (mod, cls) in sorted(_REGISTRY.items()):
        c = getattr(importlib.import_module(mod), cls)
        out.append((i, c.name, c.description))
    return out
