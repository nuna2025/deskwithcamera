"""Model 2: MobileNetV2 embedding + logistic regression.

A MobileNetV2 trained on ImageNet turns each card into 1280 numbers (the network is NOT retrained).
Only the small logistic regression is trained on your 8-20 cards. Understands shapes and texture as well
as colors, but is not explainable and needs PyTorch + the pretrained weights (about 14 MB).

Weights: torchvision downloads them automatically on first use. If the computer is offline, download
    https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
once and save it as  projects/mushroom/data/weights/mobilenet_v2.pth  (it is picked up automatically).
"""
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from projects.mushroom.models.common import CardModel

WEIGHTS_FILE = Path(__file__).resolve().parent.parent / "data" / "weights" / "mobilenet_v2.pth"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class MobileNetLogReg(CardModel):
    id, key = 2, "mobilenet_lr"
    name = "MobileNetV2 embedding + Logistic regression"
    description = "Shapes + colors, 1280 features. Needs PyTorch and pretrained weights."

    _net = None             # shared between instances (loading takes a moment)

    @classmethod
    def _load(cls):
        if cls._net is not None:
            return cls._net
        try:
            import torch
            from torchvision.models import mobilenet_v2
        except ImportError as e:
            raise RuntimeError("Model 2 needs PyTorch: pip install torch torchvision") from e
        if WEIGHTS_FILE.exists():
            net = mobilenet_v2(weights=None)
            net.load_state_dict(torch.load(WEIGHTS_FILE, map_location="cpu"))
        else:
            from torchvision.models import MobileNet_V2_Weights
            try:
                net = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            except Exception as e:
                raise RuntimeError(
                    "Could not download the MobileNetV2 weights. Download "
                    "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth and save it as "
                    f"{WEIGHTS_FILE}") from e
        cls._net = net.features.eval()
        return cls._net

    def extract(self, img):
        import torch
        net = self._load()
        rgb = cv2.cvtColor(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        with torch.no_grad():
            return net(torch.from_numpy(x)).mean(dim=(2, 3)).squeeze(0).numpy()

    def make_classifier(self):
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
