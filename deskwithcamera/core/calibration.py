"""Store and apply the calibration result (camera pixels <-> coordinates inside the frame).

H is a 3x3 matrix that maps "camera pixel -> normalized frame coordinates (0..1, 0..1)".
Because it is normalized, it keeps working when the projector resolution changes.
  (0, 0) = top-left inside the frame,  (1, 1) = bottom-right inside the frame

How other modules use it:
    u, v = app.calib.cam_to_norm(x_cam, y_cam)
    px, py = app.calib.cam_to_board(x_cam, y_cam, surf.get_size())   # relative to the module's surf
"""
import json
import os

import numpy as np

# calibration.json lives in the project root: it describes the physical setup (camera + projector),
# so it is shared by every project
CALIB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration.json")


class Calibration:
    def __init__(self, path=CALIB_FILE):
        self.path = path
        self.H = None            # camera pixel -> normalized coordinates
        self.cam_size = None     # camera resolution at calibration time (w, h)
        self.rms = None          # root-mean-square error (pixels inside the frame)

    @property
    def ok(self):
        return self.H is not None

    def set(self, H, cam_size, rms):
        self.H = np.asarray(H, dtype=float)
        self.cam_size = (int(cam_size[0]), int(cam_size[1]))
        self.rms = float(rms)

    def clear(self):
        self.H = self.cam_size = self.rms = None

    @staticmethod
    def _apply(M, x, y):
        v = M @ np.array([x, y, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def cam_to_norm(self, x, y):
        if not self.ok:
            raise RuntimeError("Not calibrated")
        return self._apply(self.H, x, y)

    def cam_to_board(self, x, y, size):
        u, v = self.cam_to_norm(x, y)
        return u * size[0], v * size[1]

    def norm_to_cam(self, u, v):
        if not self.ok:
            raise RuntimeError("Not calibrated")
        return self._apply(np.linalg.inv(self.H), u, v)

    # --- Save / load ---------------------------------------------------------
    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"H": self.H.tolist(), "cam_size": self.cam_size, "rms": self.rms}, f, indent=2)

    @classmethod
    def load(cls, path=CALIB_FILE):
        c = cls(path)
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            c.set(d["H"], d["cam_size"], d["rms"])
        except (OSError, KeyError, ValueError):
            c.clear()
        return c
