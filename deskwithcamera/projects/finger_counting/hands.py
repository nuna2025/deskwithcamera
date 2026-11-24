"""Hand points (MediaPipe) and the geometry used to count fingers.

Points per hand: 21 (MediaPipe numbering)
    0 wrist | thumb 1-4 | index 5-8 | middle 9-12 | ring 13-16 | pinky 17-20   (last of each = fingertip)
Everything below works on (21, 3) arrays x, y, z (camera pixels; MediaPipe's z is scaled like x) - distances are 3D,
so a hand that is tilted towards or away from the camera still counts correctly. Only ratios are used.
"""
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MODEL_PATH = HERE / "assets" / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

STRAIGHT_FINGER = 0.82             # index..pinky are stretched when straighter than this (see straightness())
STRAIGHT_THUMB = 0.80
TIPS = (4, 8, 12, 16, 20)
PIPS = (3, 6, 10, 14, 18)          # the joint before the tip (thumb: IP)
THUMB_REACH = 0.55                 # thumb counts as stretched when its tip is this many palms away from the base of the index finger
FEATURE_N = 94                     # length of features()


@dataclass
class Hand:
    pts: np.ndarray                # (21, 3) camera pixels (x, y) and depth (z, same scale as x; smaller = closer to the camera)
    label: str = ""                # "Left" / "Right" as MediaPipe says (not used for counting)


def _d(a, b):
    return float(np.linalg.norm(a - b))


def palm_size(p):
    """Wrist -> base of the middle finger: the yardstick for everything else."""
    return max(_d(p[0], p[9]), 1e-6)


def thumb_reach(p):
    """Thumb tip -> base of the index finger, in palms (about 0.4 tucked in, 0.8+ stretched out)."""
    p = np.asarray(p, dtype=float)
    return _d(p[4], p[5]) / palm_size(p)


def straightness(p, a, b, c, d):
    """Straight-line distance a->d divided by the length of the bent line a-b-c-d: 1.0 = straight finger,
    about 0.7 = bent at a knuckle, 0.4 or less = curled up. Does not depend on size, rotation or tilt."""
    p = np.asarray(p, dtype=float)
    path = _d(p[a], p[b]) + _d(p[b], p[c]) + _d(p[c], p[d])
    return _d(p[a], p[d]) / max(path, 1e-6)


def finger_straightness(p):
    """[thumb, index, middle, ring, pinky] straightness values (for the rules and the camera preview)."""
    out = [straightness(p, 2, 3, 3, 4)]
    for mcp in (5, 9, 13, 17):
        out.append(straightness(p, mcp, mcp + 1, mcp + 2, mcp + 3))
    return out


def finger_states(p):
    """[thumb, index, middle, ring, pinky] -> True = stretched. A finger is stretched when it is STRAIGHT
    (a finger that is bent or curled is not, even if it points towards the camera);
    the thumb also has to be away from the hand."""
    s = finger_straightness(p)
    st = [thumb_reach(p) > THUMB_REACH and s[0] > STRAIGHT_THUMB]
    st += [v > STRAIGHT_FINGER for v in s[1:]]
    return st


def count_rule(p):
    return int(sum(finger_states(p)))


def tips_by_reach(p, k):
    """The k fingers whose tips are farthest from the wrist (to mark them when only a number is known)."""
    p = np.asarray(p, dtype=float)
    reach = [_d(p[t], p[0]) for t in TIPS]
    return sorted(np.argsort(reach)[::-1][:k].tolist())


def features(p):
    """Position-, size- and rotation-independent description of one hand (for the trained model).

    The hand gets its own coordinate system: wrist = origin, wrist -> base of the middle finger = 'up',
    index base -> little-finger base = 'right', one palm = length 1. So it does not matter where the hand is,
    how big it looks, which way it points or how it is tilted towards the camera."""
    p = np.asarray(p, dtype=float)
    q = (p - p[0]) / palm_size(p)
    y = q[9] / max(np.linalg.norm(q[9]), 1e-9)
    v = q[17] - q[5]
    x = v - np.dot(v, y) * y
    x /= max(np.linalg.norm(x), 1e-9)
    z = np.cross(x, y)
    c = q @ np.stack([x, y, z], axis=1)                       # points in the hand's own system
    reach = np.linalg.norm(c[list(TIPS)], axis=1)             # tip distance from the wrist (palms)
    ratio = reach / np.maximum(np.linalg.norm(c[list(PIPS)], axis=1), 1e-6)
    return np.concatenate([c[1:].reshape(-1), reach, ratio, _extra(p)]).astype(np.float32)   # FEATURE_N numbers


def _angle(u, v):
    """Angle between two bones, 0 (straight on) ... 1 (folded back), no unit needed."""
    n = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.arccos(np.clip(np.dot(u, v) / max(n, 1e-9), -1.0, 1.0)) / np.pi)


def _extra(p):
    """24 numbers that describe every finger directly: how bent each of its 3 joints is (15), how straight the finger is (5)
    and how far the neighbouring fingertips are from each other (4). None of them changes when the hand is mirrored."""
    p = np.asarray(p, dtype=float)
    chains = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    bend = []
    for ch in chains:
        bones = [p[b] - p[a] for a, b in zip(ch[:-1], ch[1:])]
        bend += [_angle(bones[i], bones[i + 1]) for i in range(3)]
    straight = finger_straightness(p)
    tips = p[list(TIPS)]
    palm = palm_size(p)
    gap = [np.linalg.norm(tips[i] - tips[i + 1]) / palm for i in range(4)]
    return np.array(bend + list(straight) + gap, dtype=np.float32)


def mirrored(f):
    """The same hand seen as its mirror image (left <-> right hand, palm <-> back): flips the depth axis.
    Recordings are saved in both versions, so one model fits both hands."""
    f = np.array(f, dtype=np.float32)
    f[:60].reshape(20, 3)[:, 2] *= -1
    return f


def to_norm(H, pts):
    """Camera pixels -> table coordinates 0..1 (H = calibration matrix)."""
    a = np.hstack([np.asarray(pts, dtype=float)[:, :2], np.ones((len(pts), 1))]) @ np.asarray(H, dtype=float).T
    return a[:, :2] / a[:, 2:3]


class HandTracker:
    """MediaPipe hand landmarker. The model file (about 7 MB) is downloaded once, the first time, in the background."""

    def __init__(self):
        self.status = "idle"          # idle | loading | ready | error
        self.message = ""
        self._lm = None
        self._lm_img = None
        self._last_ts = 0

    def start(self):
        if self.status != "idle":
            return
        self.status = "loading"
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
            if not MODEL_PATH.exists():
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                self.message = "downloading the hand model"
                print(f"[fingers] downloading the hand model to {MODEL_PATH} ...")
                tmp = MODEL_PATH.with_suffix(".part")
                urllib.request.urlretrieve(MODEL_URL, tmp)
                tmp.replace(MODEL_PATH)
            opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO, num_hands=3,
                min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
            self._lm = vision.HandLandmarker.create_from_options(opts)
            self._lm_img = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(      # for detect_zoom (single pictures)
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.IMAGE, num_hands=2,
                min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
            self._mp = mp
            self.status = "ready"
            print("[fingers] hand model ready")
        except ImportError:
            self.status, self.message = "error", "mediapipe missing: pip install mediapipe"
            print(f"[fingers] {self.message}")
        except Exception as e:
            self.status, self.message = "error", f"hand model not available ({type(e).__name__})"
            print(f"[fingers] could not load the hand model: {e}")

    @property
    def ready(self):
        return self.status == "ready"

    def detect(self, frame_bgr, ts_ms=None):
        """-> list of Hand (camera pixels of the given frame). ts_ms: time of the frame in a video (offline use);
        without it the real clock is used."""
        import cv2
        h, w = frame_bgr.shape[:2]
        small = frame_bgr if w <= 960 else cv2.resize(frame_bgr, (960, int(h * 960 / w)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        ts = max(int(time.monotonic() * 1000) if ts_ms is None else int(ts_ms), self._last_ts + 1)
        self._last_ts = ts
        res = self._lm.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)
        hands = []
        for i, lms in enumerate(res.hand_landmarks):
            pts = np.array([[l.x * w, l.y * h, l.z * w] for l in lms], dtype=float)
            label = res.handedness[i][0].category_name if res.handedness else ""
            hands.append(Hand(pts, label))
        return hands

    def detect_zoom(self, frame_bgr, cols=3, rows=2, overlap=0.4):
        """Second chance for a picture where detect() found nothing (offline use, it is slow).
        MediaPipe first shrinks the whole picture to a tiny square to look for a hand, so a small hand gets lost.
        Here the picture is cut into overlapping tiles and every tile is searched on its own (the hand is relatively
        bigger there). Hands that touch a tile edge are ignored (cut off), the rest is converted back to full-picture pixels."""
        import cv2
        h, w = frame_bgr.shape[:2]
        tw, th = int(w / (cols - (cols - 1) * overlap)), int(h / (rows - (rows - 1) * overlap))
        xs = np.linspace(0, w - tw, cols).astype(int)
        ys = np.linspace(0, h - th, rows).astype(int)
        found = []
        for y0 in ys:
            for x0 in xs:
                tile = cv2.cvtColor(frame_bgr[y0:y0 + th, x0:x0 + tw], cv2.COLOR_BGR2RGB)
                res = self._lm_img.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(tile)))
                for i, lms in enumerate(res.hand_landmarks):
                    a = np.array([[l.x, l.y, l.z] for l in lms], dtype=float)
                    m = 0.02                                                     # a point this close to a tile edge: the hand is cut off
                    cut = ((a[:, 0] < m) & (x0 > 0)) | ((a[:, 0] > 1 - m) & (x0 + tw < w)) | \
                          ((a[:, 1] < m) & (y0 > 0)) | ((a[:, 1] > 1 - m) & (y0 + th < h))
                    if cut.any():
                        continue
                    pts = np.stack([a[:, 0] * tw + x0, a[:, 1] * th + y0, a[:, 2] * tw], axis=1)
                    label = res.handedness[i][0].category_name if res.handedness else ""
                    found.append(Hand(pts, label))
        unique = []                                                              # the same hand seen in two tiles: keep one
        for hd in sorted(found, key=lambda x: -palm_size(x.pts)):
            if all(_d(hd.pts[0], u.pts[0]) > 0.5 * palm_size(u.pts) for u in unique):
                unique.append(hd)
        return unique
