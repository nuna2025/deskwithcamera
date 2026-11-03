"""Card detection in the top-down view (pure functions, no pygame - easy to test).

Pipeline
    camera frame --(calibration H)--> top-down view of the projected area ("board")
    board - empty-table reference   --> difference mask --> blobs --> rotated rectangles = cards
    each card: side (+ / -) from its center, perspective-corrected crop for the dataset
Cards may be squares or rectangles, at any angle.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from projects.mushroom import settings as cfg


@dataclass
class Card:
    box: np.ndarray                  # 4x2 corners in board pixels, clockwise from top-left
    center: tuple
    size: tuple                      # (long side, short side) in board pixels
    angle: float                     # minAreaRect angle (degrees)
    area: float
    side: str                        # "+" or "-"
    flags: list = field(default_factory=list)   # problems: crosses_center, too_big, irregular, touches_hud

    @property
    def ok(self):
        return not self.flags


def rectify(frame, H, size=None):
    """Camera frame -> top-down view of the projected content area (H = camera px -> 0..1)."""
    size = size or cfg.BOARD_SIZE
    S = np.diag([size[0], size[1], 1.0]) @ H
    return cv2.warpPerspective(frame, S, tuple(size), flags=cv2.INTER_LINEAR)


def order_box(pts):
    """Corners clockwise, starting with the one closest to the top-left."""
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def detect_cards(board, reference, board_size=None, return_mask=False):
    """Find all cards on the board. Returns a list of Card (flagged ones included).
    With return_mask=True returns (cards, mask) - the mask is what the detector 'sees' (for debugging)."""
    bw, bh = board_size or (board.shape[1], board.shape[0])
    hud_px = int(cfg.HUD_FRAC * bh)

    diff = cv2.absdiff(board, reference).max(axis=2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    otsu, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu alone sits between "black table" and "bright paper" and would throw away dark artwork
    # (e.g. a dark red hairy cap on a card), so only use a fraction of it
    thr = max(cfg.MIN_DIFF, int(otsu * cfg.OTSU_SCALE))
    mask = (diff > thr).astype(np.uint8) * 255
    mask[:hud_px, :] = 0                                      # HUD zone is ignored
    em = int(cfg.EDGE_MARGIN_FRAC * bw)                       # strip along the edges: projected frame line
    if em > 0:
        mask[:, :em] = 0
        mask[:, bw - em:] = 0
        mask[bh - em:, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # fill holes: dark artwork surrounded by the bright paper margin must count as part of the card
    outer, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, outer, -1, 255, thickness=cv2.FILLED)
    # erase everything thinner than a card: the white frame / center line show up in the difference
    # as thin lines as soon as the camera or the exposure moves a tiny bit
    k = max(5, int(cfg.OPEN_FRAC * bw)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []
    for cnt in contours:
        (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
        short, long = min(w, h), max(w, h)
        if short < cfg.MIN_SHORT_FRAC * bw:
            continue                                          # noise / thin lines
        area = cv2.contourArea(cnt)
        box = order_box(cv2.boxPoints(((cx, cy), (w, h), angle)))
        flags = []
        if long > cfg.MAX_LONG_FRAC * bw:
            flags.append("too_big")
        if area / max(w * h, 1.0) < cfg.MIN_FILL:
            flags.append("irregular")
        if box[:, 0].min() < bw / 2 < box[:, 0].max():
            flags.append("crosses_center")
        if box[:, 1].min() <= hud_px + 3:
            flags.append("touches_hud")
        cards.append(Card(box, (cx, cy), (long, short), angle, area, "-" if cx < bw / 2 else "+", flags))
    return (cards, mask) if return_mask else cards


class StabilityTracker:
    """Counts how many frames in a row the detected cards stayed (almost) where they were."""

    def __init__(self):
        self.prev = None
        self.count = 0

    def update(self, cards):
        cur = sorted((c.center for c in cards))
        if (self.prev is not None and len(cur) == len(self.prev) and len(cur) > 0 and
                max(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(cur, self.prev)) <= cfg.STABLE_TOL_PX):
            self.count += 1
        else:
            self.count = 1 if cur else 0
        self.prev = cur
        return self.count

    @property
    def stable(self):
        return self.count >= cfg.STABLE_FRAMES

    def reset(self):
        self.prev, self.count = None, 0


# --- cutting out single cards ----------------------------------------------------------
def crop_raw(board, card):
    """Perspective-corrected crop of the whole card, long side horizontal."""
    tl, tr, br, bl = card.box
    wd, hd = float(np.linalg.norm(tr - tl)), float(np.linalg.norm(bl - tl))
    dst = np.array([[0, 0], [wd, 0], [wd, hd], [0, hd]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(card.box.astype(np.float32), dst)
    raw = cv2.warpPerspective(board, M, (max(int(round(wd)), 1), max(int(round(hd)), 1)))
    if hd > wd:
        raw = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
    return raw


def crop_inner(raw, inner=None):
    inner = inner or cfg.CROP_INNER
    h, w = raw.shape[:2]
    my, mx = int(h * (1 - inner) / 2), int(w * (1 - inner) / 2)
    return raw[my:h - my, mx:w - mx]


def letterbox(img, size=None):
    """Scale to fit into size x size, keep the aspect ratio, pad with white."""
    size = size or cfg.CARD_SIZE
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(int(round(w * s)), 1), max(int(round(h * s)), 1)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return canvas


def white_reference(raw):
    """Mean RGB of the whitish margin of the card (to correct the lighting later). None if not found."""
    h, w = raw.shape[:2]
    m = max(int(0.06 * min(h, w)), 2)
    ring = np.ones((h, w), bool)
    ring[m:h - m, m:w - m] = False
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    whitish = ring & (hsv[..., 1] < 70) & (hsv[..., 2] > 120)
    if whitish.sum() < 20:
        return None
    b, g, r = raw[whitish].mean(axis=0)
    return int(round(r)), int(round(g)), int(round(b))