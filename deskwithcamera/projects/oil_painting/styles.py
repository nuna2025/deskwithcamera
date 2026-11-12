"""Image styles for the photo project (pure functions: BGR image in, BGR image out, no pygame).

Everything here uses plain OpenCV, so nothing extra has to be installed and it runs in well under a
second on a laptop. A neural style (e.g. AnimeGAN) can later be added as one more entry in STYLES.
"""
import inspect

import cv2
import numpy as np


def cartoon(img, levels=7, edge_block=9):
    """Flat colors + black outlines."""
    small = img
    for _ in range(3):                                    # smooth the colors but keep the edges
        small = cv2.bilateralFilter(small, d=7, sigmaColor=40, sigmaSpace=7)
    step = 256 // levels
    flat = (small // step) * step + step // 2             # few color levels = "painted" look
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, edge_block, 4)
    return cv2.bitwise_and(flat.astype(np.uint8), cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))


def watercolor(img):
    """Soft painted look."""
    return cv2.stylization(img, sigma_s=60, sigma_r=0.45)


def pencil(img):
    """Colored pencil drawing."""
    return cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)[1]


def anime(img):
    """Illustration look that stays true to the photo (nothing is invented):
    stronger local contrast and colors, flat color regions that follow the real edges, thin ink lines."""
    img = cv2.resize(img, (640, max(int(img.shape[0] * 640 / img.shape[1]), 1)), interpolation=cv2.INTER_AREA)
    # 1) local contrast (clouds, shadows) and stronger colors
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6)).apply(l)
    a = np.clip((a.astype(np.float32) - 128) * 1.5 + 128, 0, 255).astype(np.uint8)
    b = np.clip((b.astype(np.float32) - 128) * 1.5 + 128, 0, 255).astype(np.uint8)
    img2 = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 255)
    hsv[..., 2] = 255 * (hsv[..., 2] / 255) ** 0.85
    img2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # 2) flat color regions
    flat = cv2.bilateralFilter(cv2.pyrMeanShiftFiltering(img2, sp=9, sr=22), 7, 25, 7)
    # 3) thin ink lines (strong edges only, tiny specks removed)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = cv2.GaussianBlur(g, (0, 0), 0.8) - 0.98 * cv2.GaussianBlur(g, (0, 0), 1.6)
    edge = (d < -1.2).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(edge, connectivity=8)
    keep = np.isin(labels, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 14])
    line = cv2.GaussianBlur((keep * 255).astype(np.uint8), (0, 0), 0.6) / 255.0
    ink = np.array([60, 45, 40], np.float32)
    out = flat.astype(np.float32) * (1 - 0.75 * line[..., None]) + ink * (0.75 * line[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def ghibli_like(img):
    """Soft, warm, hand-painted anime-background look (an approximation, not a copy of any studio).
    Like anime() it only uses what is in the photo: brighter colors, warm light, soft painted gradients,
    faint colored lines and a gentle glow on the highlights."""
    img = cv2.resize(img, (640, max(int(img.shape[0] * 640 / img.shape[1]), 1)), interpolation=cv2.INTER_AREA)
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(l)
    a = np.clip((a.astype(np.float32) - 128) * 1.3 + 128, 0, 255).astype(np.uint8)
    b = np.clip((b.astype(np.float32) - 128) * 1.35 + 128, 0, 255).astype(np.uint8)
    x = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 255)
    hsv[..., 2] = 255 * (hsv[..., 2] / 255) ** 0.8
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    shade = 1 - cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    x[..., 2] = np.clip(x[..., 2] * 1.04 + 8 * shade, 0, 255)        # warm light, a bit more in the shadows
    x[..., 0] = np.clip(x[..., 0] * 0.97, 0, 255)
    x = x.astype(np.uint8)
    flat = cv2.bilateralFilter(cv2.pyrMeanShiftFiltering(x, sp=7, sr=16), 7, 22, 7)
    soft = cv2.edgePreservingFilter(x, flags=1, sigma_s=30, sigma_r=0.3)
    base = cv2.addWeighted(flat, 0.55, soft, 0.45, 0).astype(np.float32)   # painted areas + soft gradients
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = cv2.GaussianBlur(g, (0, 0), 0.8) - 0.98 * cv2.GaussianBlur(g, (0, 0), 1.6)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((d < -1.6).astype(np.uint8) * 255, connectivity=8)
    keep = np.isin(labels, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 30])
    line = cv2.GaussianBlur((keep * 255).astype(np.uint8), (0, 0), 0.7) / 255.0
    ink = np.array([45, 60, 70], np.float32)
    out = base * (1 - 0.4 * line[..., None]) + ink * (0.4 * line[..., None])
    glow = cv2.GaussianBlur(np.clip((out - 190) / 65, 0, 1) * out, (0, 0), 9)
    out = 255 - (255 - out) * (255 - 0.35 * glow) / 255
    return np.clip(out, 0, 255).astype(np.uint8)


def oil_rough(img, strength=0.5):
    """Oil painting with visible rough brush strokes that keep the details (impasto look).
    The strokes follow the contours of the picture, and a light relief makes them look like thick paint.
    strength 0.0 = smooth, 0.5 = default, 1.0 = very pronounced strokes."""
    if not hasattr(cv2, "xphoto"):
        raise RuntimeError("oil_rough needs opencv-contrib-python (pip install opencv-contrib-python)")
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.35, 0, 255)
    hsv[..., 2] = 255 * (hsv[..., 2] / 255) ** 0.85
    base = cv2.xphoto.oilPainting(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), max(3, round(w / 240)), 1)
    # direction of the strokes: along the edges (perpendicular to the gradient), smoothed over the picture
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (0, 0), 6) for a in (gx * gx, gy * gy, gx * gy))
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    # bristle streaks: random noise smeared along 12 directions, each pixel takes its own direction
    noise = np.random.default_rng(3).normal(0, 1, (h, w)).astype(np.float32)
    n_ang, length = 12, max(9, (round(w / 64) // 2) * 2 + 1)
    idx = np.round((theta % np.pi) / np.pi * n_ang).astype(int) % n_ang
    tex = np.zeros((h, w), np.float32)
    for k in range(n_ang):
        a, c = np.pi * k / n_ang, (length - 1) / 2
        ker = np.zeros((length, length), np.float32)
        cv2.line(ker, (int(round(c - np.cos(a) * c)), int(round(c - np.sin(a) * c))),
                 (int(round(c + np.cos(a) * c)), int(round(c + np.sin(a) * c))), 1, 1)
        tex[idx == k] = cv2.filter2D(noise, -1, ker / ker.sum())[idx == k]
    tex /= tex.std()
    shade = cv2.Sobel(cv2.GaussianBlur(tex, (0, 0), 0.8), cv2.CV_32F, 1, 1, ksize=3)   # light from the top left
    shade /= np.abs(shade).max() + 1e-6
    out = base.astype(np.float32) * (1 + strength * 1.32 * shade[..., None]) + 12 * strength * tex[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


OIL_COARSENESS = 70      # brush size = image size / this number. 130 = fine strokes, 70 = rough, 45 = very rough


def oil(img, coarseness=None):
    """Oil painting: brush strokes + stronger colors + faint canvas texture (needs opencv-contrib)."""
    if not hasattr(cv2, "xphoto"):
        raise RuntimeError("oil style needs opencv-contrib-python (pip install opencv-contrib-python)")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.4, 0, 255)                  # stronger colors
    hsv[..., 2] = 255 * (hsv[..., 2] / 255) ** 0.78                   # lift dark areas
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    out = cv2.xphoto.oilPainting(img, max(3, round(max(img.shape[:2]) / (coarseness or OIL_COARSENESS))), 1)
    n = cv2.GaussianBlur(np.random.default_rng(0).normal(0, 1, out.shape[:2]).astype(np.float32), (0, 0), 1.0)
    return np.clip(out.astype(np.float32) * (1 + 0.05 * n / n.std())[..., None], 0, 255).astype(np.uint8)


# name -> function. The order is the order of the V key.
STYLES = {"anime": anime, "ghibli_like": ghibli_like, "cartoon": cartoon, "watercolor": watercolor, "oil": oil, "oil_rough": oil_rough, "pencil": pencil}


def has_strength(name):
    """True if the style has a strength setting (0..1)."""
    return "strength" in inspect.signature(STYLES[name]).parameters


def apply(name, img, max_side=480, strength=0.5):
    """Scale the image down to max_side (speed) and apply the style. strength only affects styles that have it."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)), interpolation=cv2.INTER_AREA)
    return STYLES[name](img, strength=strength) if has_strength(name) else STYLES[name](img)



# --- Oil painting: the only style the module uses. Two independent choices: brush stroke + color ---
STROKE_LEVELS = (("Light", 0.2), ("Medium", 0.5), ("Strong", 0.9))
COLOR_GRADES = ("Pure", "Natural", "Vivid", "Warm", "Cool", "Muted")      # Pure = default


def autolevels(img, lo=0.5, hi=98.0, amount=0.7, max_gain=2.6):
    """Take the projector's veil and yellow cast off a photo the camera saw lying on the projected table.

    Per color channel the darkest values (lo percentile) become black and the brightest (hi percentile) white.
    amount < 1 keeps a little of the original, max_gain stops one channel from being stretched too far.
    (Measured on a photo that the camera saw next to the real photo: the color error dropped about 3x.)"""
    f = img.astype(np.float32)
    out = np.empty_like(f)
    for c in range(3):
        black, white = np.percentile(f[..., c], [lo, hi])
        white = max(white, black + 30)
        out[..., c] = (f[..., c] - black) * min(255.0 / (white - black), max_gain)
    return np.clip(f * (1 - amount) + out * amount, 0, 255).astype(np.uint8)


def vibrance(img, k, contrast=1.0, gamma=1.0):
    """Saturation boost that leaves grays alone: colorful pixels get the full boost, near-gray pixels almost none."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    w = np.clip(hsv[..., 1] / 255.0 / 0.35, 0, 1)
    w = w * w * (3 - 2 * w)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + (k - 1) * w), 0, 255)
    hsv[..., 2] = 255 * (hsv[..., 2] / 255) ** gamma   # gamma > 1 = darker
    f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip((f - 128) * contrast + 128, 0, 255).astype(np.uint8)


PURE_VIBRANCE, PURE_CONTRAST, PURE_GAMMA = 2.4, 1.12, 1.12   # former variant 8 "Pure D" (vivid x2.4, a bit darker)


def grade(img, name):
    """Color mood applied after painting. Natural = unchanged. (Pure is handled in paint(): it also needs a step BEFORE painting.)"""
    if name in ("Natural", "Pure"):
        return img
    f = img.astype(np.float32)

    def saturate(f, k):
        hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * k, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if name == "Vivid":
        f = (saturate(f, 1.35) - 128) * 1.08 + 128
    elif name == "Warm":
        f = saturate(f, 1.1)
        f[..., 0] *= 0.88
        f[..., 1] *= 1.02
        f[..., 2] *= 1.08
    elif name == "Cool":
        f[..., 0] *= 1.10
        f[..., 2] *= 0.92
    elif name == "Muted":
        f = (saturate(f, 0.65) - 128) * 0.92 + 128 + 8
    return np.clip(f, 0, 255).astype(np.uint8)


def paint(img, stroke=1, color=0, max_side=480):
    """Oil painting. stroke = index in STROKE_LEVELS, color = index in COLOR_GRADES."""
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    name = COLOR_GRADES[color]
    if name == "Pure":                                    # veil off first, paint, then strong but clean colors
        return vibrance(oil_rough(autolevels(img), STROKE_LEVELS[stroke][1]), PURE_VIBRANCE, PURE_CONTRAST, PURE_GAMMA)
    return grade(oil_rough(img, STROKE_LEVELS[stroke][1]), name)
