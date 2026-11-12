"""Oil painting: put a photo in the middle of the table, take it away, its oil painting appears there.

Flow
    1. Calibrate first (key 2). The program takes the empty-table picture by itself a few seconds after the
       module starts ("Clear the table").
    2. phase "place":  "Place your photo in the center" (dashed frame). The photo is found and painted.
    3. phase "ready":  the painting is done: "Take the photo away: your painting is ready".
    4. phase "show":   the table has been empty for a moment -> the painting is projected in the middle.
                       Hold a hand on NEW for another photo.

Buttons (hold a hand on them): HOME, STROKE (Light / Medium / Strong), COLOR (Pure / Natural / Vivid / Warm / Cool / Muted), RESET (3 s: empty-table picture again),
NEW (in the middle, only while a painting is ready or shown). STROKE / COLOR paint the same photo again.
Keys (projection window AND camera preview window): B = new empty-table picture (same as RESET), N = new photo, T = stroke, C = color
"""
import math

import numpy as np
import pygame

import config
from core.buttons import ROW_Y, ButtonSpec
from core.module import Module
from core.opts import opt
from core.shapes import draw_lower_bg
from projects.oil_painting import dataset, styles
from projects.mushroom.cards import StabilityTracker, clean_reference, crop_inner, crop_raw, detect_cards, rectify

try:
    import cv2
except ImportError:
    cv2 = None

PREVIEW_WINDOW = "Oil painting   (B = empty table, N = new photo, T = stroke, C = color)"
CROP_INNER = 0.95            # cut off the table / paper edge around the photo
MAX_SIDE = 480               # photos are scaled down to this before painting (speed)
REF_DELAY = 6.0              # seconds after the module starts until the empty-table picture is taken
RESET_DELAY = 4.0            # ... after RESET / key B
REF_WINDOW = 2.5             # the empty-table picture is the MEDIAN of the last 2.5 s: an arm passing by drops out
MAX_LONG_FRAC = 0.62         # a photo may be this long (fraction of the board width); cards are smaller
EDGE_FRAC = 0.03             # a photo must stay this far (fraction of the width) from the left, right and bottom edge
EMPTY_SECONDS = 1.0          # the table must be empty this long before the painting is shown
AFTER_NEW_SECONDS = 1.0      # after NEW: ignore the table for a moment (the painting disappears from it)
DASH_COLOR = config.BG_COLOR


class OilPainting(Module):
    title = "Photo to painting"
    area = "Oil painting"
    menu_label = "Oil painting"
    menu_caption = "Photo to painting"

    def __init__(self):
        self._preview_on = config.CAMERA_PREVIEW
        self.data_dir = dataset.DATA_DIR
        self.stroke = 1                       # index in styles.STROKE_LEVELS (Medium)
        self.color = 0                        # index in styles.COLOR_GRADES (0 = Pure)
        self.reset()

    def reset(self):
        self.reference, self._ref_H = None, None
        self._ref_buf, self._buf_t = [], 0.0
        self._ref_countdown = 0.0             # > 0: seconds until the reference is taken (hands away!)
        self.board, self.cards, self.photo = None, [], None
        self.tracker = StabilityTracker()
        self.phase = "place"                  # place | ready | show
        self._empty_t = 0.0                   # seconds the table has been empty (phase ready)
        self._hold = 0.0                      # seconds to ignore the table (after NEW)
        self._giant_t = 0.0                   # seconds something table-sized has been seen
        self._photo_box = None                # where the photo lay (board pixels): 'photo taken away' is checked there
        self._mask = None                     # what the detector sees (half-size), latest
        self.source = None                    # cut-out photo (BGR), kept so STROKE / COLOR can paint it again
        self.frame = None                     # latest full camera frame (saved with the capture)
        self._ref_norm = None                 # brightness-independent copy of the empty-table picture
        self.folder = None                    # session folder of the current photo (None = nothing saved yet)
        self.result = None                    # painted image (BGR)
        self.surface = None                   # result as pygame surface (cache)

    # --- lifecycle ---------------------------------------------------------
    def on_enter(self, app):
        self.reset()
        self._ref_countdown = REF_DELAY

    def on_exit(self, app):
        if cv2 is not None and self._preview_on:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
            except cv2.error:
                pass

    def handle_event(self, event, app):
        if event.type == pygame.KEYDOWN:
            ch = {pygame.K_b: "b", pygame.K_n: "n", pygame.K_t: "t", pygame.K_c: "c"}.get(event.key)
            if ch:
                self.key(ch)

    def key(self, ch, app=None):
        if ch == "b":
            self.start_reference(app, RESET_DELAY)
        elif ch == "n":
            self.new_photo(app)
        elif ch == "t":
            self.stroke = (self.stroke + 1) % len(styles.STROKE_LEVELS)
            self._repaint()
        elif ch == "c":
            self.color = (self.color + 1) % len(styles.COLOR_GRADES)
            self._repaint()

    def start_reference(self, app=None, seconds=RESET_DELAY):
        """RESET / key B: forget the empty-table picture and take it again after `seconds` (hands away!)."""
        self.reference, self._ref_norm = None, None
        self._ref_countdown = seconds
        self._ref_buf, self._buf_t = [], 0.0
        self.new_photo(app)

    def _take_reference(self, app):
        """Median of the pictures of the last REF_WINDOW seconds, objects already lying there removed."""
        frames = self._ref_buf or [self.board]
        ref = np.median(np.array(frames), axis=0).astype(np.uint8)
        self.reference, n_there = clean_reference(ref)
        if n_there:
            print(f"[oil painting] {n_there} object(s) were on the table when the empty-table picture was taken (removed from it)")
        self._ref_H = app.calib.H.copy()
        self._ref_norm = None
        self._ref_buf = []

    def new_photo(self, app=None):
        """NEW: forget the photo and the painting, wait for the next photo."""
        self.cards, self.photo, self.source, self.result, self.surface = [], None, None, None, None
        self.folder = None
        self._photo_box = None
        self.tracker.reset()
        self._empty_t = 0.0
        self._hold = AFTER_NEW_SECONDS
        self._set_phase("place", app)

    def _set_phase(self, phase, app=None):
        changed = phase != self.phase
        self.phase = phase
        if changed and app is not None:
            app.refresh_buttons()

    # --- buttons -----------------------------------------------------------
    def buttons(self, app):
        row_mid = ROW_Y + opt("CIRCLE_D") / 2
        new_slot = (0.5 - opt("BUTTON_W") / 2, row_mid - opt("BUTTON_H") / 2, opt("BUTTON_W"), opt("BUTTON_H"))
        specs = [
            ButtonSpec("color", "COLOR", lambda app: self.key("c", app), "right",
                       caption=lambda app: styles.COLOR_GRADES[self.color]),
            ButtonSpec("stroke", "STROKE", lambda app: self.key("t", app), "right2",
                       caption=lambda app: styles.STROKE_LEVELS[self.stroke][0]),
        ]
        if self.phase in ("ready", "show"):
            specs.append(ButtonSpec("new", "NEW", lambda app: self.new_photo(app), new_slot))
        else:                                               # empty-table picture wrong? hold a hand here, then clear the table
            specs.append(ButtonSpec("reset", "RESET", lambda app: self.key("b", app), new_slot, hold=3.0))
        return specs

    # --- per frame ------------------------------------------------------------
    def update(self, dt, app):
        if cv2 is None or not app.calib.ok:
            return
        frame = app.camera.latest()
        if frame is None:
            return
        self.frame = frame
        self.board = rectify(frame, app.calib.H)
        if self._ref_H is not None and not np.allclose(self._ref_H, app.calib.H):
            self.reference, self._ref_norm = None, None

        if self._ref_countdown > 0:
            self._ref_countdown -= dt
            self._buf_t += dt
            if self._ref_countdown < REF_WINDOW and self._buf_t >= 0.25:
                self._buf_t = 0.0
                self._ref_buf.append(self.board)
            if self._ref_countdown <= 0:
                self._take_reference(app)
        elif self.reference is not None:
            self._look_at_table(dt, app)

        if self._preview_on:
            self._show_preview(app)

    def _look_at_table(self, dt, app):
        if self.phase == "show":
            return                                          # our own painting is on the table: do not look
        if self._hold > 0:
            self._hold -= dt
            self.cards = []
            return
        bw = self.board.shape[1]
        cards = self._detect()
        for c in cards:                                     # the photo lies in the middle and may be big
            c.flags = [f for f in c.flags if f not in ("crosses_center", "too_big")]
            if c.size[0] > MAX_LONG_FRAC * bw:
                c.flags.append("too_big")
            (x0, y0), (x1, y1) = c.box.min(axis=0), c.box.max(axis=0)
            m = EDGE_FRAC * bw
            if x0 < m or x1 > bw - m or y1 > self.board.shape[0] - m:
                c.flags.append("touches_edge")              # a photo lies in the middle; arms and hands come from outside
        self.cards = cards
        if any(c.size[0] > 0.85 * bw for c in cards):       # the whole table looks 'new': the empty-table picture is stale
            self._giant_t += dt
            if self._giant_t >= 3.0:
                self._giant_t = 0.0
                print("[oil painting] the whole table differs from the empty-table picture: taking it again")
                self.start_reference(app, 3.0)
        else:
            self._giant_t = 0.0
        self.tracker.update(self.cards)
        self.photo = self.cards[0] if len(self.cards) == 1 else None

        if self.phase == "place":
            if self.photo is not None and not self.photo.flags and self.tracker.stable:
                self._paint_photo(app)
                self._empty_t = 0.0
                self._set_phase("ready", app)
        elif self.phase == "ready":
            if self._photo_gone():                          # photo taken away
                self._empty_t += dt
                if self._empty_t >= EMPTY_SECONDS:
                    self._set_phase("show", app)
            else:
                self._empty_t = 0.0

    def _norm(self, img, excl=None):
        """Brightness-independent view of the table (used for finding the photo).

        Every pixel is divided by the blurred picture around it. A webcam that changes exposure / white balance
        by itself, a projector hot spot or a soft shadow then cancels out; a photo (small, with its own colors)
        stays. Without this the whole table differs from the empty-table picture and is taken for a giant photo.
        excl = mask (255 = photo): these pixels are left out of the blur, otherwise a dark photo would make the
        table around it look brighter (a halo that is taken for part of the photo)."""
        h, w = img.shape[:2]
        y0 = int(opt("HUD_FRAC") * h)
        low = img[y0:].astype(np.float32)
        wt = np.ones(low.shape[:2], np.float32) if excl is None else (1.0 - excl[y0:] / 255.0).astype(np.float32)
        sz = (w // 4, low.shape[0] // 4)
        small = cv2.resize(low * wt[..., None], sz, interpolation=cv2.INTER_AREA)
        wsmall = cv2.resize(wt, sz, interpolation=cv2.INTER_AREA)
        pad = sz[0] // 4
        border = lambda a: cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        sigma = 0.15 * sz[0]
        num = cv2.GaussianBlur(border(small), (0, 0), sigma)[pad:-pad, pad:-pad]
        den = cv2.GaussianBlur(border(wsmall), (0, 0), sigma)[pad:-pad, pad:-pad]
        blur = cv2.resize(num / np.maximum(den, 0.05)[..., None], (w, low.shape[0]), interpolation=cv2.INTER_LINEAR)
        out = np.full(img.shape, 128, np.float32)           # button area: constant (it is ignored anyway)
        out[y0:] = np.clip(low / (blur + 8.0) * 128.0, 0, 255)
        return out.astype(np.uint8)

    def _detect(self):
        """Photos on the table: first look with the plain brightness-independent view, then again with the
        found photo left out of the blur (sharper outline). Works on a half-size picture (speed); the
        results are scaled back to board pixels."""
        h, w = self.board.shape[:2]
        half = (w // 2, h // 2)
        board = cv2.resize(self.board, half, interpolation=cv2.INTER_AREA)
        if self._ref_norm is None or self._ref_norm[0] != self.reference.shape:
            ref = cv2.resize(self.reference, half, interpolation=cv2.INTER_AREA)
            self._ref_norm = (self.reference.shape, ref, self._norm(ref))
        _, ref, ref_norm = self._ref_norm
        cards, mask = detect_cards(self._norm(board), ref_norm, return_mask=True)
        if cards:
            excl = cv2.dilate(mask, np.ones((31, 31), np.uint8))
            cards, mask = detect_cards(self._norm(board, excl), self._norm(ref, excl), return_mask=True)
        self._mask = mask
        for c in cards:                                     # back to board pixels
            c.box = c.box * 2.0
            c.center = (c.center[0] * 2.0, c.center[1] * 2.0)
            c.size = (c.size[0] * 2.0, c.size[1] * 2.0)
            c.area *= 4.0
        return cards

    def _photo_gone(self):
        """True when the place where the photo lay is empty again. Only that place is checked, so a stale spot
        elsewhere in the empty-table picture (e.g. an arm that was there when it was taken) does not matter."""
        if self._photo_box is None or self._mask is None:
            return not self.cards
        box = self._photo_box.astype(np.float32)
        c = box.mean(axis=0)
        poly = ((box - c) * 0.8 + c) / 2.0                  # shrunk to 80 %, half-size like the mask
        fp = np.zeros(self._mask.shape[:2], np.uint8)
        cv2.fillPoly(fp, [poly.astype(np.int32)], 255)
        n = int((fp > 0).sum())
        if n < 50:
            return not self.cards
        return float((self._mask[fp > 0] > 0).mean()) < 0.15

    def _paint_photo(self, app):
        c = self.photo
        self._photo_box = c.box.copy()
        raw = crop_raw(self.board, c)
        self.source = crop_inner(raw, CROP_INNER)
        self._repaint()
        try:                                                # saving must never stop the program
            self.folder = dataset.save_capture(self.data_dir, self.frame, self.board, self.reference,
                                               self._overlay_image(), self.source, raw, self.result,
                                               app.calib, styles.STROKE_LEVELS[self.stroke][0], styles.COLOR_GRADES[self.color])
            print(f"[oil painting] saved to {self.folder}")
        except Exception as e:
            self.folder = None
            print(f"[oil painting] could not save: {e}")

    def _repaint(self):
        """Paint the stored photo with the current stroke / color."""
        if self.source is None or cv2 is None:
            return
        self.result = styles.paint(self.source, self.stroke, self.color, MAX_SIDE)
        rgb = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)
        self.surface = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
        if self.folder is not None:                         # same capture, new look: keep the file up to date
            try:
                dataset.update_painting(self.folder, self.result, styles.STROKE_LEVELS[self.stroke][0], styles.COLOR_GRADES[self.color])
            except Exception as e:
                print(f"[oil painting] could not update the saved painting: {e}")

    def _overlay_image(self):
        img = self.board.copy()
        for c in self.cards:
            cv2.polylines(img, [c.box.astype(np.int32)], True, (0, 0, 255) if c.flags else (255, 170, 60), 3)
        return img

    # --- header hint ---------------------------------------------------------------
    def hint(self, app):
        """(text, mode) for the second header line."""
        if cv2 is None:
            return "OpenCV missing", "warn"
        if not app.calib.ok:
            return "Calibrate first (key 2)", "warn"
        if not app.camera.is_on:
            return "Cam OFF", "warn"
        if self._ref_countdown > 0:
            return f"Clear the table: reference in {math.ceil(self._ref_countdown)}", "blink"
        if self.reference is None:
            return "Waiting for the camera ...", "info"
        if self.phase == "show":
            return "Your painting: hold a hand on NEW for another photo", "info"
        if self.phase == "ready":
            return "Take the photo away: your painting is ready", "blink"
        if len(self.cards) > 1:
            return "One photo at a time", "warn"
        if not self.cards:
            return "Place your photo in the center", "blink"
        flags = self.cards[0].flags
        if "touches_edge" in flags:
            return "Hands on the table? Only the photo, in the center", "warn"
        if "touches_hud" in flags:
            return "Photo too high: keep it below the buttons", "warn"
        if "too_big" in flags:
            return "Photo too big for the table", "warn"
        if "irregular" in flags:
            return "Unclear shape: flatten the photo", "warn"
        if not self.tracker.stable:
            return "Hold still ...", "info"
        return "Painting ...", "info"

    # --- drawing -----------------------------------------------------------------
    def draw(self, surf, app):
        w, h = surf.get_size()
        hud_y = round(opt("HUD_FRAC") * h)
        draw_lower_bg(surf)                                 # no center line in this module
        area = pygame.Rect(0, hud_y, w, h - hud_y)
        if self.phase == "show" and self.surface is not None:
            self._draw_painting(surf, area, h)
        elif self.phase == "place":
            self._draw_frame(surf, app, area, h)

    def _draw_painting(self, surf, area, h):
        m = round(h * 0.015)
        box = area.inflate(-2 * m, -2 * m)
        sw, sh = self.surface.get_size()
        s = min(box.w / sw, box.h / sh)
        size = (max(int(sw * s), 1), max(int(sh * s), 1))
        scaled = pygame.transform.smoothscale(self.surface, size)
        mask = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=round(h * 0.035))
        out = scaled.convert_alpha()
        out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        surf.blit(out, out.get_rect(center=box.center))

    def _draw_frame(self, surf, app, area, h):
        """Dashed rounded frame in the middle: 'put the photo here'."""
        rect = pygame.Rect(0, 0, round(area.w * 0.52), round(area.h * 0.78))
        rect.center = area.center
        r = round(h * 0.05)
        dash, gap, wd = round(h * 0.035), round(h * 0.022), max(3, round(h * 0.004))
        # corners as arcs, straight parts dashed
        for cx, cy, a0 in ((rect.left + r, rect.top + r, math.pi / 2), (rect.right - r, rect.top + r, 0.0),
                           (rect.right - r, rect.bottom - r, -math.pi / 2), (rect.left + r, rect.bottom - r, math.pi)):
            pygame.draw.arc(surf, DASH_COLOR, pygame.Rect(cx - r, cy - r, 2 * r, 2 * r), a0, a0 + math.pi / 2, wd)
        for (x0, y0), (x1, y1) in (((rect.left + r, rect.top), (rect.right - r, rect.top)),
                                   ((rect.left + r, rect.bottom), (rect.right - r, rect.bottom)),
                                   ((rect.left, rect.top + r), (rect.left, rect.bottom - r)),
                                   ((rect.right, rect.top + r), (rect.right, rect.bottom - r))):
            length = math.hypot(x1 - x0, y1 - y0)
            n = max(int(length // (dash + gap)), 1)
            step = length / n
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            for i in range(n):
                a = i * step + gap / 2
                b = min(a + dash, (i + 1) * step - gap / 2)
                pygame.draw.line(surf, DASH_COLOR, (x0 + ux * a, y0 + uy * a), (x0 + ux * b, y0 + uy * b), wd)
        if not self.cards:
            font = app.board.font(h * 0.04)
            t = font.render("Place your photo here", True, DASH_COLOR)
            surf.blit(t, t.get_rect(center=rect.center))

    # --- laptop preview ------------------------------------------------------------
    def _show_preview(self, app):
        if self.board is None:
            return
        try:
            img = self.board.copy()
            for c in self.cards:
                cv2.polylines(img, [c.box.astype(np.int32)], True, (0, 0, 255) if c.flags else (255, 170, 60), 3)
            if self.result is not None:
                r = self.result
                s = 300 / max(r.shape[:2])
                r = cv2.resize(r, None, fx=s, fy=s)
                img[10:10 + r.shape[0], img.shape[1] - 10 - r.shape[1]:img.shape[1] - 10] = r
            cv2.imshow(PREVIEW_WINDOW, img)
            k = cv2.waitKey(1) & 0xFF
            ch = {ord("b"): "b", ord("n"): "n", ord("t"): "t", ord("c"): "c"}.get(k)
            if ch:
                self.key(ch, app)
        except cv2.error:
            self._preview_on = False
            print("[oil painting] preview window not available (install opencv-contrib-python)")
