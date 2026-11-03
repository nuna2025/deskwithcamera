"""Card capture module: find the cards on the table, count them per side, save them as a dataset session.

Workflow
    1. Calibrate first (key 2). The camera must see the whole frame.
    2. Empty table (no cards)  ->  press B   (takes the empty-table reference picture)
    3. Put the cards BELOW the gray line, completely inside their half (+ right, - left), not touching
    4. Wait until the status says "Ready"  ->  press S   (saves the session)

Keys work in the projection window AND in the camera preview window on the laptop:
    B = empty-table reference     S = save session     D = show/hide the detector's mask (debugging)
    (ESC / C / Tab are handled by the app)
"""
import time

import numpy as np
import pygame

import config
from core.module import Module
from core.shapes import draw_minus, draw_plus, draw_text, draw_vline
from projects.mushroom import settings as cfg
from projects.mushroom.cards import StabilityTracker, detect_cards, rectify
from projects.mushroom.dataset import save_session
from projects.mushroom.plus_minus_split import LABELS

try:
    import cv2
except ImportError:
    cv2 = None

PREVIEW_WINDOW = "Card capture   (B = empty table, S = save, D = mask)"
COLORS = {"+": (60, 120, 255), "-": (255, 170, 60)}          # BGR: + orange-red, - blue
BAD = (0, 0, 255)


class CardCapture(Module):
    title = "Card capture"

    def __init__(self):
        self.reset()
        self._preview_on = config.CAMERA_PREVIEW
        self.data_dir = cfg.DATA_DIR

    def reset(self):
        self.reference = None
        self._ref_H = None
        self._ref_frames = []
        self._ref_pending = 0
        self.cards = []
        self.tracker = StabilityTracker()
        self.board = None
        self.frame = None
        self.msg, self.msg_until = "", 0.0
        self.last_saved = None
        self.mask = None
        self.show_mask = False

    # --- lifecycle --------------------------------------------------------
    def on_enter(self, app):
        self.reset()                 # exposure / lighting may have changed: take a new reference

    def on_exit(self, app):
        if cv2 is not None and self._preview_on:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
            except cv2.error:
                pass

    def handle_event(self, event, app):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_b:
                self.key("b", app)
            elif event.key == pygame.K_s:
                self.key("s", app)
            elif event.key == pygame.K_d:
                self.key("d", app)

    def key(self, ch, app):
        if ch == "b":
            self._ref_frames, self._ref_pending = [], cfg.REFERENCE_FRAMES
            self.cards = []
            self.tracker.reset()
        elif ch == "s":
            self.save(app)
        elif ch == "d":
            self.show_mask = not self.show_mask

    # --- per frame -----------------------------------------------------------
    def update(self, dt, app):
        if cv2 is None or not app.calib.ok:
            return
        frame = app.camera.latest()
        if frame is None:
            return
        self.frame = frame
        if self._ref_H is not None and not np.allclose(self._ref_H, app.calib.H):
            self.reference = None                         # calibration changed -> old reference is invalid
        self.board = rectify(frame, app.calib.H)

        if self._ref_pending:
            self._ref_frames.append(self.board)
            self._ref_pending -= 1
            if self._ref_pending == 0:
                self.reference = np.median(np.array(self._ref_frames), axis=0).astype(np.uint8)
                self._ref_H = app.calib.H.copy()
                self._ref_frames = []
        elif self.reference is not None:
            self.cards, self.mask = detect_cards(self.board, self.reference, return_mask=True)
            self.tracker.update(self.cards)

        if self._preview_on:
            self._show_preview(app)

    # --- saving --------------------------------------------------------------
    def counts(self):
        n = {"+": 0, "-": 0}
        for c in self.cards:
            n[c.side] += 1
        return n

    def problems(self):
        return [f for c in self.cards for f in c.flags]

    def can_save(self):
        if self.reference is None or not self.tracker.stable or not self.cards or self.problems():
            return False
        return True

    def save(self, app):
        if not self.can_save():
            self.say("Cannot save yet: " + self.status_text(app)[0])
            return
        folder, rows = save_session(self.data_dir, self.frame, self.board, self.reference,
                                    self.overlay_image(), self.cards, app.calib, LABELS)
        self.last_saved = folder
        self.say(f"Saved {len(rows)} cards")
        print(f"[capture] saved {len(rows)} cards to {folder}")

    def say(self, text, seconds=4.0):
        self.msg, self.msg_until = text, time.time() + seconds

    # --- status line -------------------------------------------------------------
    def status_text(self, app):
        """(text, is_ok)"""
        if cv2 is None:
            return "OpenCV missing", False
        if not app.calib.ok:
            return "Calibrate first (key 2)", False
        if not app.camera.is_on:
            return "Cam OFF", False
        if self._ref_pending:
            return "Taking reference ...", True
        if self.reference is None:
            return "Empty table, then press B", True
        probs = self.problems()
        if "crosses_center" in probs:
            return "Card on the center line", False
        if "touches_hud" in probs:
            return "Card too high: keep below the line", False
        if "too_big" in probs:
            return "Cards touching? move them apart", False
        if "irregular" in probs:
            return "Unclear shape: move cards apart", False
        if not self.cards:
            return "Place the cards", True
        if not self.tracker.stable:
            return "Hold still ...", True
        return "Ready: press S to save", True

    # --- drawing -------------------------------------------------------------------
    def draw(self, surf, app):
        w, h = surf.get_size()
        hud_y = round(cfg.HUD_FRAC * h)
        draw_vline(surf, w // 2)
        size, sy = round(h * 0.055), round(h * 0.055)             # small symbols, high up
        n = self.counts()
        font = app.board.font(h * 0.06)
        for side, cx, draw_sym in (("-", w // 4, draw_minus), ("+", 3 * w // 4, draw_plus)):
            draw_sym(surf, (cx - size, sy), size)
            # number of recognized cards right next to the symbol
            draw_text(surf, font, str(n[side]), (cx + round(size * 0.9), sy),
                      config.CAM_ON_COLOR if n[side] else config.TEXT_COLOR, anchor="midleft")
        pygame.draw.line(surf, (70, 70, 70), (0, hud_y), (w, hud_y), 2)      # cards go below this line

        if time.time() < self.msg_until:
            text, ok = self.msg, True
        else:
            text, ok = self.status_text(app)
        small = app.board.font(h * 0.04)
        img = small.render(text, True, config.TEXT_COLOR if ok else config.CAM_OFF_COLOR)
        rect = img.get_rect(center=(w // 2, round(h * 0.115)))
        pygame.draw.rect(surf, config.BG_COLOR, rect.inflate(24, 6))          # hides the center line behind the text
        surf.blit(img, rect)

    # --- preview on the laptop screen -------------------------------------------------
    def overlay_image(self):
        img = self.board.copy()
        bh, bw = img.shape[:2]
        cv2.line(img, (bw // 2, 0), (bw // 2, bh), (90, 90, 90), 1)
        cv2.line(img, (0, int(cfg.HUD_FRAC * bh)), (bw, int(cfg.HUD_FRAC * bh)), (90, 90, 90), 1)
        for i, c in enumerate(sorted(self.cards, key=lambda c: (c.side, c.center[1], c.center[0])), 1):
            col = BAD if c.flags else COLORS[c.side]
            cv2.polylines(img, [c.box.astype(np.int32)], True, col, 3)
            cv2.putText(img, f"{c.side}{i}", (int(c.center[0]) - 20, int(c.center[1]) + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            if c.flags:                                   # say WHY a card is red
                cv2.putText(img, ",".join(c.flags), (int(c.center[0]) - 70, int(c.center[1]) + 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        return img

    def _show_preview(self, app):
        if self.board is None:
            return
        try:
            img = self.overlay_image()
            if self.show_mask and self.mask is not None:          # D: what the detector sees
                img = cv2.addWeighted(img, 0.35, cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR), 0.65, 0)
            n = self.counts()
            text, ok = self.status_text(app)
            cv2.putText(img, f"-: {n['-']}   +: {n['+']}   {text}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255) if ok else BAD, 2)
            cv2.imshow(PREVIEW_WINDOW, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("b"), ord("B")):
                self.key("b", app)
            elif k in (ord("s"), ord("S")):
                self.key("s", app)
            elif k in (ord("d"), ord("D")):
                self.key("d", app)
        except cv2.error:
            self._preview_on = False
            print("[capture] preview window not available (install opencv-contrib-python)")