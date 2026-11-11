"""Card capture module: find the cards on the table, count them per side, save them as a dataset session.

Workflow
    1. Calibrate first (key 2). The camera must see the whole frame.
    2. Clear the table: the empty-table reference picture is taken by itself a few seconds after the
       module starts. Key: B = take a new one now (e.g. after the light or the camera changed).
       RESET keeps it and just looks at the table again (cards already lying there are found again).
    3. Put the cards BELOW the gray line, completely inside their half (+ right, - left), not touching
    4. Wait until the header says "Ready"  ->  hold a hand on SAVE  (or press S)

Buttons (hold a hand on them): HOME (if there is a Home screen), RESET, SAVE
Keys work in the projection window AND in the camera preview window on the laptop:
    B = empty-table reference     S = save session     D = show/hide the detector's mask (debugging)
    (ESC / C / Tab are handled by the app)
"""
import math
import time

import numpy as np
import pygame

import config
from core.buttons import ButtonSpec, row_center
from core.module import Module
from core.opts import opt
from core.shapes import draw_lower_bg, draw_minus, draw_plus, draw_split_line
from projects.mushroom import settings as cfg
from projects.mushroom.cards import StabilityTracker, clean_reference, detect_cards, rectify
from projects.mushroom.dataset import save_session
from projects.mushroom.plus_minus_split import LABELS

try:
    import cv2
except ImportError:
    cv2 = None

PREVIEW_WINDOW = "Card capture   (B = empty table, S = save, D = mask)"
COLORS = {"+": (60, 120, 255), "-": (255, 170, 60)}          # BGR: + orange-red, - blue
BAD = (0, 0, 255)


REF_DELAY = 3.0              # seconds after the module starts until the empty-table picture is taken
RESET_DELAY = 2.5            # ... after RESET
HAND_GRACE = 2.5             # a huge / irregular blob (hand, arm, shadow) shorter than this is ignored: the last result stays


class CardCapture(Module):
    title = "Card capture"
    area = "Classification"
    menu_label = "Classification"
    menu_caption = "Mushroom cards"

    def __init__(self):
        self.reset()
        self._preview_on = config.CAMERA_PREVIEW
        self.data_dir = cfg.DATA_DIR

    def reset(self):
        self.reference = None
        self._ref_H = None
        self._ref_frames = []
        self._ref_pending = 0
        self._ref_countdown = 0.0        # > 0: seconds until the reference is taken (hands away!)
        self._hand = False               # a hand / arm / shadow is in the picture right now
        self._hand_t = 0.0
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
        self._ref_countdown = REF_DELAY

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
            self._ref_countdown = 0.0
            self._ref_frames, self._ref_pending = [], cfg.REFERENCE_FRAMES
            self.cards = []
            self.tracker.reset()
        elif ch == "s":
            self.save(app)
        elif ch == "d":
            self.show_mask = not self.show_mask

    # --- buttons -----------------------------------------------------------
    def buttons(self, app):
        return [
            ButtonSpec("save", "SAVE", lambda app: self.save(app), "right", enabled=lambda app: self.can_save()),
            ButtonSpec("reset", "RESET", lambda app: self.reset_reference(), "right2", hold=3.0),   # destroys the cards: longer hold
        ]

    def reset_reference(self):
        """RESET: forget the cards found so far and look at the table again.

        The empty-table picture is KEPT, so cards that are already lying on the table are found again
        right away (no need to clear the table). Only if there is no empty-table picture yet, a new one
        is taken (clear the table first). Key B always takes a new empty-table picture."""
        self.cards = []
        self._hand, self._hand_t = False, 0.0
        self.tracker.reset()
        if self.reference is not None:
            self.say("Looking at the table again ...", 2.0)
            return
        self._ref_frames, self._ref_pending = [], 0
        self._ref_countdown = RESET_DELAY

    # --- per frame -----------------------------------------------------------
    def update(self, dt, app):
        if cv2 is None or not app.calib.ok:
            return
        frame = app.camera.latest()
        if frame is None:
            return
        if self._ref_countdown > 0:
            self._ref_countdown -= dt
            if self._ref_countdown <= 0:
                self.key("b", app)
        if app.panel.busy():                 # a hand is on a button: keep the last picture (cards, board) until it leaves
            return
        self.frame = frame
        if self._ref_H is not None and not np.allclose(self._ref_H, app.calib.H):
            self.reference = None                         # calibration changed -> old reference is invalid
        self.board = rectify(frame, app.calib.H)

        if self._ref_pending:
            self._ref_frames.append(self.board)
            self._ref_pending -= 1
            if self._ref_pending == 0:
                ref = np.median(np.array(self._ref_frames), axis=0).astype(np.uint8)
                self.reference, n_there = clean_reference(ref)      # cards that already lie on the table are not 'empty table'
                if n_there:
                    self.say(f"{n_there} card(s) were already on the table", 3.0)
                self._ref_H = app.calib.H.copy()
                self._ref_frames = []
        elif self.reference is not None:
            cards, self.mask = detect_cards(self.board, self.reference, return_mask=True)
            odd = any({"too_big", "irregular"} & set(c.flags) for c in cards)     # not a card: hand, arm, shadow
            self._hand_t = self._hand_t + dt if odd else 0.0
            self._hand = odd and self._hand_t < HAND_GRACE
            if not self._hand:                            # (while a hand is in view the last result stays)
                self.cards = cards
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
        if self.reference is None or self._hand or not self.tracker.stable or not self.cards or self.problems():
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
        if self._ref_countdown > 0:
            return f"Clear the table: reference in {math.ceil(self._ref_countdown)}", True
        if self._ref_pending:
            return "Taking reference ...", True
        if self.reference is None:
            return "Waiting for the camera ...", True
        if self._hand:
            return "Hand in view ...", True
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
        return "Ready: hold your hand on SAVE", True

    def hint(self, app):
        """(text, mode) for the second header line."""
        if time.time() < self.msg_until:
            return self.msg, "info"
        text, ok = self.status_text(app)
        if not ok:
            return text, "warn"
        if text.startswith(("Ready", "Clear the table", "Place the cards")):
            return text, "blink"                # needs the user to act
        return text, "info"

    # --- drawing -------------------------------------------------------------------
    def draw(self, surf, app):
        w, h = surf.get_size()
        draw_lower_bg(surf)                                        # light area below the button row: cards go here
        draw_split_line(surf, w // 2)                              # center line: light above, dark on the lower area
        cy = row_center(h)                                         # counters share the row of the buttons
        size = round(h * 0.055)
        font = app.board.font(h * 0.07, "Black")
        n = self.counts()
        g = round(0.08 * w - size / 2)                             # distance center line -> symbol (same on both sides)
        for side in ("-", "+"):
            img = font.render(str(n[side]), True, config.CAM_ON_COLOR if n[side] else config.TEXT_COLOR)
            if side == "-":                                        # number next to the center line, symbol to its left
                rect = img.get_rect(midright=(w // 2 - g, cy))
                sx = rect.left - round(size * 0.45) - size // 2
                draw_minus(surf, (sx, cy), size)
            else:                                                  # symbol next to the center line, number to its right
                sx = w // 2 + g + size // 2
                draw_plus(surf, (sx, cy), size)
                rect = img.get_rect(midleft=(sx + round(size * 0.95), cy))
            surf.blit(img, rect)

    # --- preview on the laptop screen -------------------------------------------------
    def overlay_image(self):
        img = self.board.copy()
        bh, bw = img.shape[:2]
        cv2.line(img, (bw // 2, 0), (bw // 2, bh), (90, 90, 90), 1)
        cv2.line(img, (0, int(opt("HUD_FRAC") * bh)), (bw, int(opt("HUD_FRAC") * bh)), (90, 90, 90), 1)
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
