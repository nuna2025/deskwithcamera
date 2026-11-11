"""Classification module: teach the computer with cards, then test it with a new card.

Step 1  "teach me":  put NEGATIVE cards on the left, POSITIVE cards on the right (completely inside their half, not
                     touching). Wait until the header says "Ready", then hold a hand on LEARN.
Clear the table:     the cards are saved (data/sessions/...); "Take the teaching cards away". The model already starts
                     learning in the background.
Learning:            when the table is empty: "Learning ..." with a moving bar until the model is ready (it learns from the
                     cards and from the saved sessions, see LEARN_USE_SAVED in settings.py).
Step 2  "test":      put a card to test on the table: a colored ring and a label say "negative" (red) or "positive" (blue)
                     with the confidence. AGAIN = teach from the start.

Before: calibrate (key 2). The empty-table picture is taken by itself a few seconds after the module starts.
Buttons (hold a hand on them): HOME, RESET (look at the table again), LEARN / AGAIN
Keys work in the projection window AND in the camera preview window on the laptop:
    B = new empty-table picture   S = only save the cards   L = learn   A = again   D = show/hide the detector's mask
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
from projects.mushroom import settings as cfg
from projects.mushroom.cards import StabilityTracker, clean_reference, detect_cards, rectify
from projects.mushroom import screens
from projects.mushroom.cards import crop_inner, crop_raw, letterbox
from projects.mushroom.dataset import load_cards, save_session
from projects.mushroom.learner import Learner
from projects.mushroom.plus_minus_split import LABELS

try:
    import cv2
except ImportError:
    cv2 = None

PREVIEW_WINDOW = "Card capture   (B = empty table, S = save, L = learn, A = again, D = mask)"
COLORS = {"+": (60, 120, 255), "-": (255, 170, 60)}          # BGR: + orange-red, - blue
BAD = (0, 0, 255)


REF_DELAY = 3.0              # seconds after the module starts until the empty-table picture is taken
RESET_DELAY = 2.5            # ... after RESET
HAND_GRACE = 2.5             # a huge / irregular blob (hand, arm, shadow) shorter than this is ignored: the last result stays


class CardCapture(Module):
    title = "Card capture"
    area = "Classification"
    menu_label = "Classification"
    menu_caption = "Teach and test cards"

    def __init__(self):
        self.learner = Learner()
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
        self.phase = "learn"             # learn (step 1) | clear (take the cards away) | learning | test (step 2)
        self._phase_t = 0.0              # seconds in the current phase
        self.n_train = (0, 0)            # (negative, positive) cards the model learns from
        self._clear_t = 0.0              # seconds the table has been empty in the 'clear' phase
        self._results = []               # [(center, side, confidence)] classified cards (step 2)

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
            elif event.key == pygame.K_l:
                self.key("l", app)
            elif event.key == pygame.K_a:
                self.key("a", app)

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
        elif ch == "l" and self.phase == "learn":
            self.start_learning(app)
        elif ch == "a" and self.phase == "test":
            self.again(app)

    # --- buttons -----------------------------------------------------------
    def buttons(self, app):
        if self.phase in ("clear", "learning"):
            return []
        if self.phase == "test":
            return [ButtonSpec("again", "AGAIN", lambda app: self.again(app), "right")]
        return [
            ButtonSpec("learn", "LEARN", lambda app: self.start_learning(app), "right", enabled=lambda app: self.can_save()),
            ButtonSpec("reset", "RESET", lambda app: self.reset_reference(), "right2", hold=3.0),   # destroys the cards: longer hold
        ]

    def _set_phase(self, phase, app):
        self.phase, self._phase_t = phase, 0.0
        app.refresh_buttons()

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
        self._phase_t += dt
        if self.phase in ("clear", "learning"):
            self._poll_learning(app)
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
        elif self.reference is not None and self.phase != "learning":
            cards, self.mask = detect_cards(self.board, self.reference, return_mask=True)
            odd = any({"too_big", "irregular"} & set(c.flags) for c in cards)     # not a card: hand, arm, shadow
            self._hand_t = self._hand_t + dt if odd else 0.0
            self._hand = odd and self._hand_t < HAND_GRACE
            if not self._hand:                            # (while a hand is in view the last result stays)
                self.cards = cards
                self.tracker.update(self.cards)
                if self.phase == "test":
                    self._classify_cards()
                elif self.phase == "clear":
                    self._clear_t = self._clear_t + dt if not self.cards else 0.0
                    if self._clear_t >= 1.0:                      # the table has been empty for a second
                        self._set_phase("learning", app)

        if self._preview_on:
            self._show_preview(app)

    # --- learn and test -------------------------------------------------------
    def start_learning(self, app):
        """LEARN: save the cards on the table, then let the model learn from them (and the saved sessions)."""
        if self.phase != "learn":
            return
        if not self.can_save():
            self.say("Cannot learn yet: " + self.status_text(app)[0])
            return
        folder, rows = save_session(self.data_dir, self.frame, self.board, self.reference,
                                    self.overlay_image(), self.cards, app.calib, LABELS)
        self.last_saved = folder
        images, labels, _ = load_cards(self.data_dir, None if cfg.LEARN_USE_SAVED else [folder.name])
        n_neg, n_pos = labels.count("safe"), labels.count("poisonous")
        if n_neg < 1 or n_pos < 1:
            self.say("Learning needs cards on BOTH sides: negative (left) and positive (right)", 6.0)
            return
        self.n_train = (n_neg, n_pos)
        print(f"[learn] {n_neg} negative + {n_pos} positive cards, model {cfg.MODEL_ID}")
        self.learner.start(images, labels, cfg.MODEL_ID)          # learns in the background while the cards are taken away
        self._clear_t = 0.0
        self._set_phase("clear", app)

    def _poll_learning(self, app):
        st = self.learner.state
        if st == "ready" and self.phase == "learning" and self._phase_t >= 1.5:   # (at least 1.5 s on this screen)
            if self.learner.note:
                self.say(self.learner.note, 6.0)
            self._results = []
            self.tracker.reset()
            self._set_phase("test", app)
        elif st == "error":
            self.say("Learning failed: " + self.learner.error, 8.0)
            self._set_phase("learn", app)

    def again(self, app):
        """AGAIN: back to step 1 (teach from the start)."""
        self.cards, self._results = [], []
        self.tracker.reset()
        self._set_phase("learn", app)

    def _board_wh(self):
        if self.board is not None:
            return self.board.shape[1], self.board.shape[0]
        return cfg.BOARD_SIZE

    def _classify_cards(self):
        """Step 2: classify every card once it holds still (and again if it moved)."""
        bw, _ = self._board_wh()
        tol_move = cfg.PREDICT_MOVE_TOL_FRAC * bw
        results = []
        for c in self.cards:
            if c.flags:
                continue
            hit = [r for r in self._results if math.hypot(r[1][0] - c.center[0], r[1][1] - c.center[1]) <= tol_move]
            if hit:
                results.append((c.box, c.center, hit[0][2], hit[0][3]))
            elif self.tracker.stable and self.learner.state == "ready":
                crop = letterbox(crop_inner(crop_raw(self.board, c)))
                side, conf = self.learner.classify(crop)
                results.append((c.box, c.center, side, conf))
        self._results = results

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
        if self.phase == "learning":
            return "Learning ...", True
        if self.phase == "clear":
            return "Take the teaching cards away - learning starts when the table is empty", True
        if self._ref_countdown > 0:
            return f"Clear the table: reference in {math.ceil(self._ref_countdown)}", True
        if self._ref_pending:
            return "Taking reference ...", True
        if self.reference is None:
            return "Waiting for the camera ...", True
        if self._hand:
            return "Hand in view ...", True
        probs = self.problems()
        if self.phase == "test":
            probs = [p for p in probs if p != "crosses_center"]       # no center line in step 2
        if "crosses_center" in probs:
            return "Card on the center line", False
        if "touches_hud" in probs:
            return "Card too high: keep below the line", False
        if "too_big" in probs:
            return "Cards touching? move them apart", False
        if "irregular" in probs:
            return "Unclear shape: move cards apart", False
        if self.phase == "test":
            if not self.cards:
                return "Step 2 of 2: put a card to test - positive or negative?", True
            if not self.tracker.stable:
                return "Hold still ...", True
            return "Put another card to test, or hold a hand on AGAIN", True
        if not self.cards:
            return "Step 1 of 2: teach me - negative cards left, positive cards right", True
        if not self.tracker.stable:
            return "Hold still ...", True
        return "Ready: hold your hand on LEARN", True

    def hint(self, app):
        """(text, mode) for the second header line."""
        if time.time() < self.msg_until:
            return self.msg, "info"
        text, ok = self.status_text(app)
        if not ok:
            return text, "warn"
        if text.startswith(("Ready", "Clear the table", "Step", "Take the teaching")):
            return text, "blink"                # needs the user to act
        return text, "info"

    # --- drawing -------------------------------------------------------------------
    def draw(self, surf, app):
        if self.phase == "learning":
            screens.draw_learning(surf, app, self._phase_t, *self.n_train)
        elif self.phase == "test":
            items = [(box, center, side, conf) for box, center, side, conf in self._results]
            screens.draw_result(surf, app, items, self._board_wh())
        else:                                              # step 1 - and also while the teaching cards are taken away
            screens.draw_teach(surf, app, self.counts())

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
            elif k in (ord("l"), ord("L")):
                self.key("l", app)
            elif k in (ord("a"), ord("A")):
                self.key("a", app)
        except cv2.error:
            self._preview_on = False
            print("[capture] preview window not available (install opencv-contrib-python)")
