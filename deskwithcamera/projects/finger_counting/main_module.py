"""Finger counting: hold one or two hands over the table, the stretched fingers are counted and shown.

Two ways of counting (METHOD button), to compare them:
    Landmarks  MediaPipe finds 21 points per hand; a finger counts as stretched when its tip is far from the wrist.
    Model      a small classifier that learned from YOUR recordings (RECORD): the same hand points, but the
               decision comes from examples instead of a rule.

RECORD (hold a hand on it): the table asks for 0, 1, 2, 3, 4, 5 fingers one after the other (one hand, ~3 s each),
saves the hand points of those seconds (data/samples.npz) and trains the model on ALL recordings made so far.
Afterwards the model is switched on. More rounds = better model.

Buttons (hold a hand on them): HOME, METHOD (Landmarks / Model), RECORD (becomes STOP while recording).
Keys: M = method, R = record / stop (only with DEV_BUTTONS or keys), V = video session for the data set, F = film only what is in data/redo.txt again,
X = (during a session) mark the pattern that just failed (see recorder.py).
Camera preview window on the laptop: M, R.
Every time a NEW number is shown, the table picture is saved in data/captures/ (SAVE_CAPTURES = False switches that off).
The first start downloads the MediaPipe hand model (about 7 MB) once: pip install mediapipe scikit-learn
"""
import json
import threading
import time
from collections import Counter, deque
from datetime import datetime

import numpy as np
import pygame

import config
from core.buttons import ButtonSpec
from core.module import Module
from core.opts import opt
from core.shapes import draw_lower_bg
from projects.finger_counting import hands as H
from projects.finger_counting.recorder import VideoSession, load_redo
from projects.finger_counting.model import DATA_DIR, FingerModel
from projects.mushroom.cards import rectify

try:
    import cv2
except ImportError:
    cv2 = None

DEV_BUTTONS = False                           # True = METHOD / RECORD buttons on the table (for users they are not needed)
SAVE_CAPTURES = True                          # every time a NEW number is shown: save the table picture (data/captures/...)
PREVIEW_WINDOW = "Finger counting   (V = video session, F = redo, X = mark)"
METHODS = ("Landmarks", "Model")
TEXT_COLOR = config.BG_COLOR                  # dark blue on the cream surface (same as the other modules)
DOT_COLOR = (0x1F, 0x4F, 0xD8)
START_DARK = True                         # the lower area is dark (less light on the hand = hardly any shadow); key B switches to the cream surface
DARK_BG = (0x0A, 0x1C, 0x3A)
DOT_DARK = (0x5F, 0xE8, 0xFF)
STEP_SECONDS = 0.05                           # at most 20 detections per second
SMOOTH = 7                                    # the shown number is the most frequent one of the last 7 detections ...
SMOOTH_MIN = 4                                # ... and has to appear at least this often
MARGIN = 0.03                                 # a hand counts if its wrist is on the table (+ this margin)
REC_SEQUENCE = (0, 1, 2, 3, 4, 5)
REC_PREP, REC_TIME, REC_EVERY = 2.5, 3.0, 0.08
REC_MIN, REC_TRIES = 15, 3                    # a number is asked again if fewer recordings came out (max. tries)


class FingerCounting(Module):
    title = "Finger counting"
    area = "Finger counting"
    menu_label = "Finger counting"
    menu_caption = "How many fingers?"

    def __init__(self):
        self.tracker = H.HandTracker()
        self.model = FingerModel()
        self._preview_on = config.CAMERA_PREVIEW
        self.method = 0                       # index in METHODS
        self.reset()

    def reset(self):
        self.hands = []                       # live: dicts u, v, count, tips (table coordinates)
        self.hist = deque(maxlen=SMOOTH)      # last detections: tuples of per-hand counts
        self.shown = ()                       # smoothed counts per hand (left to right), () = no hand
        self.rec = None                       # recording state (None = not recording)
        self.note = ("", 0.0)                 # (text, until) message for the header
        self._t_last = 0.0
        self._err_printed = False
        self._frame = None
        self.dark = START_DARK
        self._fg = TEXT_COLOR
        self.vs = None                        # recording session (key V), None = not recording a video
        self.in_buttons = False               # a hand is (partly) in the button area: it is not counted

    # --- lifecycle -------------------------------------------------------------
    def on_enter(self, app):
        self.reset()
        self.tracker.start()
        if self.model.load():
            self.method = 1
            print(f"[fingers] trained model loaded: {self.model.info}")
        else:
            self.method = 0

    def on_exit(self, app):
        self.stop_video()
        if cv2 is not None and self._preview_on:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
            except cv2.error:
                pass

    def handle_event(self, event, app):
        if event.type == pygame.KEYDOWN:
            ch = {pygame.K_m: "m", pygame.K_r: "r", pygame.K_v: "v", pygame.K_f: "f", pygame.K_x: "x", pygame.K_b: "b"}.get(event.key)
            if ch:
                self.key(ch, app)

    def key(self, ch, app=None):
        if ch == "m" and self.model.ready and self.rec is None:
            self.method = (self.method + 1) % len(METHODS)
            self.hist.clear()
        elif ch == "b":                                          # dark <-> cream surface (to compare how well the hand is seen)
            self.dark = not self.dark
            if self.vs is not None:
                self.vs.dark = self.dark
            self.note = ("Surface: " + ("dark" if self.dark else "cream"), time.time() + 3)
        elif ch == "v":
            self.stop_video() if self.vs is not None else self.start_video(app)
        elif ch == "f" and self.vs is None:
            self.start_video(app, redo=True)
        elif ch == "x" and self.vs is not None:
            self.vs.mark(time.time())
        elif ch == "r":
            if self.rec is None:
                self.start_record(app)
            else:
                self.stop_record(app)

    # --- video session (key V) ------------------------------------------------------
    def start_video(self, app, redo=False):
        items = None
        if redo:
            items = load_redo()
            if not items:
                self.note = ("data/redo.txt is empty: nothing to film again", time.time() + 6)
                return
        frame = app.camera.latest() if app is not None else None
        if frame is None or cv2 is None:
            self.note = ("No camera picture: cannot record", time.time() + 5)
            return
        try:
            self.vs = VideoSession(app, frame, redo=items)
            self.vs.dark = self.dark
            self.hands, self.shown = [], ()
        except Exception as e:
            print(f"[fingers] could not start the recording session: {e}")
            self.note = ("Could not start the recording", time.time() + 5)

    def stop_video(self):
        if self.vs is not None:
            vs, self.vs = self.vs, None
            vs.finish(vs.now(time.time()))
            self.note = ("Session saved", time.time() + 6)

    # --- buttons -----------------------------------------------------------------
    def buttons(self, app):
        if not DEV_BUTTONS:
            return []
        specs = [ButtonSpec("method", "METHOD", lambda app: self.key("m", app), "right",
                            enabled=lambda app: self.model.ready and self.rec is None,
                            caption=lambda app: METHODS[self.method])]
        if self.rec is None:
            specs.append(ButtonSpec("record", "RECORD", lambda app: self.start_record(app), "right2",
                                    enabled=lambda app: self.tracker.ready,
                                    caption=lambda app: self._samples_caption()))
        else:
            specs.append(ButtonSpec("stop", "STOP", lambda app: self.stop_record(app), "right2"))
        return specs

    def _samples_caption(self):
        n = sum(self.model.counts().values())
        return f"{n} recordings" if n else ""

    # --- recording ---------------------------------------------------------------
    def start_record(self, app=None):
        if not self.tracker.ready:
            return
        self.rec = dict(i=0, phase="prep", t=REC_PREP, tries=1, last=0.0, X=[], y=[], cur=[])
        self.hist.clear()
        self.shown = ()
        if app is not None:
            app.refresh_buttons()

    def stop_record(self, app=None, message="Recording stopped"):
        self.rec = None
        self.note = (message, time.time() + 5)
        if app is not None:
            app.refresh_buttons()

    def _record_step(self, dt, app, hands):
        r = self.rec
        n = REC_SEQUENCE[r["i"]]
        r["t"] -= dt
        if r["phase"] == "rec" and len(hands) == 1 and time.time() - r["last"] >= REC_EVERY:
            r["last"] = time.time()
            f = H.features(hands[0].pts)
            r["cur"] += [f, H.mirrored(f)]                  # both versions: left and right hand look like mirror images
        if r["t"] > 0:
            return
        if r["phase"] == "prep":
            r["phase"], r["t"], r["cur"] = "rec", REC_TIME, []
            return
        if len(r["cur"]) < 2 * REC_MIN and r["tries"] < REC_TRIES:       # hand was not seen long enough: once more
            r["tries"] += 1
            r["phase"], r["t"] = "prep", REC_PREP
            return
        r["X"] += r["cur"]
        r["y"] += [n] * len(r["cur"])
        r["i"], r["tries"], r["phase"], r["t"] = r["i"] + 1, 1, "prep", REC_PREP
        if r["i"] >= len(REC_SEQUENCE):
            self._finish_record(app)

    def _finish_record(self, app):
        r = self.rec
        self.model.add_samples(r["X"], r["y"])
        ok, msg = self.model.train()
        if ok:
            self.method = 1
            msg = f"Model ready: {msg}"
        self.stop_record(app, msg)

    # --- per frame ---------------------------------------------------------------
    def update(self, dt, app):
        if self.vs is not None:                               # recording a video: nothing else is calculated
            frame = app.camera.latest()
            self.vs.step(time.time(), frame)
            self._video_preview(frame, app)
            if self.vs is None:                                   # stopped with V in the preview window
                return
            if self.vs.done:
                self.vs = None
                self.note = ("Session saved", time.time() + 8)
            return
        if cv2 is None or not app.calib.ok or not self.tracker.ready:
            return
        now = time.time()
        if now - self._t_last < STEP_SECONDS:
            return
        step = now - self._t_last if self._t_last else dt
        self._t_last = now
        frame = app.camera.latest()
        if frame is None:
            return
        self._frame = frame
        try:
            found = self.tracker.detect(frame)
        except Exception as e:
            if not self._err_printed:
                print(f"[fingers] detection failed: {e}")
                self._err_printed = True
            return
        on_table = []
        self.in_buttons = False
        top = opt("HUD_FRAC")                                              # the button row: the band above the table area
        for hd in found:
            nrm = H.to_norm(app.calib.H, hd.pts)
            wrist = nrm[0]
            if not (-MARGIN <= wrist[0] <= 1 + MARGIN and -MARGIN <= wrist[1] <= 1 + MARGIN):
                continue                                                   # not on the table at all
            if nrm[:, 1].min() < top:                                      # any part of the hand in the button area: not counted
                self.in_buttons = True
                continue
            on_table.append(hd)
        on_table.sort(key=lambda hd: float(hd.pts[0, 0]))                     # left to right
        self.hands = [self._describe(hd, app) for hd in on_table]
        if self.rec is not None:
            self._record_step(step, app, on_table)
            return
        counts = tuple(x["count"] for x in self.hands) if len(self.hands) <= 2 else ()
        self.hist.append(counts)
        common, n = Counter(self.hist).most_common(1)[0]
        if n >= SMOOTH_MIN or (not common and n >= 2):
            if common and common != self.shown and SAVE_CAPTURES:
                self._save_capture(frame, on_table, app, common)
            self.shown = common
        if self._preview_on:
            self._show_preview(app, on_table)

    def _save_capture(self, frame, hands, app, shown):
        """A new number is on the table: save what the camera saw (own thread, the table does not stutter).
        data/captures/<time>_<number>/  frame.png (camera), board.png (top-down table), overlay.png (hand points + values), info.json"""
        try:
            board = rectify(frame, app.calib.H)
            info = dict(time=datetime.now().isoformat(timespec="seconds"), method=METHODS[self.method], shown=list(shown),
                        total=int(sum(shown)), hands=[])
            overlay = frame.copy()
            for hd, d in zip(hands, self.hands):
                pts = hd.pts
                info["hands"].append(dict(label=hd.label, counted=d["count"], stretched=[bool(s) for s in H.finger_states(pts)],
                                          straightness=[round(float(v), 3) for v in H.finger_straightness(pts)],
                                          thumb_away=round(float(H.thumb_reach(pts)), 3)))
                for x, y, _ in pts:
                    cv2.circle(overlay, (int(x), int(y)), 6, (255, 170, 60), -1)
                for i, t in enumerate(H.TIPS):
                    cv2.circle(overlay, (int(pts[t, 0]), int(pts[t, 1])), 12, (0, 0, 255), 2)
                st = " ".join(f"{v:.2f}" for v in H.finger_straightness(pts))
                x0, y0 = int(pts[0, 0]), int(pts[0, 1])
                cv2.putText(overlay, f"{d['count']}   straight: {st}", (x0, min(y0 + 60, overlay.shape[0] - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                cv2.putText(overlay, f"thumb away {H.thumb_reach(pts):.2f}", (x0, min(y0 + 105, overlay.shape[0] - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            cv2.putText(overlay, f"{METHODS[self.method]}: {info['total']}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
            folder = DATA_DIR / "captures" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_n{info['total']}"
            threading.Thread(target=self._write_capture, args=(folder, frame.copy(), board, overlay, info), daemon=True).start()
        except Exception as e:
            print(f"[fingers] could not save the capture: {e}")

    @staticmethod
    def _write_capture(folder, frame, board, overlay, info):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            for name, img in (("frame.png", frame), ("board.png", board), ("overlay.png", overlay)):
                cv2.imwrite(str(folder / name), img)
            (folder / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            print(f"[fingers] saved {folder.name}")
        except Exception as e:
            print(f"[fingers] could not save the capture: {e}")

    def _describe(self, hd, app):
        p = hd.pts
        if self.method == 1 and self.model.ready:
            bits = self.model.predict_bits(H.features(p))
            if bits is not None:                                         # per-finger model: it says which fingers
                count, marked = int(sum(bits)), [i for i, b in enumerate(bits) if b]
            else:
                count = self.model.predict(H.features(p))
                marked = H.tips_by_reach(p, count)
        else:
            states = H.finger_states(p)
            count = int(sum(states))
            marked = [i for i, s in enumerate(states) if s]
        tips = H.to_norm(app.calib.H, p[list(H.TIPS)])
        return dict(count=count, tips=[tuple(tips[i]) for i in marked])

    # --- header hint ---------------------------------------------------------------
    def hint(self, app):
        if cv2 is None:
            return "OpenCV missing", "warn"
        if not app.calib.ok:
            return "Calibrate first (key 2)", "warn"
        if not app.camera.is_on:
            return "Cam OFF", "warn"
        if self.vs is not None:
            return self.vs.hint(time.time())
        if self.tracker.status == "error":
            return self.tracker.message, "warn"
        if not self.tracker.ready:
            return "Loading the hand model ...", "info"
        r = self.rec
        if r is not None:
            n = REC_SEQUENCE[r["i"]]
            word = "finger" if n == 1 else "fingers"
            if r["phase"] == "prep":
                return f"Show {n} {word} with ONE hand: recording in {int(r['t']) + 1}", "blink"
            if len(self.hands) != 1:
                return ("Move your hand over the table" if self.in_buttons and not self.hands else "Show ONE hand over the table"), "warn"
            return f"Recording {n} {word} ... hold still", "info"
        if self.note[0] and time.time() < self.note[1]:
            return self.note[0], "info"
        if len(self.hands) > 2:
            return "Two hands at most", "warn"
        if not self.hands:
            if self.in_buttons:
                return "Move your hand over the table", "blink"
            return "Show your hand(s) over the table", "blink"
        return "Counting your fingers", "info"

    # --- drawing -----------------------------------------------------------------
    def draw(self, surf, app):
        w, h = surf.get_size()
        hud_y = round(opt("HUD_FRAC") * h)
        if self.dark:
            r = max(0, config.CORNER_RADIUS - config.LINE_WIDTH)
            pygame.draw.rect(surf, DARK_BG, (0, hud_y, w, h - hud_y), border_bottom_left_radius=r, border_bottom_right_radius=r)
            pygame.draw.line(surf, config.LINE_COLOR, (0, hud_y), (w, hud_y), max(2, config.LINE_WIDTH // 2))
            self._fg = config.TEXT_COLOR
        else:
            draw_lower_bg(surf)                             # no center line in this module
            self._fg = TEXT_COLOR
        area = pygame.Rect(0, hud_y, w, h - hud_y)
        if self.vs is not None:
            self.vs.draw(surf, app, time.time(), w, h)
            return
        if self.rec is not None:
            self._draw_target(surf, app, area, w, h)
        elif self.shown and self.hands:
            self._draw_count(surf, app, area, w, h)
        for hd in self.hands:                               # a dot on every stretched fingertip
            for u, v in hd["tips"]:
                c = (int(u * w), int(v * h))
                pygame.draw.circle(surf, DOT_DARK if self.dark else DOT_COLOR, c, round(h * 0.02))
                pygame.draw.circle(surf, (255, 255, 255), c, round(h * 0.02), max(2, round(h * 0.004)))

    def _draw_count(self, surf, app, area, w, h):
        total = int(sum(self.shown))
        big = app.board.font(h * 0.34, "Bold").render(str(total), True, self._fg)
        surf.blit(big, big.get_rect(center=(w // 2, area.centery - h * 0.03)))
        word = app.board.font(h * 0.06).render("finger" if total == 1 else "fingers", True, self._fg)
        surf.blit(word, word.get_rect(midtop=(w // 2, area.centery + h * 0.13)))
        if len(self.shown) == 2:                            # both hands: the two numbers at the bottom
            for x, side, c in ((0.22, "left", self.shown[0]), (0.78, "right", self.shown[1])):
                t = app.board.font(h * 0.07, "Bold").render(str(c), True, self._fg)
                s = app.board.font(h * 0.04).render(side, True, self._fg)
                surf.blit(t, t.get_rect(midbottom=(w * x, area.bottom - h * 0.07)))
                surf.blit(s, s.get_rect(midbottom=(w * x, area.bottom - h * 0.025)))

    def _draw_target(self, surf, app, area, w, h):
        """While recording: the number you are asked to show, and a progress bar."""
        r = self.rec
        n = REC_SEQUENCE[r["i"]]
        big = app.board.font(h * 0.34, "Bold").render(str(n), True, self._fg)
        surf.blit(big, big.get_rect(center=(w // 2, area.centery - h * 0.03)))
        bar = pygame.Rect(0, 0, round(w * 0.3), round(h * 0.018))
        bar.midtop = (w // 2, area.centery + h * 0.16)
        pygame.draw.rect(surf, self._fg, bar, max(2, round(h * 0.004)), border_radius=bar.h // 2)
        total = REC_PREP if r["phase"] == "prep" else REC_TIME
        done = 1 - max(r["t"], 0) / total
        if r["phase"] == "rec":
            fill = bar.copy()
            fill.w = max(round(bar.w * done), bar.h)
            pygame.draw.rect(surf, self._fg, fill, border_radius=bar.h // 2)
        step = app.board.font(h * 0.04).render(f"{r['i'] + 1} / {len(REC_SEQUENCE)}", True, self._fg)
        surf.blit(step, step.get_rect(midtop=(w // 2, bar.bottom + h * 0.02)))

    # --- laptop preview ------------------------------------------------------------
    def _video_preview(self, frame, app):
        """During a video session the preview window stays alive (and listens to V / X), without any hand detection."""
        if not self._preview_on or frame is None:
            return
        now = time.time()
        if now - getattr(self, "_t_vprev", 0.0) < 0.1:
            return
        self._t_vprev = now
        try:
            img = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            cv2.circle(img, (40, 40), 14, (0, 0, 230), -1)
            cv2.putText(img, "REC  (V = stop and save, X = mark)", (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 230), 2)
            cv2.imshow(PREVIEW_WINDOW, img)
            k = cv2.waitKey(1) & 0xFF
            ch = chr(k).lower() if 0 < k < 128 else ""
            if ch in ("v", "x", "b"):
                self.key(ch, app)
        except cv2.error:
            self._preview_on = False

    def _show_preview(self, app, hands):
        if self._frame is None:
            return
        try:
            img = self._frame.copy()
            hh, ww = img.shape[:2]
            for hd, info in zip(hands, self.hands):
                for x, y, _ in hd.pts:
                    cv2.circle(img, (int(x), int(y)), 5, (255, 170, 60), -1)
                cv2.putText(img, str(info["count"]), (int(hd.pts[0, 0]), int(hd.pts[0, 1]) + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                st = " ".join(f"{v:.2f}" for v in H.finger_straightness(hd.pts))      # to tune H.STRAIGHT_FINGER / STRAIGHT_THUMB
                cv2.putText(img, f"straight (thumb..pinky): {st}", (int(hd.pts[0, 0]), int(hd.pts[0, 1]) + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(img, f"thumb away from hand: {H.thumb_reach(hd.pts):.2f} (> {H.THUMB_REACH})",
                            (int(hd.pts[0, 0]), int(hd.pts[0, 1]) + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(img, METHODS[self.method], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            img = cv2.resize(img, (960, int(hh * 960 / ww)))
            cv2.imshow(PREVIEW_WINDOW, img)
            k = cv2.waitKey(1) & 0xFF
            ch = chr(k).lower() if 0 < k < 128 else ""
            if ch in ("m", "r", "v", "f", "x", "b"):          # same keys as in the projector window
                self.key(ch, app)
        except cv2.error:
            self._preview_on = False
