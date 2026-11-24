"""Recording session (for YOU, not for the users): ONE continuous video of your left hand showing every finger pattern.

The table tells you what to show. The video is not cut during the recording: labels.csv says when each pattern was
shown (in video seconds), so how long a piece is kept can be decided later.

Label = five digits, one per finger:  thumb index middle ring pinky   (1 = stretched, 0 = folded)
    01100 = index + middle        10000 = thumb only        00000 = fist        11111 = open hand
Saved in data/videos/<time>/:  session.mp4 (camera, full resolution) | labels.csv | session.json (settings, calibration)
Keys (module Finger counting):  V = start / stop the full session
                                F = film again ONLY the patterns listed in data/redo.txt
                                X = during a session: 'this one did not work' -> it is added to data/redo.txt
redo.txt: one pattern per line:  label[,take[,note]]   e.g.  01100,1   or   00000,2,fist thumb inside   (# = comment)
"""
import csv
import itertools
import json
import queue
import threading
import time
from datetime import datetime

import numpy as np
import pygame

import config
from core.opts import opt
from projects.finger_counting.model import DATA_DIR

try:
    import cv2
except ImportError:
    cv2 = None

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
INTRO_SECONDS = 20.0      # explanation + time to get ready, before the first pattern
PREP_SECONDS = 4.0        # the next pattern is shown: get your hand into shape
HOLD_SECONDS = 8.0        # recording: turn and tilt the hand slowly
REST_SECONDS = 10.0       # between the takes
REDO_INTRO = 8.0         # before a redo session
REDO_FILE = DATA_DIR / "redo.txt"
TAKES = 2                 # take 1 = for training, take 2 = for testing (different video, no leakage)
FPS = 20
EXTRA = {"00000": ("fist, thumb inside", "fist, thumb outside"), "11111": ("fingers spread", "fingers together")}
TEXT_COLOR = config.BG_COLOR


def name_of(bits):
    on = [FINGERS[i] for i, b in enumerate(bits) if b == "1"]
    return " + ".join(on) if on else "no finger"


def patterns():
    """[(bits, note)] sorted by number of fingers: 00000, then the 5 one-finger patterns, ... 11111 (+ variants of 0 and 5)."""
    out = []
    for n in range(6):
        for combo in itertools.product("01", repeat=5):
            bits = "".join(combo)
            if bits.count("1") == n:
                out.append((bits, ""))
                for note in EXTRA.get(bits, ()):
                    out.append((bits, note))
    return out


def load_redo():
    """Patterns listed in data/redo.txt -> [dict(bits, take, note)]."""
    items = []
    if REDO_FILE.exists():
        for line in REDO_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",", 2)]
            bits = parts[0]
            if len(bits) != 5 or set(bits) - set("01"):
                continue
            take = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            items.append(dict(bits=bits, take=take, note=parts[2] if len(parts) > 2 else ""))
    return items


def save_redo(items):
    REDO_FILE.parent.mkdir(parents=True, exist_ok=True)
    seen, lines = set(), []
    for it in items:
        key = (it["bits"], it["take"], it["note"])
        if key not in seen:
            seen.add(key)
            lines.append(",".join([it["bits"], str(it["take"])] + ([it["note"]] if it["note"] else [])))
    REDO_FILE.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_schedule(items=None):
    """Segments in video time: dict(kind intro|prep|rec|rest, start, end, take, step, bits, note).
    items = a redo list: only those patterns, no rests."""
    seg, t = [], 0.0

    def add(kind, dur, **kw):
        nonlocal t
        seg.append(dict(kind=kind, start=t, end=t + dur, **kw))
        t += dur
    if items is not None:
        add("intro", REDO_INTRO, take=1, step=0, bits="", note="")
        for i, it in enumerate(items, 1):
            add("prep", PREP_SECONDS, take=it["take"], step=i, bits=it["bits"], note=it["note"])
            add("rec", HOLD_SECONDS, take=it["take"], step=i, bits=it["bits"], note=it["note"])
        return seg
    add("intro", INTRO_SECONDS, take=1, step=0, bits="", note="")
    pats = patterns()
    for take in range(1, TAKES + 1):
        if take > 1:
            add("rest", REST_SECONDS, take=take, step=0, bits="", note="")
        for i, (bits, note) in enumerate(pats, 1):
            add("prep", PREP_SECONDS, take=take, step=i, bits=bits, note=note)
            add("rec", HOLD_SECONDS, take=take, step=i, bits=bits, note=note)
    return seg


class VideoSession:
    def __init__(self, app, frame, redo=None):
        self.redo = redo                                   # None = full session, list = film these again
        self.schedule = build_schedule(redo)
        self.total = self.schedule[-1]["end"]
        self.n_pat = len(redo) if redo is not None else len(patterns())
        self.dark = False                                  # the module sets this: dark lower area -> light texts
        self.marks = []                                    # patterns marked with X: [dict(bits, take, note)]
        self.toast = ("", 0.0)
        h, w = frame.shape[:2]
        self.size = (w, h)
        self.folder = DATA_DIR / "videos" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}{'_redo' if redo is not None else ''}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(self.folder / "session.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, self.size)
        if not self.writer.isOpened():
            raise RuntimeError("could not open the video file (codec mp4v missing?)")
        self.q = queue.Queue(maxsize=80)         # (duplicated frames share one array, so this is about 0.5 GB at most)
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()
        self.t0 = None
        self.written = 0
        self.dropped = 0
        self.done = False
        self.error = ""
        self._app_info = dict(cam_size=getattr(app.calib, "cam_size", None),
                              H=None if getattr(app.calib, "H", None) is None else np.asarray(app.calib.H).tolist())
        self._write_labels(self.total)
        print(f"[fingers] recording session -> {self.folder}  ({self.total / 60:.1f} min)")

    # --- video ---------------------------------------------------------------------
    def _write_loop(self):
        while True:
            f = self.q.get()
            if f is None:
                break
            try:
                self.writer.write(f)
            except Exception as e:
                self.error = str(e)

    def now(self, clock):
        return 0.0 if self.t0 is None else clock - self.t0

    def step(self, clock, frame):
        """Call once per frame. The video runs at a fixed FPS in real time: slow camera frames are repeated, so video
        seconds = real seconds and labels.csv always fits."""
        if frame is None or self.done:
            return
        if self.t0 is None:
            self.t0 = clock
        t = clock - self.t0
        want = int(min(t, self.total) * FPS) + 1
        for _ in range(min(want - self.written, 10)):
            try:
                self.q.put_nowait(frame)
                self.written += 1
            except queue.Full:
                self.dropped += 1
                self.written += 1
        if t >= self.total:
            self.finish(t)

    def mark(self, clock):
        """X: the pattern shown right now did not work (wrong shape, hand out of view ...): film it again later."""
        s, t = self.segment(clock)
        if s["kind"] not in ("prep", "rec") or self.done:
            return
        it = dict(bits=s["bits"], take=s["take"], note=s["note"])
        if it not in self.marks:
            self.marks.append(it)
            if self.redo is None:                          # full session: add to the list right away
                save_redo(load_redo() + [it])
        self.toast = (f"Marked for later: {s['bits']} ({name_of(s['bits'])})", clock + 3.0)

    def segment(self, clock):
        t = self.now(clock)
        for s in self.schedule:
            if s["start"] <= t < s["end"]:
                return s, t
        return self.schedule[-1], t

    def finish(self, t=None):
        if self.done:
            return
        self.done = True
        self.q.put(None)
        self.thread.join(timeout=60)
        self.writer.release()
        length = min(t if t is not None else self.total, self.total)
        self._write_labels(length)
        if self.redo is not None:                          # redo session: what was filmed is done, what failed again stays
            left = [it for i, it in enumerate(self.redo, 1)
                    if not any(sg["kind"] == "rec" and sg["step"] == i and sg["end"] <= length + 1e-6 for sg in self.schedule)]
            save_redo(left + self.marks)
        (self.folder / "session.json").write_text(json.dumps(dict(
            fps=FPS, size=self.size, seconds=round(length, 1), frames=self.written, dropped=self.dropped,
            intro=INTRO_SECONDS, prep=PREP_SECONDS, hold=HOLD_SECONDS, rest=REST_SECONDS, takes=TAKES,
            hand="left (right hand = mirror image)", started=datetime.now().isoformat(timespec="seconds"),
            calibration=self._app_info), indent=2), encoding="utf-8")
        print(f"[fingers] session saved: {self.folder}  ({length / 60:.1f} min, {self.dropped} frames dropped)")

    def _write_labels(self, upto):
        """labels.csv: one row per recorded pattern. Rows after `upto` (stopped early) are left out."""
        with open(self.folder / "labels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["take", "step", "label", "count", "fingers", "note", "prep_start_s", "rec_start_s", "rec_end_s", "marked"])
            prep = {}
            for s in self.schedule:
                if s["kind"] == "prep":
                    prep[(s["take"], s["step"])] = s["start"]
                elif s["kind"] == "rec" and s["end"] <= upto + 1e-6:
                    w.writerow([s["take"], s["step"], s["bits"], s["bits"].count("1"), name_of(s["bits"]), s["note"],
                                round(prep[(s["take"], s["step"])], 2), round(s["start"], 2), round(s["end"], 2),
                                "redo" if dict(bits=s["bits"], take=s["take"], note=s["note"]) in self.marks else ""])

    # --- screen --------------------------------------------------------------------
    def hint(self, clock):
        if self.done:
            return "Session saved", "info"
        s, t = self.segment(clock)
        left = int(s["end"] - t) + 1
        if s["kind"] == "intro":
            return f"Left hand over the table. Starting in {left}", "blink"
        if s["kind"] == "rest":
            nxt = self._next(s)
            note = f" ({nxt['note']})" if nxt["note"] else ""
            return f"Rest your arm. Next: {nxt['bits']} = {name_of(nxt['bits'])}{note}. Take {s['take']} in {left}", "blink"
        info = f"{s['bits']} = {name_of(s['bits'])}" + (f" ({s['note']})" if s["note"] else "")
        if s["kind"] == "prep":
            return f"Get ready: {info}. Recording in {left}", "blink"
        return f"REC {info}: turn and tilt your hand slowly", "info"

    def _next(self, s):
        """The first pattern after this segment."""
        i = self.schedule.index(s)
        return next(x for x in self.schedule[i + 1:] if x["kind"] == "prep")

    @property
    def fg(self):
        return config.TEXT_COLOR if self.dark else TEXT_COLOR

    def draw(self, surf, app, clock, w, h):
        area = pygame.Rect(0, round(opt("HUD_FRAC") * h), w, h - round(opt("HUD_FRAC") * h))
        font = app.board.font
        if self.done:
            t = font(h * 0.06).render("Session saved", True, self.fg)
            surf.blit(t, t.get_rect(center=area.center))
            return
        s, t = self.segment(clock)
        if s["kind"] == "intro":
            self._boxes(surf, app, "01100", area, h, y=area.top + area.h * 0.20, size=h * 0.10)
            lines = ["Label = 5 digits: thumb  index  middle  ring  pinky", "1 = stretched     0 = folded",
                     "01100 = index + middle",
                     "Your LEFT hand. Show each shape and turn / tilt it slowly."]
            for i, text in enumerate(lines):
                r = font(h * (0.052 if i < 3 else 0.04)).render(text, True, self.fg)
                surf.blit(r, r.get_rect(center=(w // 2, area.top + area.h * (0.42 + 0.085 * i))))
            self._countdown(surf, app, s, t, w, h, area)
            return
        if s["kind"] == "rest":
            nxt = self._next(s)
            r = font(h * 0.05).render(f"Take {s['take'] - 1} done. Rest your arm.   Next:", True, self.fg)
            surf.blit(r, r.get_rect(center=(w // 2, area.top + area.h * 0.14)))
            self._boxes(surf, app, nxt["bits"], area, h, y=area.top + area.h * 0.38, size=h * 0.12)
            name = name_of(nxt["bits"]) + (f"  ({nxt['note']})" if nxt["note"] else "")
            r = font(h * 0.052).render(f"{nxt['bits']}   {name}", True, self.fg)
            surf.blit(r, r.get_rect(center=(w // 2, area.top + area.h * 0.62)))
            self._countdown(surf, app, s, t, w, h, area)
            return
        self._draw_rec(surf, app, s, t, w, h, area)             # recording: the light area stays EMPTY, all info in the dark band

    def _draw_rec(self, surf, app, s, t, w, h, area):
        """While recording only the dark button band is used. The light area stays plain, like in real use
        (nothing projected onto the hand, nothing the hand can cover)."""
        font = app.board.font
        band = area.top                                          # height of the dark band
        light = config.TEXT_COLOR
        size = band * 0.36
        self._boxes(surf, app, s["bits"], area, h, y=band * 0.34, size=size, fill=light, back=config.BG_COLOR)
        bar = pygame.Rect(0, 0, round(w * 0.30), max(6, round(band * 0.035)))
        bar.midtop = (w // 2, round(band * 0.76))
        pygame.draw.rect(surf, light, bar, max(2, round(h * 0.004)), border_radius=bar.h // 2)
        if s["kind"] == "rec":
            fill = bar.copy()
            fill.w = max(round(bar.w * (t - s["start"]) / (s["end"] - s["start"])), bar.h)
            pygame.draw.rect(surf, light, fill, border_radius=bar.h // 2)
            pygame.draw.circle(surf, (230, 30, 30), (round(w * 0.16), round(band * 0.30)), round(band * 0.07))   # recording dot
        label = f"Redo {s['step']} / {self.n_pat} (take {s['take']})" if self.redo is not None else f"Take {s['take']} / {TAKES}   {s['step']} / {self.n_pat}"
        sub = font(band * 0.095).render(label, True, light)
        surf.blit(sub, sub.get_rect(midleft=(w * 0.14, band * 0.90)))
        if time.time() < self.toast[1]:
            msg, bold = self.toast[0], "Bold"
        else:
            msg, bold = "X = this one did not work", None
        foot = font(band * 0.095, bold).render(msg, True, light) if bold else font(band * 0.095).render(msg, True, light)
        surf.blit(foot, foot.get_rect(midright=(w * 0.86, band * 0.90)))

    def _boxes(self, surf, app, bits, area, h, y, size, fill=None, back=None):
        fill = fill or self.fg
        back = back or (config.BG_COLOR if self.dark else (255, 255, 255))
        w = surf.get_width()
        gap = size * 0.35
        x0 = w / 2 - (5 * size + 4 * gap) / 2
        for i, b in enumerate(bits):
            r = pygame.Rect(int(x0 + i * (size + gap)), int(y - size / 2), int(size), int(size))
            rad = int(size * 0.22)
            if b == "1":
                pygame.draw.rect(surf, fill, r, border_radius=rad)
                digit = app.board.font(size * 0.55, "Bold").render("1", True, back)
            else:
                pygame.draw.rect(surf, fill, r, max(3, int(size * 0.05)), border_radius=rad)
                digit = app.board.font(size * 0.55, "Bold").render("0", True, fill)
            surf.blit(digit, digit.get_rect(center=r.center))
            nm = app.board.font(size * 0.28).render(FINGERS[i], True, fill)
            surf.blit(nm, nm.get_rect(midtop=(r.centerx, r.bottom + size * 0.12)))

    def _bar(self, surf, s, t, w, h, area):
        bar = pygame.Rect(0, 0, round(w * 0.3), round(h * 0.018))
        bar.midtop = (w // 2, area.top + area.h * 0.84)
        pygame.draw.rect(surf, self.fg, bar, max(2, round(h * 0.004)), border_radius=bar.h // 2)
        if s["kind"] == "rec":
            fill = bar.copy()
            fill.w = max(round(bar.w * (t - s["start"]) / (s["end"] - s["start"])), bar.h)
            pygame.draw.rect(surf, self.fg, fill, border_radius=bar.h // 2)

    def _countdown(self, surf, app, s, t, w, h, area):
        left = int(s["end"] - t) + 1
        r = app.board.font(h * 0.11, "Bold").render(str(left), True, self.fg)
        surf.blit(r, r.get_rect(center=(w // 2, area.bottom - h * 0.10)))
