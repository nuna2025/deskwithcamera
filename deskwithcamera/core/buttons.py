"""Hand-on-button ("dwell") buttons.

The user holds a hand on a projected button for a moment; the camera sees that the button area has
changed compared with the picture of the EMPTY buttons, and after BUTTON_HOLD seconds the button fires.

Rules that make it reliable
  * After every screen change the program waits (hands away!), then takes a picture of the empty
    buttons (the "baseline"). That is why the header says "Hands away: preparing buttons ...".
  * The detection zone is the INNER part of a button. The progress bar is drawn in the rim around it,
    so our own animation is never mistaken for a hand. The text on a button never changes: a changing
    value (e.g. the current color) is shown as a caption BELOW the button, outside of the zone.
  * A button fires once; then the hand must leave (and stay away BUTTON_RELEASE seconds) before any
    button works again.
  * A "witness" zone between the buttons detects changes of the whole picture (exposure, bumped
    camera): then the baseline is simply taken again and nobody is "pressed".
  * The mouse pointer counts as a hand too (BUTTON_MOUSE), so everything can be tested without camera.

Slots (top row of the content area):   (left) ................ [right2] [right]
  "left" is the ROUND button (HOME on every module screen, EXIT on the Home screen); "right" / "right2"
  are the small rectangular buttons, vertically centered on the round one.
A slot can also be a tuple (x, y, w, h) in fractions of the content area (used for the Home tiles).
"""
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pygame

import config
from core.opts import opt


@dataclass
class ButtonSpec:
    name: str
    label: str                              # never changes while the module is on screen
    action: Callable                        # action(app)
    slot: object = "right"                  # "left" | "right" | "right2" | (x, y, w, h)
    hold: Optional[float] = None            # seconds (default: BUTTON_HOLD or BUTTON_HOLD_DANGER)
    danger: bool = False                    # red + longer hold (EXIT)
    enabled: Optional[Callable] = None      # enabled(app) -> bool   (disabled = dim, no reaction)
    caption: Optional[Callable] = None      # caption(app) -> str    drawn BELOW the button

    def hold_seconds(self):
        if self.hold is not None:
            return self.hold
        return opt("BUTTON_HOLD_DANGER") if self.danger else opt("BUTTON_HOLD")


# --- geometry ---------------------------------------------------------------------------
ROW_Y = 0.035                           # top of the button row (fraction of the content height)


def row_center(h):
    """y of the middle of the button row = middle of the round button (also used for the counters)."""
    return round((ROW_Y + opt("CIRCLE_D") / 2) * h)


def slot_rect(slot, w, h):
    if isinstance(slot, tuple):
        x, y, sw, sh = slot
        return pygame.Rect(round(x * w), round(y * h), round(sw * w), round(sh * h))
    inset = round(0.06 * h)             # keeps the corner buttons clear of the rounded frame corners
    if slot == "left":                  # round button: a square rectangle, drawn as a circle
        d = round(opt("CIRCLE_D") * h)
        return pygame.Rect(inset, round(ROW_Y * h), d, d)
    bw, bh = round(opt("BUTTON_W") * w), round(opt("BUTTON_H") * h)
    gap = round(opt("BUTTON_GAP") * w)
    if slot == "right":
        x = w - inset - bw
    elif slot == "right2":
        x = w - inset - 2 * bw - gap
    else:
        raise ValueError(f"unknown button slot: {slot!r}")
    return pygame.Rect(x, row_center(h) - bh // 2, bw, bh)


def witness_rect(w, h):
    """Quiet strip in the middle of the button row: no button here, used to notice whole-picture changes."""
    d = round(opt("CIRCLE_D") * h)
    return pygame.Rect(round(0.45 * w), round(ROW_Y * h), round(0.10 * w), d)


def is_round(rect):
    return rect.w == rect.h


def fit_size(board, text, size, max_w, weight):
    """Largest font size (<= size) in which the text is at most max_w pixels wide."""
    size = int(size)
    while size > 8:
        if board.font(size, weight).size(text)[0] <= max_w:
            return size
        size -= 2
    return 8


def fit_font(board, text, size, max_w, weight=None):
    return board.font(fit_size(board, text, size, max_w, weight or opt("BUTTON_WEIGHT")), weight or opt("BUTTON_WEIGHT"))


class _State:
    def __init__(self):
        self.mode = "idle"          # idle | holding | wait (fired, waiting for the hand to leave)
        self.t = 0.0                # seconds held
        self.gap = 0.0              # seconds the hand has been missing while holding
        self.absent = 0.0           # seconds without a hand (for arming / release)
        self.armed = False          # False right after a screen change: the hand must leave first


class ButtonPanel:
    def __init__(self):
        self.specs = []
        self.states = {}
        self.rects = {}
        self.phase = "live"         # settle -> baseline -> live
        self._t = 0.0
        self._base = {}
        self._acc = {}
        self._acc_n = 0
        self._stuck = 0.0           # seconds a fired button has been waiting although it may be a stale baseline
        self._rebase = False
        self._dt = 0.0
        self._en_prev = {}          # enabled state of every button in the last frame
        self._resettle = {}         # button -> seconds left to wait after its look changed (dim <-> bright)
        self._frac = {}             # last changed fraction per zone (for the log line when a button fires)
        self._global_t = 0.0        # seconds the WHOLE picture has been different (shadow of a body, exposure change)
        self._prev_cells = {}       # coarse picture of every zone in the last frame (to see movement)

    # --- setup ---------------------------------------------------------------------------
    def set_buttons(self, specs):
        self.specs = list(specs)
        self.states = {s.name: _State() for s in self.specs}
        self.phase, self._t = "settle", 0.0
        self._base, self._acc, self._acc_n = {}, {}, 0
        self._stuck, self._rebase = 0.0, False
        self._en_prev, self._resettle = {}, {}

    # --- camera zones -----------------------------------------------------------------------
    def _zone(self, rect, gray, size):
        w, h = size
        gh, gw = gray.shape
        k = opt("BUTTON_ZONE_SHRINK")
        r = rect.inflate(-2 * k * rect.w, -2 * k * rect.h)
        x0, y0 = int(r.left * gw / w), int(r.top * gh / h)
        x1, y1 = max(int(r.right * gw / w), x0 + 1), max(int(r.bottom * gh / h), y0 + 1)
        return gray[y0:y1, x0:x1].astype(np.float32)

    def _all_rects(self, size):
        rects = dict(self.rects)
        rects["_witness"] = self._free_witness(size)
        return rects

    def _free_witness(self, size):
        """The quiet strip sits in the middle of the row; if a button is there (e.g. NEW), it moves to a free gap."""
        w, h = size
        base = witness_rect(w, h)
        taken = [r.inflate(round(0.02 * w), 0) for r in self.rects.values()]
        if not any(base.colliderect(t) for t in taken):
            return base
        for cx in (0.30, 0.26, 0.34, 0.22, 0.70, 0.74):
            r = base.copy()
            r.centerx = round(cx * w)
            if not any(r.colliderect(t) for t in taken):
                return r
        return base

    @staticmethod
    def _cells(z):
        """Zone reduced to about 8 x 4 block averages (camera noise averages out, real movement stays)."""
        h, w = z.shape
        ch, cw = max(h // 4, 1), max(w // 8, 1)
        return z[:h // ch * ch, :w // cw * cw].reshape(h // ch, ch, w // cw, cw).mean(axis=(1, 3))

    def _moving(self, name, z):
        """True while the picture in this zone changes from frame to frame (an arm passing by, not a hand held still)."""
        c = self._cells(z)
        prev = self._prev_cells.get(name)
        self._prev_cells[name] = c
        return prev is not None and prev.shape == c.shape and float(np.abs(c - prev).mean()) > opt("BUTTON_MOTION_THR")

    def _frac_changed(self, name, cur):
        base = self._base.get(name)
        if base is None or base.shape != cur.shape:
            return None
        d = cur - base                                    # a hand lit by the projector is BRIGHTER than the dark button;
        if not opt("BUTTON_ONLY_BRIGHTER"):               # a shadow (or the arm's shade) is only darker -> ignore it
            d = np.abs(d)
        return float((d > opt("BUTTON_DIFF_THR")).mean())

    # --- one step --------------------------------------------------------------------------------
    def update(self, dt, app, size, board_gray=None, mouse=None):
        """Call once per frame. size = content area (w, h). board_gray = gray top-down view or None.
        mouse = pointer position inside the content area or None. Runs the actions of fired buttons."""
        w, h = size
        self.rects = {s.name: slot_rect(s.slot, w, h) for s in self.specs}
        if not self.specs:
            return
        self._t += dt
        self._dt = dt

        if self.phase == "settle":
            if self._t >= opt("BUTTON_SETTLE"):
                self.phase, self._t = ("baseline" if board_gray is not None else "live"), 0.0
            return
        if self.phase == "baseline":
            if board_gray is None:
                self.phase = "live"
                return
            for name, r in self._all_rects(size).items():
                z = self._zone(r, board_gray, size)
                self._acc[name] = self._acc.get(name, 0) + z
            self._acc_n += 1
            if self._t >= opt("BUTTON_BASELINE_SECONDS") and self._acc_n >= 3:
                self._base = {n: a / self._acc_n for n, a in self._acc.items()}
                self.phase = "live"
            return

        presence = self._presence(board_gray, mouse, size, app)
        fired = []
        for s in self.specs:
            st, here = self.states[s.name], presence[s.name]
            if st.mode == "idle" and not st.armed:
                st.absent = 0.0 if here else st.absent + dt
                st.armed = st.absent >= opt("BUTTON_RELEASE")
            elif st.mode == "idle":
                if here:
                    st.mode, st.t, st.gap = "holding", 0.0, 0.0
            elif st.mode == "holding":
                if here:
                    st.gap = 0.0
                    st.t += dt
                    if st.t >= s.hold_seconds():
                        st.mode, st.absent = "wait", 0.0
                        fired.append(s)
                else:
                    st.gap += dt
                    if st.gap > opt("BUTTON_GAP_TOLERANCE"):
                        st.mode, st.t = "idle", 0.0
            elif st.mode == "wait":
                st.absent = 0.0 if here else st.absent + dt
                if st.absent >= opt("BUTTON_RELEASE"):
                    st.mode, st.t, st.armed = "idle", 0.0, True
        # a button that waits for the hand to leave for too long: the picture of the empty buttons is stale
        # (e.g. the camera exposure changed) -> take a new one instead of nagging the user forever
        if any(st.mode == "wait" for st in self.states.values()):
            self._stuck += dt
            if self._stuck >= opt("BUTTON_STUCK_SECONDS"):
                self._rebase, self._stuck = True, 0.0
                print("[buttons] a button still sees 'a hand' after "
                      f"{opt('BUTTON_STUCK_SECONDS'):.0f} s: taking a new picture of the empty buttons")
        else:
            self._stuck = 0.0
        # while one button waits for the hand to leave, no other button may start
        if any(st.mode == "wait" for st in self.states.values()):
            for st in self.states.values():
                if st.mode == "holding":
                    st.mode, st.t = "idle", 0.0
        for s in fired:
            print(f"[buttons] {s.name} fired (changed fraction {self._frac.get(s.name, 0):.2f}, "
                  f"threshold {opt('BUTTON_PRESENT_FRAC'):.2f})")
            s.action(app)

    def _presence(self, gray, mouse, size, app):
        pres = {s.name: False for s in self.specs}
        for s in self.specs:                            # a button changes its look when it becomes (dis)abled (dim <-> bright):
            en = s.enabled is None or bool(s.enabled(app))
            if self._en_prev.get(s.name, en) != en:
                self._resettle[s.name] = opt("BUTTON_RESETTLE")    # ... wait, then learn the new look as 'empty'
                self.states[s.name] = _State()
                self.states[s.name].armed = True
            self._en_prev[s.name] = en
        if gray is not None and self._base:
            rects = self._all_rects(size)
            cur = {n: self._zone(r, gray, size) for n, r in rects.items()}
            wb = self._base.get("_witness")
            if wb is not None and wb.shape == cur["_witness"].shape and cur["_witness"].mean() > 1:
                g = float(wb.mean() / cur["_witness"].mean())     # the quiet strip got brighter/darker: exposure
                if 0.5 < g < 2.0 and abs(g - 1) > 0.03:           # -> bring every zone back to the old brightness
                    cur = {n: np.clip(z * g, 0, 255) for n, z in cur.items()}
            wfrac = self._frac_changed("_witness", cur["_witness"])
            if self._rebase:
                self._rebase = False
                self._base = dict(cur)
                self._frac = {}
                for s in self.specs:
                    self.states[s.name] = _State()
                    self.states[s.name].armed = True
                return pres
            if wfrac is None:
                self._base = dict(cur)
            elif wfrac > 0.5:
                # The WHOLE picture changed: a body in the beam (shadow) or a new exposure. Nobody presses now, but the old
                # picture of the empty buttons is kept: when the shadow leaves again, everything is as before. Only when
                # the change stays, it becomes the new normal.
                self._global_t += self._dt
                if self._global_t >= opt("BUTTON_GLOBAL_SECONDS"):
                    self._base, self._global_t = dict(cur), 0.0
                for s in self.specs:
                    if self.states[s.name].mode == "holding":
                        self.states[s.name].mode, self.states[s.name].t = "idle", 0.0
                return pres
            else:
                self._global_t = 0.0
                for name, z in cur.items():
                    if self._resettle.get(name, 0) > 0:           # look of this button just changed: learn it, no hand
                        self._resettle[name] -= self._dt
                        self._base[name] = z
                        continue
                    frac = self._frac_changed(name, z)
                    self._frac[name] = frac or 0.0
                    if frac is None:
                        self._base[name] = z
                    elif name == "_witness" or frac <= opt("BUTTON_PRESENT_FRAC"):
                        a = opt("BUTTON_BASELINE_ALPHA")
                        self._base[name] = self._base[name] + a * (z - self._base[name])   # follow slow drift
                    moving = self._moving(name, z)
                    if name != "_witness" and frac is not None and frac > opt("BUTTON_PRESENT_FRAC") and not moving:
                        pres[name] = True              # (a hand that is still moving does not press: it is just passing)
        if mouse is not None and opt("BUTTON_MOUSE"):
            for s in self.specs:
                if self.rects[s.name].collidepoint(mouse):
                    pres[s.name] = True
        for s in self.specs:                            # disabled buttons never react
            if s.enabled is not None and not s.enabled(app):
                pres[s.name] = False
                self.states[s.name] = _State()
                self.states[s.name].armed = True
        return pres

    def busy(self):
        """True while a hand is on a button (holding) or the user has to take it away (fired)."""
        return any(st.mode in ("holding", "wait") for st in self.states.values())

    # --- header message ------------------------------------------------------------------------------
    def message(self):
        """(text, mode) for the header's second line while the panel needs the user's attention, else None."""
        if not self.specs:
            return None
        if self.phase in ("settle", "baseline"):
            return "Hands away: preparing buttons ...", "blink"
        for s in self.specs:
            if self.states[s.name].mode == "wait":
                return f"{s.label}! Now take your hand away", "blink"
        for s in self.specs:
            if self.states[s.name].mode == "holding":
                return f"Keep holding ... {s.label}", "info"
        return None

    # --- drawing --------------------------------------------------------------------------------------
    def _font_sizes(self, board, w, h):
        """One font size per group (round buttons / rectangular buttons): all buttons of a kind look alike."""
        weight = opt("BUTTON_WEIGHT")
        sizes = {}
        for s in self.specs:
            r = self.rects.get(s.name) or slot_rect(s.slot, w, h)
            group = is_round(r)
            size = fit_size(board, s.label, min(r.h * 0.42, h * 0.05), r.w * (0.72 if group else 0.8), weight)
            sizes[group] = min(size, sizes.get(group, size))
        return sizes

    def draw(self, surf, app):
        w, h = surf.get_size()
        live = self.phase == "live"
        board = app.board
        weight = opt("BUTTON_WEIGHT")
        sizes = self._font_sizes(board, w, h)
        for s in self.specs:
            r = self.rects.get(s.name) or slot_rect(s.slot, w, h)
            st = self.states[s.name]
            enabled = s.enabled is None or s.enabled(app)
            rnd = is_round(r)
            if s.danger:
                col = config.CAM_LABEL_COLOR                                   # the red of the title
            elif rnd:
                col = config.CAM_ON_COLOR                                      # HOME: bright green
            else:
                col = config.LINE_COLOR
            if not (live and enabled):
                col = _mix(col, config.BG_COLOR, 0.55)                       # dim
            # feedback color (progress, "fired"): green; on the green HOME button cream, so it stays visible
            fb = config.LINE_COLOR if (rnd and not s.danger) else config.CAM_ON_COLOR
            ring = col if st.mode != "wait" else fb
            if rnd:
                rad, width = r.w // 2, opt("CIRCLE_RING")
                pygame.draw.circle(surf, config.BG_COLOR, r.center, rad)     # face (also hides the center line)
                pygame.draw.circle(surf, ring, r.center, rad, width)
            else:
                width = 6 if st.mode == "wait" else 3
                pygame.draw.rect(surf, config.BG_COLOR, r, border_radius=14)
                pygame.draw.rect(surf, ring, r, width, border_radius=14)
            font = board.font(sizes[rnd], weight)
            img = font.render(s.label, True, col)
            surf.blit(img, img.get_rect(center=r.center))
            if st.mode == "holding":                                         # progress in the rim
                frac = min(1.0, st.t / s.hold_seconds())
                if rnd:
                    self._progress_ring(surf, r, frac, fb)
                else:
                    bar = pygame.Rect(r.left + 10, r.bottom - 9, r.w - 20, 5)
                    pygame.draw.rect(surf, _mix(config.LINE_COLOR, config.BG_COLOR, 0.7), bar, border_radius=3)
                    pygame.draw.rect(surf, fb, (bar.x, bar.y, round(bar.w * frac), bar.h),
                                     border_radius=3)
            if s.caption is not None:
                text = s.caption(app)
                if text:
                    small = board.font(max(10, h * 0.032), weight)
                    cap = small.render(text, True, config.CAM_ON_COLOR)
                    surf.blit(cap, cap.get_rect(midtop=(r.centerx, r.bottom + round(0.003 * h))))

    @staticmethod
    def _progress_ring(surf, r, frac, color):
        """Arc on the ring of a round button, growing clockwise from the top."""
        width = opt("CIRCLE_RING")
        radius = r.w / 2 - width / 2
        steps = max(2, int(120 * frac))
        for i in range(steps + 1):
            a = -math.pi / 2 + 2 * math.pi * frac * i / steps
            pos = (round(r.centerx + radius * math.cos(a)), round(r.centery + radius * math.sin(a)))
            pygame.draw.circle(surf, color, pos, max(2, width // 2))


def _mix(a, b, t):
    """Blend color a toward color b by t (0..1)."""
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))
