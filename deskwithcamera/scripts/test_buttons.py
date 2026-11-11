"""Tests for the hand-on-button logic without camera, projector or window (the mouse is simulated).

    python scripts/test_buttons.py
"""
import os
import sys
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pygame

from core.buttons import ButtonPanel, ButtonSpec, slot_rect, witness_rect
from core.opts import opt

SIZE = (1000, 560)
fired = []


def make_panel():
    p = ButtonPanel()
    p.set_buttons([ButtonSpec("home", "HOME", lambda a: fired.append("home"), "left"),
                   ButtonSpec("save", "SAVE", lambda a: fired.append("save"), "right"),
                   ButtonSpec("exit", "EXIT", lambda a: fired.append("exit"), "right2", danger=True)])
    return p


def run(p, seconds, mouse=None, gray=None, dt=1 / 30):
    for _ in range(round(seconds / dt)):
        p.update(dt, None, SIZE, gray, mouse)


def center(slot):
    return slot_rect(slot, *SIZE).center


def check(cond, text):
    print(("ok   " if cond else "FAIL ") + text)
    if not cond:
        sys.exit(1)


# 1. settle phase: nothing works, header asks for hands away
p = make_panel()
check(p.message() == ("Hands away: preparing buttons ...", "blink"), "settle: header says hands away")
run(p, opt("BUTTON_SETTLE") + 0.2, mouse=center("left"))
check(fired == [], "settle: a hand on a button does nothing")
check(p.phase == "live", "settle ends -> live (no camera)")

# 2. a hand that was already there must leave first (armed)
run(p, 4.0, mouse=center("left"))
check(fired == [], "hand that was there from the start does not fire")
run(p, opt("BUTTON_RELEASE") + 0.2, mouse=None)

# 3. hold 2 s -> fires once, then waits for release
run(p, 1.0, mouse=center("left"))
check(fired == [] and p.message()[0].startswith("Keep holding"), "1 s: not yet, message 'Keep holding'")
run(p, 1.3, mouse=center("left"))
check(fired == ["home"], "2 s: HOME fired")
check(p.message()[1] == "blink" and "take your hand away" in p.message()[0], "after firing: blinking 'take your hand away'")
run(p, 5.0, mouse=center("left"))
check(fired == ["home"], "hand stays: fires only once")
run(p, 0.3, mouse=None)
run(p, 2.0, mouse=center("right"))
check(fired == ["home"], "straight to SAVE: the 0.6 s release time comes first, so 2.0 s is not enough")
run(p, 0.6, mouse=center("right"))
check(fired == ["home", "save"], "... but holding on SAVE fires it after release time + 2 s")

# 4. short touch / flicker
run(p, 1.0, mouse=None)
n = len(fired)
run(p, 1.0, mouse=center("right"))
run(p, 0.2, mouse=None)
run(p, 0.8, mouse=center("right"))
check(len(fired) == n, "touch 1 s, 0.2 s gap, 0.8 s: below 2 s of holding -> no fire")
run(p, 0.5, mouse=None)
run(p, 1.0, mouse=center("right"))
run(p, 0.2, mouse=None)
run(p, 1.3, mouse=center("right"))
check(len(fired) == n + 1, "a 0.2 s flicker does not restart the hold (fires at 2 s of holding)")

# 5. danger button needs longer
run(p, 1.5, mouse=None)
fired.clear()
run(p, 2.5, mouse=center("right2"))
check(fired == [], "EXIT: not after 2.5 s")
run(p, 1.3, mouse=center("right2"))
check(fired == ["exit"], "EXIT: after 3.5 s")

# 6. camera detection with synthetic gray images
p = make_panel()
rng = np.random.default_rng(0)
base = (rng.normal(90, 6, (360, 640))).clip(0, 255).astype(np.uint8)


def hand_on(img, slot):
    out = img.copy()
    r = slot_rect(slot, *SIZE)
    x0, x1, y0, y1 = [int(v) for v in (r.left * 0.64, r.right * 0.64, r.top * 0.643, r.bottom * 0.643)]
    out[y0 - 6:y1 + 6, x0 - 6:x1 + 6] = 200                           # a bright hand covers the button
    return out


fired.clear()
run(p, opt("BUTTON_SETTLE") + 0.1, gray=base)
run(p, opt("BUTTON_BASELINE_SECONDS") + 0.3, gray=base)
check(p.phase == "live" and p._base, "camera: baseline taken")
run(p, opt("BUTTON_RELEASE") + 0.2, gray=base)
run(p, 2.3, gray=hand_on(base, "right"))
check(fired == ["save"], "camera: hand on SAVE for 2 s -> SAVE fires (and only SAVE)")
run(p, 3.0, gray=base)
fired.clear()
run(p, 3.0, gray=np.clip(base.astype(int) + 60, 0, 255).astype(np.uint8))      # whole picture got brighter
check(fired == [], "camera: whole-picture change (exposure) fires nothing")
run(p, 1.0, gray=np.clip(base.astype(int) + 60, 0, 255).astype(np.uint8))
run(p, 3.0, gray=np.clip(base.astype(int) + 60, 0, 255).astype(np.uint8))
check(fired == [], "camera: ... and the new brightness becomes the new baseline")

# 7. exposure drift: the quiet strip gets 12 % brighter -> zones are compensated, nothing fires
p = make_panel()
fired.clear()
run(p, opt("BUTTON_SETTLE") + 0.1, gray=base)
run(p, opt("BUTTON_BASELINE_SECONDS") + 0.3, gray=base)
run(p, opt("BUTTON_RELEASE") + 0.2, gray=base)
bright = np.clip(base.astype(float) * 1.12, 0, 255).astype(np.uint8)
run(p, 4.0, gray=bright)
check(fired == [], "camera: 12 % exposure drift fires nothing")

# 8. stale baseline: SAVE zone looks "changed" forever (no hand) -> fires once, then the panel recovers by itself
p = make_panel()
fired.clear()
run(p, opt("BUTTON_SETTLE") + 0.1, gray=base)
run(p, opt("BUTTON_BASELINE_SECONDS") + 0.3, gray=base)
run(p, opt("BUTTON_RELEASE") + 0.2, gray=base)
stale = hand_on(base, "right")
run(p, 2.3, gray=stale)
check(fired == ["save"], "stale baseline: fires once")
fired.clear()
run(p, opt("BUTTON_STUCK_SECONDS") + 1.0, gray=stale)
check(p.message() is None, "stale baseline: 'take your hand away' disappears by itself")
run(p, 5.0, gray=stale)
check(fired == [], "stale baseline: and does not fire again")

# 9. a button changes its look when it becomes enabled (SAVE dim -> bright): that is not a hand
state = {"on": False}
p = ButtonPanel()
p.set_buttons([ButtonSpec("save", "SAVE", lambda a: fired.append("save"), "right", enabled=lambda a: state["on"])])
fired.clear()
run(p, opt("BUTTON_SETTLE") + 0.1, gray=base)
run(p, opt("BUTTON_BASELINE_SECONDS") + 0.3, gray=base)
run(p, 1.0, gray=base)
state["on"] = True
bright_save = hand_on(base, "right")                 # the enabled button looks different from the dim one
run(p, 6.0, gray=bright_save)
check(fired == [], "enabled button looks different: no false press, no freeze")
check(not p.busy(), "... and the picture is not frozen")

# 10. a shadow (darker zone) is not a hand
p = make_panel()
fired.clear()
run(p, opt("BUTTON_SETTLE") + 0.1, gray=base)
run(p, opt("BUTTON_BASELINE_SECONDS") + 0.3, gray=base)
run(p, opt("BUTTON_RELEASE") + 0.2, gray=base)
shade = base.copy()
r = slot_rect("right", *SIZE)
shade[int(r.top * 0.643) - 6:int(r.bottom * 0.643) + 6, int(r.left * 0.64) - 6:int(r.right * 0.64) + 6] = 30   # dark
import config
config.BUTTON_ONLY_BRIGHTER = True
run(p, 4.0, gray=shade)
check(fired == [], "camera: with BUTTON_ONLY_BRIGHTER a shadow over a button fires nothing")
del config.BUTTON_ONLY_BRIGHTER
p = make_panel()
run(p, opt("BUTTON_SETTLE") + 0.1, gray=base)
run(p, opt("BUTTON_BASELINE_SECONDS") + 0.3, gray=base)
run(p, opt("BUTTON_RELEASE") + 0.2, gray=base)
run(p, 2.3, gray=shade)
check(fired == ["save"], "camera: by default a darker hand on SAVE works too")

# 11. an arm that keeps moving over SAVE does not press it; a hand held still does
p = make_panel()
fired.clear()
run(p, opt("BUTTON_SETTLE") + 0.1, gray=base)
run(p, opt("BUTTON_BASELINE_SECONDS") + 0.3, gray=base)
run(p, opt("BUTTON_RELEASE") + 0.2, gray=base)
a_img, b_img = hand_on(base, "right"), base.copy()
r = slot_rect("right", *SIZE)
x0, x1, y0, y1 = [int(v) for v in (r.left * 0.64, r.right * 0.64, r.top * 0.643, r.bottom * 0.643)]
b_img[y0 - 6:y1 + 6, x0 - 6:x1 + 6] = 120
for _ in range(120):                                   # 4 s: the arm's brightness changes every frame
    run(p, 1 / 30, gray=a_img if _ % 2 else b_img)
check(fired == [], "camera: a moving arm over SAVE does not press it")
run(p, 3.0, gray=base)
run(p, 2.3, gray=a_img)
check(fired == ["save"], "camera: ... but a hand held still does")

# 12. a body crosses the beam: the whole picture gets dark for 1.5 s, then everything is as before
p = make_panel()
fired.clear()
run(p, opt("BUTTON_SETTLE") + 0.1, gray=base)
run(p, opt("BUTTON_BASELINE_SECONDS") + 0.3, gray=base)
run(p, opt("BUTTON_RELEASE") + 0.2, gray=base)
dark = (base.astype(float) * 0.4).astype(np.uint8)
run(p, 1.5, gray=dark)
run(p, 0.3, gray=base)
check(fired == [] and not p.busy() and p.message() is None, "body in the beam for 1.5 s: nothing fires, no nagging afterwards")
run(p, 2.3, gray=hand_on(base, "right"))
check(fired == ["save"], "... and a real press still works right after")
print("all button tests passed")
