"""Optional settings with defaults.

Every value can be overridden in the root config.py (just add a line with the same name).
If config.py does not have the line, the default below is used, so an older config.py keeps working.
"""
import config

DEFAULTS = {
    # --- header -----------------------------------------------------------------
    "HINT_SCALE": 0.85,             # size of the centered hint relative to the normal header text (1.0 = same size)
    "TITLE_SCALE": 1.0,             # size of the red title on the left relative to the header text (1.33 = the first size)
    "RIGHT_SCALE": 0.9,             # size of the screen name on the right (1.0 = the earlier size); its end position stays the same
    "BLINK_PERIOD": 1.0,            # seconds for one blink cycle (attention messages)
    "BLINK_ON": 0.65,               # fraction of the cycle the text is visible

    # --- layout -------------------------------------------------------------------
    "SAFE_INSET": 0.03,             # header texts keep at least this distance from the screen edge (fraction of the
                                    # screen width). The frame leaves room for the letter that sticks out at the side,
                                    # so a projector that cuts off the edges does not cut letters. 0 = switched off
    "HUD_FRAC": 0.19,              # top part of the content area: buttons + counters. Cards go BELOW this line

    # --- button shapes (fractions of the content area) ---------------------------------
    "BUTTON_W": 0.11,               # width of a rectangular button (fraction of the content width)
    "BUTTON_H": 0.065,              # height of a rectangular button (fraction of the content height)
    "BUTTON_GAP": 0.02,             # gap between two buttons on the right (fraction of the content width)
    "CIRCLE_D": 0.14,               # diameter of the round HOME / EXIT button (fraction of the content height)
    "CIRCLE_RING": 6,               # line thickness of a round button (pixels)
    "BUTTON_WEIGHT": "Black",       # font weight of the button texts and their captions
    "TILE_W": 0.22,                 # Home tile: width (fraction of the content width)
    "TILE_H": 0.15,                 # Home tile: height (fraction of the content height)

    # --- hand-on-button (dwell) buttons -----------------------------------------------
    "BUTTON_HOLD": 2.0,             # seconds the hand must stay on a normal button
    "BUTTON_HOLD_DANGER": 3.5,      # ... on a dangerous one (EXIT)
    "BUTTON_SETTLE": 1.5,           # after a screen change: wait this long (hands away!) ...
    "BUTTON_BASELINE_SECONDS": 0.5, # ... then take the picture of the empty buttons for this long
    "BUTTON_RELEASE": 0.6,          # the hand must be away this long before a button works again
    "BUTTON_GAP_TOLERANCE": 0.3,    # the hand may flicker away this long without restarting the hold
    "BUTTON_DIFF_THR": 40,          # gray-value difference (0..255) that counts as "changed"
    "BUTTON_PRESENT_FRAC": 0.35,    # this fraction of the button zone must be changed = hand on the button
    "BUTTON_ZONE_SHRINK": 0.2,      # detection zone = button shrunk by this fraction on every side
    "BUTTON_BASELINE_ALPHA": 0.05,  # how fast the empty-button picture follows slow lighting changes
    "BUTTON_MOUSE": True,           # the mouse pointer also counts as a hand (testing on the laptop)
}


def opt(name):
    return getattr(config, name, DEFAULTS[name])
