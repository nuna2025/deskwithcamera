"""Color themes. A module can set  theme = LIGHT  (default: DARK = the colors of config.py).

apply(theme) copies the colors into the config module, so all drawing code (shapes, board, buttons)
that reads config.TEXT_COLOR etc. at draw time uses the colors of the module that is on screen.
"""
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Theme:
    name: str
    bg: tuple
    frame: tuple
    frame_outer: tuple
    line: tuple
    text: tuple
    cam_label: tuple
    cam_on: tuple
    cam_off: tuple


# DARK is read from config.py once, at import time (before apply() ever changes anything)
DARK = Theme("dark", config.BG_COLOR, config.FRAME_COLOR, config.FRAME_OUTER_COLOR, config.LINE_COLOR,
             config.TEXT_COLOR, config.CAM_LABEL_COLOR, config.CAM_ON_COLOR, config.CAM_OFF_COLOR)

# light gray (about 80 %), not pure white: less glare and weaker hand shadows
LIGHT = Theme("light", (204, 204, 204), (0x07, 0x19, 0x34), (0x00, 0x00, 0xFF), (0x07, 0x19, 0x34),
              (0x07, 0x19, 0x34), (200, 0, 0), (27, 127, 42), (198, 40, 40))


def apply(theme):
    config.BG_COLOR = theme.bg
    config.FRAME_COLOR = theme.frame
    config.FRAME_OUTER_COLOR = theme.frame_outer
    config.LINE_COLOR = theme.line
    config.TEXT_COLOR = theme.text
    config.CAM_LABEL_COLOR = theme.cam_label
    config.CAM_ON_COLOR = theme.cam_on
    config.CAM_OFF_COLOR = theme.cam_off
