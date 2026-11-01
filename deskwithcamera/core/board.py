"""Base board: 16:9 frame + "Cam" status at the top left + module title at the top right.

Every module draws only INSIDE this board. A module receives a Surface cut out
of the frame's inner area, so coordinate (0, 0) is the top-left corner inside
the frame.

      Cam: ON                                   Module title
      ┌──────────────────────────────────────────────────┐
      │                                                  │
      │          (the module draws here)                 │
      │                                                  │
      └──────────────────────────────────────────────────┘
"""
from dataclasses import dataclass

import pygame

import config

FONT_NAMES = "malgungothic,applegothic,notosanscjkkr,nanumgothic,arial"


@dataclass
class Layout:
    frame: pygame.Rect      # outer rectangle where the line is drawn
    inner: pygame.Rect      # inside the line = area used by modules
    header_y: int           # top y of the header text
    font_size: int


def compute_layout(width, height):
    """Place header + 16:9 frame centered on the screen."""
    font_size = max(14, round(height * config.HEADER_FONT_RATIO))
    header_h = round(font_size * 1.3) + config.HEADER_GAP
    margin = round(min(width, height) * config.MARGIN)

    avail_w = width - 2 * margin
    avail_h = height - 2 * margin - header_h
    ar = config.ASPECT[0] / config.ASPECT[1]
    if avail_w / avail_h > ar:
        fh = avail_h
        fw = fh * ar
    else:
        fw = avail_w
        fh = fw / ar

    top = (height - (header_h + fh)) / 2
    frame = pygame.Rect(round((width - fw) / 2), round(top + header_h),
                        round(fw), round(fh))
    lw = config.LINE_WIDTH
    inner = frame.inflate(-2 * lw, -2 * lw)
    return Layout(frame, inner, round(top), font_size)


class Board:
    def __init__(self):
        self._fonts = {}

    def font(self, size):
        size = max(8, int(size))
        if size not in self._fonts:
            self._fonts[size] = pygame.font.SysFont(FONT_NAMES, size)
        return self._fonts[size]

    def draw(self, screen, title, cam_on):
        """Draw background, header and frame; return the content Surface for the module."""
        lay = compute_layout(*screen.get_size())
        screen.fill(config.BG_COLOR)
        font = self.font(lay.font_size)

        # Top left: Cam: ON / OFF (live)
        label = font.render("Cam: ", True, config.TEXT_COLOR)
        state = font.render("ON" if cam_on else "OFF", True,
                            config.CAM_ON_COLOR if cam_on else config.CAM_OFF_COLOR)
        x = lay.frame.left
        screen.blit(label, (x, lay.header_y))
        screen.blit(state, (x + label.get_width(), lay.header_y))

        # Top right: module title
        t = font.render(title, True, config.TEXT_COLOR)
        screen.blit(t, (lay.frame.right - t.get_width(), lay.header_y))

        # Frame line
        pygame.draw.rect(screen, config.LINE_COLOR, lay.frame, config.LINE_WIDTH)

        return screen.subsurface(lay.inner)
