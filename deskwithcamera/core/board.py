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
from pathlib import Path

import pygame

import config

FALLBACK_FONTS = "malgungothic,applegothic,notosanscjkkr,nanumgothic,arial"   # used if a font file is missing
FONT_ROOT = Path(__file__).resolve().parent.parent


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
    inner = frame.inflate(-2 * lw, -2 * lw)             # (the thin outer line is drawn OUTSIDE the frame rect)
    return Layout(frame, inner, round(top), font_size)


class Board:
    def __init__(self):
        self._fonts = {}

    def font(self, size, weight=None):
        """Hahmlet font in the given pixel size. weight: "Medium", "Bold", "ExtraBold" ... (default config.TEXT_WEIGHT)."""
        size = max(8, int(size))
        weight = weight or config.TEXT_WEIGHT
        key = (size, weight)
        if key not in self._fonts:
            path = FONT_ROOT / config.FONT_DIR / f"{config.FONT_FAMILY}-{weight}.ttf"
            if path.exists():
                self._fonts[key] = pygame.font.Font(str(path), size)
            else:
                print(f"[board] font file not found: {path} (using a system font instead)")
                self._fonts[key] = pygame.font.SysFont(FALLBACK_FONTS, size)
        return self._fonts[key]

    def draw(self, screen, title, cam_on):
        """Draw background, header and frame; return the content Surface for the module."""
        lay = compute_layout(*screen.get_size())
        screen.fill(config.BG_COLOR)
        font = self.font(lay.font_size, config.HEADER_WEIGHT)

        # Top left: Cam: ON / OFF (live)
        if config.SHOW_CAM_STATUS:
            label = font.render("Cam: ", True, config.CAM_LABEL_COLOR)
            state = font.render("ON" if cam_on else "OFF", True,
                                config.CAM_ON_COLOR if cam_on else config.CAM_OFF_COLOR)
            x = lay.frame.left
            screen.blit(label, (x, lay.header_y))
            screen.blit(state, (x + label.get_width(), lay.header_y))

        # Top right: module title
        t = font.render(title, True, config.TEXT_COLOR)
        screen.blit(t, (lay.frame.right - t.get_width(), lay.header_y))

        # Frame: thin outer line (blue) around a thicker red line, both with slightly rounded corners
        r, ow = config.CORNER_RADIUS, config.OUTER_LINE_WIDTH
        if ow > 0:
            pygame.draw.rect(screen, config.FRAME_OUTER_COLOR, lay.frame.inflate(2 * ow, 2 * ow), ow,
                             border_radius=r + ow)
        pygame.draw.rect(screen, config.FRAME_COLOR, lay.frame, config.LINE_WIDTH, border_radius=r)

        return screen.subsurface(lay.inner)