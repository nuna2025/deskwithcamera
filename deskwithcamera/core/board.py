"""Base board: header + 16:9 frame.

   DeskWithCamera          Hint (always bright green,            Home
   (red, italic)           blinks when urgent, 1-2 lines)        (green)
      ┌──────────────────────────────────────────────────────┐
      │                                                      │
      │          (the module draws here)                     │
      │                                                      │
      └──────────────────────────────────────────────────────┘

  * left:   the program name on the Home screen, otherwise the module's theme name (e.g. "Classification").
            Red, italic, bold. Its first letter sticks out to the left of the frame.
  * right:  the screen name (e.g. "Home", "Card capture"). Bright green. Its last letter sticks out to the right.
  * middle: the explanation, exactly centered above the frame. Always bright green. If it is too long it
            gets a second line, and only if that is not enough the letters get smaller.
  All three texts stand on ONE common baseline, the same distance above the frame.

Every module draws only INSIDE this board. A module receives a Surface cut out
of the frame's inner area, so coordinate (0, 0) is the top-left corner inside
the frame.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import pygame

import config
from core.opts import opt

FALLBACK_FONTS = "malgungothic,applegothic,notosanscjkkr,nanumgothic,arial"   # used if a font file is missing
FONT_ROOT = Path(__file__).resolve().parent.parent
LINE_FACTOR = 1.25                                                               # line height / font size
LINE_SPACING = 0.95                                                              # two-line hint: line height / font height


@dataclass
class Layout:
    frame: pygame.Rect      # outer rectangle where the line is drawn
    inner: pygame.Rect      # inside the line = area used by modules
    header_y: int           # top y of the header
    header_h: int           # height of the header (without the gap to the frame)
    font_size: int


def compute_layout(width, height):
    """Place header + 16:9 frame centered on the screen."""
    font_size = max(14, round(height * config.HEADER_FONT_RATIO))
    header_h = round(font_size * LINE_FACTOR) * 2                   # room for a two-line hint
    block_h = header_h + config.HEADER_GAP
    margin = round(min(width, height) * config.MARGIN)
    # Free space left/right: the first letter of the title (and the last one of the screen name) stick out of the
    # frame, and a projector may cut a little off the edges -> keep room for both. (Only matters on screens that are
    # wider than 16:9 relative to their height, e.g. 16:10; on 16:9 the frame is limited by the height anyway.)
    safe = round(width * opt("SAFE_INSET"))
    margin_x = max(margin, safe + round(font_size * 1.3))          # room for about one letter that sticks out

    avail_w = width - 2 * margin_x
    avail_h = height - 2 * margin - block_h
    ar = config.ASPECT[0] / config.ASPECT[1]
    if avail_w / avail_h > ar:
        fh = avail_h
        fw = fh * ar
    else:
        fw = avail_w
        fh = fw / ar

    top = (height - (block_h + fh)) / 2
    frame = pygame.Rect(round((width - fw) / 2), round(top + block_h),
                        round(fw), round(fh))
    lw = config.LINE_WIDTH
    inner = frame.inflate(-2 * lw, -2 * lw)             # (the thin outer line is drawn OUTSIDE the frame rect)
    return Layout(frame, inner, round(top), header_h, font_size)


def blink_visible(now=None):
    """True while a blinking text should be visible (cycle BLINK_PERIOD seconds)."""
    now = time.time() if now is None else now
    period = opt("BLINK_PERIOD")
    return (now % period) < period * opt("BLINK_ON")


def wrap2(font, text, max_w):
    """One line if it fits, otherwise two balanced lines (the widest of the two as narrow as possible)."""
    if font.size(text)[0] <= max_w:
        return [text]
    words = text.split()
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        width = max(font.size(a)[0], font.size(b)[0])
        if best is None or width < best[0]:
            best = (width, [a, b])
    return best[1] if best else [text]


class Board:
    def __init__(self):
        self._fonts = {}

    def font(self, size, weight=None, italic=False):
        """Hahmlet font in the given pixel size. weight: "Medium", "Bold", "ExtraBold", "Black" ... (default config.TEXT_WEIGHT)."""
        size = max(8, int(size))
        weight = weight or config.TEXT_WEIGHT
        key = (size, weight, italic)
        if key not in self._fonts:
            path = FONT_ROOT / config.FONT_DIR / f"{config.FONT_FAMILY}-{weight}.ttf"
            if path.exists():
                f = pygame.font.Font(str(path), size)
            else:
                print(f"[board] font file not found: {path} (using a system font instead)")
                f = pygame.font.SysFont(FALLBACK_FONTS, size)
            if italic:
                f.set_italic(True)
            self._fonts[key] = f
        return self._fonts[key]

    def draw(self, screen, left="", right="", hint="", mode="info", now=None):
        """Draw background, header and frame; return the content Surface for the module.

        left = red italic title (top left), right = screen name (top right),
        hint = explanation in the middle, mode = "info" | "blink" (blinking = the user has to act).
        """
        lay = compute_layout(*screen.get_size())
        f = lay.font_size
        screen.fill(config.BG_COLOR)
        fr, ow = lay.frame, config.OUTER_LINE_WIDTH
        base_y = fr.top - ow - round(f * 0.45)                  # ONE baseline for all three texts

        def put(font, text, color, x, anchor):
            img = font.render(text, True, color)
            r = img.get_rect()
            setattr(r, anchor, (x, base_y - font.get_ascent()))
            screen.blit(img, r)
            return r

        # Safe inset: the header texts never come closer to the screen edge than this (a projector often
        # cuts a little off the edges). The letter that sticks out of the frame then sticks out a bit less.
        safe = round(screen.get_width() * opt("SAFE_INSET"))

        # left: red italic title; one letter sticks out to the left of the frame
        r_left = pygame.Rect(fr.left, 0, 0, 0)
        if left:
            title = self.font(f * opt("TITLE_SCALE"), config.HEADER_WEIGHT, italic=True)
            x = max(fr.left - title.size(left[0])[0], safe)
            r_left = put(title, left, config.CAM_LABEL_COLOR, x, "topleft")

        # right: screen name; one letter sticks out to the right of the frame
        r_right = pygame.Rect(fr.right, 0, 0, 0)
        if right:
            big = self.font(f * opt("RIGHT_SCALE"), config.HEADER_WEIGHT)
            ref = self.font(f, config.HEADER_WEIGHT)             # the end position stays where it was (set by the normal size)
            x = min(fr.right + ref.size(right[-1])[0], screen.get_width() - safe)
            r_right = put(big, right, config.CAM_ON_COLOR, x, "topright")

        # middle: hint exactly centered; 2 lines if long, then smaller letters if still too long
        if hint and (mode != "blink" or blink_visible(now)):
            size = max(10, round(f * opt("HINT_SCALE")))
            mid = self.font(size, config.TEXT_WEIGHT)
            half = min(fr.centerx - r_left.right, r_right.left - fr.centerx) - round(f * 0.5)
            half = max(half, round(f * 3))
            lines = wrap2(mid, hint, half * 2)
            while max(mid.size(t)[0] for t in lines) > half * 2 and size > 12:
                size -= 2
                mid = self.font(size, config.TEXT_WEIGHT)
                lines = wrap2(mid, hint, half * 2)
            lh = round(mid.get_height() * LINE_SPACING)
            for i, line in enumerate(reversed(lines)):          # the last line stands on the baseline
                img = mid.render(line, True, config.CAM_ON_COLOR)
                screen.blit(img, img.get_rect(midtop=(fr.centerx, base_y - mid.get_ascent() - i * lh)))

        # Frame: thin outer line (blue) around a thicker line, both with slightly rounded corners
        r = config.CORNER_RADIUS
        if ow > 0:
            pygame.draw.rect(screen, config.FRAME_OUTER_COLOR, fr.inflate(2 * ow, 2 * ow), ow,
                             border_radius=r + ow)
        pygame.draw.rect(screen, config.FRAME_COLOR, fr, config.LINE_WIDTH, border_radius=r)

        return screen.subsurface(lay.inner)
