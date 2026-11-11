"""Shared shapes drawn with lines only (used by all modules)."""
import pygame

import config
from core.opts import opt


def draw_o(surf, center, size, color=None, width=None):
    pygame.draw.circle(surf, color or config.LINE_COLOR, center, size // 2,
                       width or config.LINE_WIDTH)


def draw_x(surf, center, size, color=None, width=None):
    cx, cy = center
    r = size // 2
    color = color or config.LINE_COLOR
    width = width or config.LINE_WIDTH
    pygame.draw.line(surf, color, (cx - r, cy - r), (cx + r, cy + r), width)
    pygame.draw.line(surf, color, (cx - r, cy + r), (cx + r, cy - r), width)


def draw_plus(surf, center, size, color=None, width=None):
    cx, cy = center
    r = size // 2
    color = color or config.LINE_COLOR
    width = width or config.LINE_WIDTH
    pygame.draw.line(surf, color, (cx - r, cy), (cx + r, cy), width)
    pygame.draw.line(surf, color, (cx, cy - r), (cx, cy + r), width)


def draw_minus(surf, center, size, color=None, width=None):
    cx, cy = center
    r = size // 2
    pygame.draw.line(surf, color or config.LINE_COLOR, (cx - r, cy), (cx + r, cy),
                     width or config.LINE_WIDTH)


def draw_vline(surf, x, color=None, width=None):
    """Vertical line over the full height of the content area (e.g. center line)."""
    w = width or config.LINE_WIDTH
    pygame.draw.line(surf, color or config.LINE_COLOR,
                     (x, 0), (x, surf.get_height()), w)


def lower_area_y(surf):
    """y of the boundary between the button area (top) and the light lower area (cards / photos)."""
    return round(opt("HUD_FRAC") * surf.get_height())


def draw_lower_bg(surf):
    """Fill the area below the button row with the light background (bottom corners follow the frame)."""
    w, h = surf.get_size()
    y = lower_area_y(surf)
    r = max(0, config.CORNER_RADIUS - config.LINE_WIDTH)
    pygame.draw.rect(surf, opt("LOWER_BG"), (0, y, w, h - y),
                     border_bottom_left_radius=r, border_bottom_right_radius=r)


def draw_split_line(surf, x, width=None):
    """Center line: LINE_COLOR in the button area, dark (BG_COLOR) on the light lower area.
    It runs through the bottom frame line, so it touches the frame edge."""
    w = width or config.LINE_WIDTH
    h = surf.get_height()
    y = lower_area_y(surf)
    pygame.draw.line(surf, config.LINE_COLOR, (x, 0), (x, y), w)
    end = h
    target = surf
    parent = surf.get_parent() if hasattr(surf, "get_parent") else None
    if parent is not None:
        ox, oy = surf.get_abs_offset()
        target, x, y, end = parent, x + ox, y + oy, h + oy + config.LINE_WIDTH
    pygame.draw.line(target, config.BG_COLOR, (x, y), (x, end), w)


def draw_text(surf, font, text, pos, color=None, anchor="center"):
    """anchor: center / topleft / topright ..."""
    img = font.render(text, True, color or config.TEXT_COLOR)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect
