"""Shared shapes drawn with lines only (used by all modules)."""
import pygame

import config


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


def draw_vline(surf, x, color=None, width=None):
    """Vertical line over the full height of the content area (e.g. center line)."""
    w = width or config.LINE_WIDTH
    pygame.draw.line(surf, color or config.LINE_COLOR,
                     (x, 0), (x, surf.get_height()), w)


def draw_text(surf, font, text, pos, color=None, anchor="center"):
    """anchor: center / topleft / topright ..."""
    img = font.render(text, True, color or config.TEXT_COLOR)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect
