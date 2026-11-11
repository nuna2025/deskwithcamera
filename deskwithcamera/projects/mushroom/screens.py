"""Drawing of the Classification screens (pure drawing, no logic).

    step 1  "teach me":    Negative / Positive labels + counters, center line, light area for the cards
                           (also shown while the cards are taken away: the counters go down to 0)
    learning:              "Learning ..." with a moving bar
    step 2  "test":        a colored ring + label around every card that was classified
"""
import math

import pygame

import config
from core.buttons import row_center
from core.shapes import draw_lower_bg, draw_minus, draw_plus, draw_split_line

NEG_COLOR = config.CAM_LABEL_COLOR        # red   = the red of the title
POS_COLOR = config.FRAME_OUTER_COLOR      # blue  = the blue line around the frame
NAMES = {"-": "negative", "+": "positive"}
COLORS = {"-": NEG_COLOR, "+": POS_COLOR}


def draw_teach(surf, app, counts):
    """Step 1: light card area with the center line, Negative / Positive labels above the counters."""
    w, h = surf.get_size()
    draw_lower_bg(surf)
    draw_split_line(surf, w // 2)
    cy = row_center(h) + round(0.022 * h)                       # counters a bit below the middle of the button row
    size = round(h * 0.055)
    font = app.board.font(h * 0.07, "Black")
    g = round(0.08 * w - size / 2)                              # distance center line -> symbol (same on both sides)
    for side in ("-", "+"):
        n = counts[side]
        img = font.render(str(n), True, config.CAM_ON_COLOR if n else config.TEXT_COLOR)
        if side == "-":                                         # number next to the center line, symbol to its left
            rect = img.get_rect(midright=(w // 2 - g, cy))
            draw_minus(surf, (rect.left - round(size * 0.45) - size // 2, cy), size)
        else:                                                   # symbol next to the center line, number to its right
            sx = w // 2 + g + size // 2
            draw_plus(surf, (sx, cy), size)
            rect = img.get_rect(midleft=(sx + round(size * 0.95), cy))
        surf.blit(img, rect)
    lab = app.board.font(h * 0.041, "Black")
    d = round(0.072 * w)                                        # same distance from the center line on both sides
    ly = row_center(h) - round(0.056 * h)
    for side, x in (("-", w // 2 - d), ("+", w // 2 + d)):
        t = lab.render(NAMES[side].capitalize(), True, COLORS[side])
        surf.blit(t, t.get_rect(center=(x, ly)))


def draw_learning(surf, app, seconds, n_neg, n_pos):
    """While the model learns: text + a bar that moves back and forth (the real progress is not known)."""
    w, h = surf.get_size()
    draw_lower_bg(surf)
    bg = config.BG_COLOR
    t = app.board.font(h * 0.07, "ExtraBold").render("Learning ...", True, bg)
    surf.blit(t, t.get_rect(center=(w // 2, round(h * 0.50))))
    bar = pygame.Rect(0, 0, round(w * 0.45), round(h * 0.035))
    bar.center = (w // 2, round(h * 0.62))
    pygame.draw.rect(surf, bg, bar, 4, border_radius=bar.h // 2)
    inner = bar.inflate(-12, -12)
    seg = round(inner.w * 0.30)
    pos = (math.sin(seconds * 2.2) + 1) / 2                     # 0 .. 1 .. 0
    seg_rect = pygame.Rect(inner.left + round((inner.w - seg) * pos), inner.top, seg, inner.h)
    pygame.draw.rect(surf, bg, seg_rect, border_radius=inner.h // 2)
    info = app.board.font(h * 0.032).render(f"{n_neg} negative  +  {n_pos} positive cards", True, bg)
    surf.blit(info, info.get_rect(center=(w // 2, round(h * 0.70))))


def _expanded_quad(box, margin):
    box = [(float(x), float(y)) for x, y in box]              # numpy float32 -> python float (pygame needs it)
    cx = sum(p[0] for p in box) / 4
    cy = sum(p[1] for p in box) / 4
    out = []
    for x, y in box:
        d = math.hypot(x - cx, y - cy) or 1.0
        k = (d + margin) / d
        out.append((cx + (x - cx) * k, cy + (y - cy) * k))
    return out


def draw_result(surf, app, items, board_size):
    """Step 2: items = [(box, center, side, confidence)] in board pixels. Ring around the card + label below it."""
    w, h = surf.get_size()
    draw_lower_bg(surf)
    bw, bh = board_size
    sx, sy = w / bw, h / bh
    ring_w = max(6, round(h * 0.011))
    font = app.board.font(h * 0.036, "Black")
    for box, center, side, conf in items:
        col = COLORS[side]
        quad = _expanded_quad([(float(x) * sx, float(y) * sy) for x, y in box], round(h * 0.022))
        pygame.draw.polygon(surf, col, quad, ring_w)
        for p in quad:                                          # round the corners
            pygame.draw.circle(surf, col, (round(p[0]), round(p[1])), ring_w // 2)
        txt = font.render(f"{NAMES[side]}  {round(conf * 100)}%", True, (255, 255, 255))
        ph = round(h * 0.075)
        sym = round(ph * 0.45)
        pw = round(ph * 0.55) + sym + round(ph * 0.25) + txt.get_width() + round(ph * 0.45)
        pill = pygame.Rect(0, 0, pw, ph)
        pill.midtop = (round(float(center[0]) * sx), round(max(p[1] for p in quad)) + round(h * 0.012))
        pill.clamp_ip(pygame.Rect(0, round(h * 0.2), w, h - round(h * 0.2)))
        pygame.draw.rect(surf, col, pill, border_radius=ph // 2)
        sym_x = pill.left + round(ph * 0.55)
        (draw_plus if side == "+" else draw_minus)(surf, (sym_x, pill.centery), sym, (255, 255, 255))
        surf.blit(txt, txt.get_rect(midleft=(sym_x + sym // 2 + round(ph * 0.25), pill.centery)))
