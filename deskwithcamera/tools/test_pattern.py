"""Test pattern: grid, diagonals, circle, corner markers (to check projector/camera alignment)."""
import pygame

import config
from core.module import Module
from core.shapes import draw_text


class TestPattern(Module):
    title = "Test Pattern"

    def draw(self, surf, app):
        w, h = surf.get_size()
        grid = (70, 70, 70)
        for i in range(1, 16):                       # 16 x 9 grid
            x = round(w * i / 16)
            pygame.draw.line(surf, grid, (x, 0), (x, h), 1)
        for j in range(1, 9):
            y = round(h * j / 9)
            pygame.draw.line(surf, grid, (0, y), (w, y), 1)

        pygame.draw.line(surf, (120, 120, 120), (0, 0), (w, h), 2)
        pygame.draw.line(surf, (120, 120, 120), (0, h), (w, 0), 2)
        pygame.draw.circle(surf, config.LINE_COLOR, (w // 2, h // 2), h // 4, 3)

        s = round(h * 0.08)                          # colored corner squares
        for rect, col in [((0, 0, s, s), (255, 70, 70)),
                          ((w - s, 0, s, s), (70, 255, 90)),
                          ((0, h - s, s, s), (80, 120, 255)),
                          ((w - s, h - s, s, s), (255, 230, 60))]:
            pygame.draw.rect(surf, col, rect)

        font = app.board.font(h * 0.07)
        draw_text(surf, font, f"{w} x {h}", (w // 2, h // 2))
