"""A center line splits the area into two cells, with O and X at the top (base screen of the sorting game)."""
import config
from core.module import Module
from core.shapes import draw_o, draw_vline, draw_x


class OXSplit(Module):
    title = "O / X"

    def zone_of(self, x, width):
        """Is the x coordinate (content area) in the left cell (O) or the right cell (X)?"""
        return "O" if x < width / 2 else "X"

    def draw(self, surf, app):
        w, h = surf.get_size()
        draw_vline(surf, w // 2)
        size = round(h * 0.12)
        y = round(h * 0.15)
        draw_o(surf, (w // 4, y), size)
        draw_x(surf, (3 * w // 4, y), size)
