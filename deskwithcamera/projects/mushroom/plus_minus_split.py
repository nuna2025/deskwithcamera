"""A center line splits the area into two cells, with - and + at the top.

    -  (left)  = safe mushrooms
    +  (right) = poisonous mushrooms

This is the base screen of the mushroom sorting game.
"""
from core.module import Module
from core.shapes import draw_minus, draw_plus, draw_vline

LEFT_ZONE = "-"           # swap these two (and the labels below) to put + on the left
RIGHT_ZONE = "+"
LABELS = {"+": "poisonous", "-": "safe"}


class PlusMinusSplit(Module):
    title = "+ / -"
    area = "Classification"

    def zone_of(self, x, width):
        """Is the x coordinate (content area) in the left or the right cell? Returns "-" or "+"."""
        return LEFT_ZONE if x < width / 2 else RIGHT_ZONE

    def label_of(self, x, width):
        """Same as zone_of, but as the class name: "safe" or "poisonous"."""
        return LABELS[self.zone_of(x, width)]

    def draw(self, surf, app):
        w, h = surf.get_size()
        draw_vline(surf, w // 2)
        size = round(h * 0.10)
        y = round(h * 0.34)                  # below the button row
        for symbol, cx in ((LEFT_ZONE, w // 4), (RIGHT_ZONE, 3 * w // 4)):
            (draw_plus if symbol == "+" else draw_minus)(surf, (cx, y), size)
