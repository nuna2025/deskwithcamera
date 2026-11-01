"""Example of a module that only writes text on the board (e.g. 'Finished', instructions)."""
from core.module import Module
from core.shapes import draw_text


class TextDemo(Module):
    title = "Text"

    def __init__(self, text="Finished"):
        self.text = text

    def draw(self, surf, app):
        w, h = surf.get_size()
        draw_text(surf, app.board.font(h * 0.2), self.text, (w // 2, h // 2))
