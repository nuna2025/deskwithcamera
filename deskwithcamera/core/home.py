"""Home screen: one flat tile per module. Hold a hand on a tile to start that module; the round EXIT button (top left)
closes the program.

Used with:  python main.py --home
"""
from core.buttons import ButtonSpec
from core.module import Module
from core.opts import opt
from core.shapes import draw_text


class HomeModule(Module):
    title = "Home"
    area = "DeskWithCamera"          # red title at the top left
    show_home = False                  # this IS home

    def __init__(self):
        self.entries = []              # (label, caption, module index in app.modules)

    def set_entries(self, entries):
        self.entries = list(entries)

    def buttons(self, app):
        n = len(self.entries)
        tw, th, gap, y = opt("TILE_W"), opt("TILE_H"), 0.05, 0.43
        x0 = (1 - (n * tw + (n - 1) * gap)) / 2
        specs = []
        for i, (label, caption, idx) in enumerate(self.entries):
            specs.append(ButtonSpec(f"tile{i}", label, lambda app, idx=idx: app.switch(idx),
                                    (x0 + i * (tw + gap), y, tw, th), caption=lambda app, c=caption: c))
        specs.append(ButtonSpec("exit", "EXIT", lambda app: app.quit(), "left",
                                hold=opt("BUTTON_HOLD_DANGER"), danger=True))
        return specs

    def hint(self, app):
        return "Hold your hand on a tile to start", "blink"

    def draw(self, surf, app):
        w, h = surf.get_size()
        draw_text(surf, app.board.font(h * 0.07, "ExtraBold"), "Choose a module", (w // 2, round(h * 0.30)))
