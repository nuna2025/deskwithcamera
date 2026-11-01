"""Photo to cartoon: main module (the first screen of this project)."""
from pathlib import Path

from core.module import Module
from core.shapes import draw_text

HERE = Path(__file__).parent
ASSETS = HERE / "assets"      # images, cards, fonts ...
DATA = HERE / "data"          # saved models, datasets ...


class MainModule(Module):
    title = "Photo to cartoon"       # shown at the top right of the board

    def on_enter(self, app):
        pass

    def update(self, dt, app):
        frame = app.camera.latest()          # BGR image from the camera, or None
        # if frame is not None and app.calib.ok:
        #     u, v = app.calib.cam_to_norm(x_cam, y_cam)   # camera pixel -> 0..1 inside the frame

    def draw(self, surf, app):
        # surf = the area inside the frame, (0, 0) is its top-left corner.
        # Whatever you draw here is projected onto the table.
        w, h = surf.get_size()
        draw_text(surf, app.board.font(h * 0.08), "Photo to cartoon", (w // 2, h // 2))
