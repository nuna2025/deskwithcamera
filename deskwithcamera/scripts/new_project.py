"""Create a new sub-project folder from a template.

Usage:  python scripts/new_project.py <folder_name> "<Title shown on the board>"
Example: python scripts/new_project.py cartoon_style "Photo to cartoon"
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "projects"

PROJECT_PY = '''"""__TITLE__"""

TITLE = "__TITLE__"
DESCRIPTION = "TODO: one-line description of this project."
INCLUDE_TOOLS = True      # also offer the shared tools (test pattern, calibration, text)


def modules():
    # import inside the function so that listing projects stays fast
    from projects.__NAME__.main_module import MainModule
    return [MainModule()]
'''

MAIN_MODULE_PY = '''"""__TITLE__: main module (the first screen of this project)."""
from pathlib import Path

from core.module import Module
from core.shapes import draw_text

HERE = Path(__file__).parent
ASSETS = HERE / "assets"      # images, cards, fonts ...
DATA = HERE / "data"          # saved models, datasets ...


class MainModule(Module):
    title = "__TITLE__"       # shown at the top right of the board

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
        draw_text(surf, app.board.font(h * 0.08), "__TITLE__", (w // 2, h // 2))
'''


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    name, title = sys.argv[1], sys.argv[2]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        sys.exit("Folder name: lowercase letters, digits and underscores, starting with a letter.")
    folder = ROOT / name
    if folder.exists():
        sys.exit(f"projects/{name} already exists.")
    (folder / "assets").mkdir(parents=True)
    (folder / "data").mkdir()
    (folder / "assets" / ".gitkeep").touch()
    (folder / "data" / ".gitkeep").touch()
    (folder / "__init__.py").write_text(f'"""{title}"""\n', encoding="utf-8")
    for fname, tpl in (("project.py", PROJECT_PY), ("main_module.py", MAIN_MODULE_PY)):
        (folder / fname).write_text(tpl.replace("__NAME__", name).replace("__TITLE__", title),
                                    encoding="utf-8")
    print(f"Created projects/{name}/  ->  run it with: python main.py --project {name}")


if __name__ == "__main__":
    main()