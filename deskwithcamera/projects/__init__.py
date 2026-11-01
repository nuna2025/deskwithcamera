"""Project discovery.

Every sub-project is one folder inside projects/ that contains a project.py:

    projects/
      mushroom/
        project.py        <- TITLE, DESCRIPTION, modules()
        ...own modules, assets/, data/ ...

Adding a folder with a project.py is enough; nothing else has to be registered.
Keep the top level of project.py light (import heavy libraries inside modules()),
because discovery imports project.py just to read TITLE and DESCRIPTION.
"""
import importlib
from dataclasses import dataclass
from pathlib import Path

PROJECTS_DIR = Path(__file__).parent


@dataclass
class ProjectInfo:
    name: str            # folder name, used with --project
    title: str
    description: str
    module: object       # the imported project.py

    def modules(self):
        return self.module.modules()

    @property
    def include_tools(self):
        return getattr(self.module, "INCLUDE_TOOLS", True)


def discover():
    found = {}
    for entry in sorted(PROJECTS_DIR.iterdir()):
        if entry.is_dir() and (entry / "project.py").exists():
            mod = importlib.import_module(f"projects.{entry.name}.project")
            found[entry.name] = ProjectInfo(
                entry.name,
                getattr(mod, "TITLE", entry.name),
                getattr(mod, "DESCRIPTION", ""),
                mod,
            )
    return found
