"""Start:  python main.py [--project NAME] [--profile window|projector|laptop2|second_only]
          python main.py --calibrate     (start on the calibration screen and calibrate again)
          python main.py --home          (Home screen with all modules; hold a hand on a tile)
          python main.py --list          (show all projects)

Keys: Left/Right/Tab = switch module, 1-9 = select module,
      C = camera on/off, ESC = quit
"""
import argparse
import sys

import config
from core.app import App
from core.home import HomeModule
from projects import discover
from tools import tool_modules


def choose_start(app, modules, tools, force_calibration):
    """Index of the module to start on.

    - saved calibration exists  -> the project's first module
    - no calibration (or --calibrate) -> the calibration screen
    """
    if not tools:
        return 0
    cal_index = next(i for i, m in enumerate(modules) if m.title == "Calibration")
    if force_calibration:
        app.calib.clear()            # forget the saved one in memory; the file is replaced after success
        return cal_index
    return len(tools) if app.calib.ok else cal_index


def main():
    projects = discover()
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=config.DEFAULT_PROJECT, help="folder name inside projects/")
    p.add_argument("--profile", choices=list(config.PROFILES), default=config.DISPLAY_PROFILE)
    p.add_argument("--camera", type=int, help="camera index (overrides config.CAMERA_INDEX)")
    p.add_argument("--calibrate", action="store_true",
                   help="ignore the saved calibration and start on the calibration screen")
    p.add_argument("--home", action="store_true",
                   help="start with the Home screen: all modules of all projects, chosen with a hand on a tile")
    p.add_argument("--list", action="store_true", help="list available projects and exit")
    args = p.parse_args()

    if args.list:
        for name, info in projects.items():
            print(f"{name:20s} {info.title}  -  {info.description}")
        return
    if args.project not in projects:
        sys.exit(f"Unknown project '{args.project}'. Available: {', '.join(projects)}")
    if args.camera is not None:
        config.CAMERA_INDEX = args.camera

    if args.home:
        tools = tool_modules()
        home = HomeModule()
        project_modules = [m for pr in projects.values() for m in pr.modules()]
        modules = tools + [home] + project_modules      # tools (keys 1-3), Home, then every module
        home.set_entries([(m.menu_label, m.menu_caption, i) for i, m in enumerate(modules) if m.menu_label])
        app = App(modules, profile=args.profile, project_title="Table games", home_index=len(tools))
    else:
        info = projects[args.project]
        tools = tool_modules() if info.include_tools else []
        modules = tools + info.modules()          # tools first (keys 1-3), project modules after
        app = App(modules, profile=args.profile, project_title=info.title)
    app.index = choose_start(app, modules, tools, args.calibrate)
    if app.calib.ok:
        print(f"[calibration] using saved calibration (rms {app.calib.rms:.1f} px). "
              f"Use --calibrate to calibrate again.")
    app.run()


if __name__ == "__main__":
    main()
