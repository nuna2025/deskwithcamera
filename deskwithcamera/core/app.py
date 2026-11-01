"""App: opening the display (projection mode), main loop, module switching, camera status."""
import os
import subprocess
import sys
import time

# Must be set before pygame starts: by default SDL minimizes a fullscreen window as soon as another
# window gets the focus (e.g. when you click into VS Code), and the desktop shows through.
os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")

import pygame

import config
from core.board import Board
from core.calibration import Calibration
from core.camera import Camera

WIN_MODES = {"extend": "/extend", "external": "/external",
             "clone": "/clone", "internal": "/internal"}


def apply_windows_mode(mode):
    """Same effect as Win+P (Windows only)."""
    if sys.platform != "win32" or mode not in WIN_MODES:
        return
    subprocess.run(["DisplaySwitch.exe", WIN_MODES[mode]], check=False)
    time.sleep(2.5)                       # wait a moment until the screens have switched


def keep_on_top():
    """Windows: keep the projection window above all other windows, even when it has no focus."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, wintypes.UINT]
        user32.SetWindowPos.restype = wintypes.BOOL
        hwnd = pygame.display.get_wm_info()["window"]
        HWND_TOPMOST, SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = -1, 0x0001, 0x0002, 0x0010
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)
    except Exception as e:                      # never crash the demo because of this
        print(f"[display] could not keep window on top: {e}")


class App:
    def __init__(self, modules, profile=None, project_title="", start_index=0):
        self.modules = modules
        self.index = start_index
        self.project_title = project_title
        self.profile_name = profile or config.DISPLAY_PROFILE
        self.profile = config.PROFILES[self.profile_name]
        self.board = Board()
        self.camera = Camera()
        self.calib = Calibration.load()      # shared by all modules: app.calib
        self.screen = None

    @property
    def module(self):
        return self.modules[self.index]

    # --- Drawing (also used by tests) ----------------------------------------
    def render(self, surface):
        content = self.board.draw(surface, self.module.title, self.camera.is_on)
        self.module.draw(content, self)

    # --- Opening the display -------------------------------------------------
    def open_display(self):
        if sys.platform == "win32":
            import ctypes                  # get the true resolution (ignore display scaling)
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        if config.AUTO_WINDOWS_MODE:
            apply_windows_mode(self.profile["win_mode"])
        pygame.init()
        monitor = self.profile["monitor"]
        if self.profile["fullscreen"]:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=monitor)
        else:
            self.screen = pygame.display.set_mode(config.WINDOW_SIZE, pygame.RESIZABLE,
                                                  display=monitor)
        if self.profile["fullscreen"] and getattr(config, "KEEP_ON_TOP", True):
            keep_on_top()
        pygame.mouse.set_visible(not self.profile["fullscreen"])
        pygame.display.set_caption(f"Tischspiele - {self.project_title} - {self.profile['label']}")

    # --- Module switching ----------------------------------------------------
    def switch(self, new_index):
        new_index %= len(self.modules)
        if new_index == self.index:
            return
        self.module.on_exit(self)
        self.index = new_index
        self.module.on_enter(self)

    # --- Main loop -----------------------------------------------------------
    def run(self):
        self.open_display()
        if config.CAMERA_ENABLED:
            self.camera.start()
        self.module.on_enter(self)
        clock = pygame.time.Clock()
        running = True
        while running:
            dt = clock.tick(30) / 1000
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key in (pygame.K_RIGHT, pygame.K_TAB):
                        self.switch(self.index + 1)
                    elif e.key == pygame.K_LEFT:
                        self.switch(self.index - 1)
                    elif e.key == pygame.K_c:
                        self.camera.toggle()
                    elif pygame.K_1 <= e.key <= pygame.K_9:
                        n = e.key - pygame.K_1
                        if n < len(self.modules):
                            self.switch(n)
                self.module.handle_event(e, self)
            self.module.update(dt, self)
            self.render(self.screen)
            pygame.display.flip()
        self.module.on_exit(self)
        self.camera.stop()
        pygame.quit()