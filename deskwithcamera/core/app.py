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
from core import theme as themes
from core.board import Board, compute_layout
from core.buttons import ButtonPanel, ButtonSpec
from core.calibration import Calibration
from core.camera import Camera
from core.sound import play_startup

try:
    from core import vision
except ImportError:                       # OpenCV missing: no hand detection, the mouse still works
    vision = None

CAM_OFF_TEXT = "Cam OFF: the camera is not working, so nothing can be recognized yet"

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
    def __init__(self, modules, profile=None, project_title="", start_index=0, home_index=None):
        self.modules = modules
        self.index = start_index
        self.home_index = home_index          # index of the Home module (None = no Home, no HOME button)
        self.project_title = project_title
        self.panel = ButtonPanel()            # hand-on-button buttons of the module on screen
        self.running = False
        self.profile_name = profile or config.DISPLAY_PROFILE
        self.profile = config.PROFILES[self.profile_name]
        self.board = Board()
        self.camera = Camera()
        self.calib = Calibration.load()      # shared by all modules: app.calib
        self.screen = None

    @property
    def module(self):
        return self.modules[self.index]

    # --- Actions for buttons ---------------------------------------------------
    def go_home(self):
        if self.home_index is not None:
            self.switch(self.home_index)

    def quit(self):
        self.running = False

    def _enter(self):
        """Start the current module and set up its buttons (HOME is added automatically)."""
        self.module.on_enter(self)
        self.refresh_buttons()

    def refresh_buttons(self):
        """(Re)build the buttons of the current module - a module calls this when its buttons change (e.g. a new step)."""
        mod = self.module
        specs = []
        if mod.show_buttons:
            if self.home_index is not None and self.index != self.home_index and mod.show_home:
                specs.append(ButtonSpec("home", "HOME", lambda app: app.go_home(), "left"))
            specs += list(mod.buttons(self))
        self.panel.set_buttons(specs)

    # --- Drawing (also used by tests) ----------------------------------------
    def render(self, surface):
        mod = self.module
        themes.apply(mod.theme or themes.DARK)
        msg = self.panel.message() if mod.show_buttons else None
        hint, mode = msg if msg else mod.hint(self)
        if config.CAMERA_ENABLED and not self.camera.is_on:          # replaces the old "Cam: OFF" at the top left
            hint, mode = CAM_OFF_TEXT, "blink"
        content = self.board.draw(surface, mod.area or self.project_title, mod.title, hint, mode)
        mod.draw(content, self)
        if mod.show_buttons:
            self.panel.draw(content, self)

    def _board_gray(self):
        """Small gray top-down view of the content area for the button detection (None if not possible)."""
        if vision is None or not self.calib.ok or not self.camera.is_on:
            return None
        frame = self.camera.latest()
        return None if frame is None else vision.board_gray(frame, self.calib.H)

    def update_buttons(self, dt):
        mod = self.module
        if not mod.show_buttons:
            return
        lay = compute_layout(*self.screen.get_size())
        mouse = None
        if pygame.mouse.get_focused():
            mx, my = pygame.mouse.get_pos()
            if lay.inner.collidepoint(mx, my):
                mouse = (mx - lay.inner.x, my - lay.inner.y)
        self.panel.update(dt, self, lay.inner.size, self._board_gray(), mouse)

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
        self._enter()

    # --- Main loop -----------------------------------------------------------
    def run(self):
        self.open_display()
        if config.CAMERA_ENABLED:
            self.camera.start()
        self._enter()
        play_startup()                       # audible "program started" signal
        clock = pygame.time.Clock()
        self.running = True
        while self.running:
            dt = clock.tick(30) / 1000
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        self.running = False
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
            self.update_buttons(dt)
            self.render(self.screen)
            pygame.display.flip()
        self.module.on_exit(self)
        self.camera.stop()
        pygame.quit()
