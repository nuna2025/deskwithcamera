"""Parent class of all modules. A new module subclasses this and only needs draw()."""


class Module:
    title = "Module"          # shown at the top right

    def on_enter(self, app):
        """Called once when the module starts."""

    def on_exit(self, app):
        """Called once when the module ends."""

    def handle_event(self, event, app):
        """pygame events (keyboard etc.)."""

    def update(self, dt, app):
        """Per-frame logic (dt = elapsed seconds). Camera frames: app.camera.latest()."""

    def draw(self, surf, app):
        """surf = area inside the frame. Coordinate (0, 0) is its top-left corner."""
