"""Parent class of all modules. A new module subclasses this and only needs draw()."""


class Module:
    title = "Module"          # screen name: top right of the header (e.g. "Card capture")
    area = ""                 # module theme name: red italic title at the top left (empty = the project title)
    menu_label = None         # set it and the module appears as a tile on the Home screen
    menu_caption = ""         # small text below that tile
    theme = None              # core.theme.LIGHT for a light background (None = default dark colors)
    show_buttons = True       # False = no hand-on-button row (e.g. calibration needs the whole area)
    show_home = True          # False = no automatic HOME button

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

    def hint(self, app):
        """Explanation in the middle of the header: (text, mode). mode: "info" (normal), "blink" (the user has to act: blinks)."""
        return "", "info"

    def buttons(self, app):
        """List of core.buttons.ButtonSpec for this module (HOME is added automatically)."""
        return []
