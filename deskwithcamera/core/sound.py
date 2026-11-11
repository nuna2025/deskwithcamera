"""Sound effects. Never raises: no audio device or a missing file just means 'no sound'."""
import os

import pygame

import config

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def play_startup():
    """Play the start sound once (so people can hear that the program is up)."""
    path = getattr(config, "STARTUP_SOUND", "assets/sounds/start.mp3")   # default works even with an older config.py
    if not path:
        print("[sound] start sound is switched off (STARTUP_SOUND is empty)")
        return
    if not os.path.isabs(path):
        path = os.path.join(_ROOT, path)
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        snd = pygame.mixer.Sound(path)
        snd.set_volume(float(getattr(config, "STARTUP_SOUND_VOLUME", 1.0)))
        snd.play()
        print(f"[sound] playing {os.path.basename(path)} ({snd.get_length():.1f} s, volume {snd.get_volume():.1f}, mixer {pygame.mixer.get_init()})")
    except Exception as e:                                   # sound is optional
        print(f"[sound] start sound not played: {e}")
