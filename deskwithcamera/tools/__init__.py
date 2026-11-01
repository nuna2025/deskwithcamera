"""Shared tool modules that every project can use (test pattern, calibration, text)."""
from tools.calibration_module import CalibrationModule
from tools.test_pattern import TestPattern
from tools.text_demo import TextDemo


def tool_modules():
    return [TestPattern(), CalibrationModule(), TextDemo()]
