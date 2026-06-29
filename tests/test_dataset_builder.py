import unittest

from apps.dataset_builder.cli import is_youtube_source
from apps.dataset_builder.settings import BuilderSettings, StepMode


class SettingsTests(unittest.TestCase):
    def test_step_frames_requires_integer(self):
        settings = BuilderSettings(source="movie.mp4", step_mode=StepMode.FRAMES, step_value=1.5)
        with self.assertRaises(ValueError):
            settings.validate()

    def test_step_seconds(self):
        settings = BuilderSettings(source="movie.mp4", step_mode=StepMode.SECONDS, step_value=2)
        self.assertEqual(settings.step_seconds, 2.0)
        self.assertIsNone(settings.step_frames)


class CliTests(unittest.TestCase):
    def test_youtube_source_detection(self):
        self.assertTrue(is_youtube_source("https://www.youtube.com/watch?v=abc"))
        self.assertTrue(is_youtube_source("https://youtu.be/abc"))
        self.assertFalse(is_youtube_source("movie.mp4"))


if __name__ == "__main__":
    unittest.main()
