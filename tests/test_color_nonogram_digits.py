import unittest
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from color_nonogram_digits import (
    DigitTemplate,
    TemplateDigitClassifier,
    load_digit_templates,
    shifted_dice_similarity,
)
from color_nonogram_vision import GridCalibration, clue_cell_rois, normalize_clue_glyph


def _digit_patch(text: str, background: tuple[int, int, int], foreground: tuple[int, int, int]) -> np.ndarray:
    image = Image.new("RGB", (32, 32), background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    box = draw.textbbox((0, 0), text, font=font)
    x = (32 - (box[2] - box[0])) // 2
    y = (32 - (box[3] - box[1])) // 2
    draw.text((x, y), text, fill=foreground, font=font)
    return np.asarray(image)


class ColorNonogramDigitTests(unittest.TestCase):
    def test_blank_patch_has_no_glyph(self):
        glyph, _, foreground_pixels = normalize_clue_glyph(
            np.full((24, 24, 3), (230, 220, 180), dtype=np.uint8)
        )
        self.assertEqual(foreground_pixels, 0)
        self.assertFalse(glyph.any())

    def test_normalization_removes_background_color(self):
        dark, _, _ = normalize_clue_glyph(_digit_patch("7", (45, 75, 95), (255, 255, 255)))
        light, _, _ = normalize_clue_glyph(_digit_patch("7", (240, 210, 80), (0, 0, 0)))
        self.assertGreater(shifted_dice_similarity(dark, light), 0.8)

    def test_template_classifier_returns_expected_label(self):
        one, _, _ = normalize_clue_glyph(_digit_patch("1", (30, 80, 50), (255, 255, 255)))
        seven, _, _ = normalize_clue_glyph(_digit_patch("7", (90, 40, 30), (255, 255, 255)))
        query, _, _ = normalize_clue_glyph(_digit_patch("7", (245, 220, 80), (0, 0, 0)))
        classifier = TemplateDigitClassifier(
            [DigitTemplate(1, one), DigitTemplate(7, seven)]
        )
        candidates = classifier.classify(query, top_k=2)
        self.assertEqual(candidates[0].value, 7)
        self.assertGreater(candidates[0].score, candidates[1].score)

    def test_ambiguous_result_is_rejected(self):
        one, _, _ = normalize_clue_glyph(_digit_patch("1", (30, 80, 50), (255, 255, 255)))
        classifier = TemplateDigitClassifier(
            [DigitTemplate(1, one), DigitTemplate(7, one.copy())]
        )
        decision = classifier.decide(one)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "top candidates are ambiguous")

    def test_calibration_produces_row_and_column_slots(self):
        calibration = GridCalibration("sample.png", (300, 200, 600, 500), 100, 0)
        rois = clue_cell_rois(calibration)
        self.assertTrue(any(axis == "row" for axis, _, _, _ in rois))
        self.assertTrue(any(axis == "column" for axis, _, _, _ in rois))

    def test_new_multi_digit_templates_survive_source_holdout(self):
        manifest = pathlib.Path(__file__).parent / "fixtures" / "color_digit_templates.json"
        templates = load_digit_templates(manifest)

        def source_screenshot(source: str) -> str:
            end = source.find(".png")
            return source[: end + 4] if end >= 0 else source

        for label in (11, 12, 14):
            for query in (template for template in templates if template.label == label):
                training = [
                    template
                    for template in templates
                    if source_screenshot(template.source) != source_screenshot(query.source)
                ]
                with self.subTest(label=label, source=query.source):
                    decision = TemplateDigitClassifier(training).decide(query.glyph)
                    self.assertTrue(decision.accepted)
                    self.assertEqual(decision.candidates[0].value, label)

    def test_thirteen_remains_in_top_three_across_sources(self):
        manifest = pathlib.Path(__file__).parent / "fixtures" / "color_digit_templates.json"
        templates = load_digit_templates(manifest)

        def source_screenshot(source: str) -> str:
            end = source.find(".png")
            return source[: end + 4] if end >= 0 else source

        for query in (template for template in templates if template.label == 13):
            training = [
                template
                for template in templates
                if source_screenshot(template.source) != source_screenshot(query.source)
            ]
            with self.subTest(source=query.source):
                candidates = TemplateDigitClassifier(training).classify(query.glyph, top_k=3)
                self.assertIn(13, [candidate.value for candidate in candidates])

    def test_real_template_manifest_is_self_consistent(self):
        manifest = pathlib.Path(__file__).parent / "fixtures" / "color_digit_templates.json"
        templates = load_digit_templates(manifest)
        classifier = TemplateDigitClassifier(templates)
        self.assertEqual(classifier.labels, tuple(range(1, 16)))
        for template in templates:
            with self.subTest(source=template.source):
                self.assertEqual(classifier.classify(template.glyph, top_k=1)[0].value, template.label)


if __name__ == "__main__":
    unittest.main()
