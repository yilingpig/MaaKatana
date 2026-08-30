from __future__ import annotations

import pathlib
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import color_nonogram_solver as solver
from color_nonogram_digits import DigitCandidate
from color_nonogram_solver import (
    _clue_roi_within_bounds,
    _palette_colors,
    _prefer_template_digit,
    _rank_digit_candidates,
    _select_color_grid_candidate,
    _clue_cell_has_mark,
    _skip_unmarked_small_clue,
    _use_small_clue_guard,
    detect_color_grid,
)


IMAGE_ROOT = pathlib.Path(__file__).parent / "fixtures" / ".." / ".." / "assets" / "resource" / "image"
LIVE_SCREENSHOT = (
    pathlib.Path(__file__).parent.parent
    / "debug"
    / "color_nonogram_live"
    / "screencap_20260819_112131_260562.png"
)


class ColorNonogramSolverTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def test_direct_solve_timeout_does_not_trigger_targeted_clue_reread(self):
        timeout = solver.SolutionSearchTimeout("timeout=3.0s")
        with mock.patch.object(solver, "_solve", side_effect=timeout), mock.patch.object(
            solver,
            "_colored_clue_observations",
            side_effect=AssertionError("targeted clue reread must not run after solve timeout"),
        ) as reread:
            with self.assertRaisesRegex(
                solver.SolutionSearchTimeout,
                "before OCR candidate recovery",
            ):
                solver._solve_with_candidate_recovery(
                    context=None,
                    image=self.image,
                    geometry=None,
                    palette=[object()],
                    rows=[[(0, 1)]],
                    columns=[[(0, 1)]],
                )
        reread.assert_not_called()
    def test_row_clue_roi_must_not_cross_detected_left_boundary(self):
        self.assertTrue(
            _clue_roi_within_bounds((617, 300, 28, 28), self.image, clue_left=612.0)
        )
        self.assertFalse(
            _clue_roi_within_bounds((582, 300, 28, 28), self.image, clue_left=612.0)
        )

    def test_column_clue_roi_must_not_be_clamped_to_screen_top(self):
        self.assertTrue(
            _clue_roi_within_bounds((1171, 18, 28, 28), self.image, clue_top=14.0)
        )
        self.assertFalse(
            _clue_roi_within_bounds((1171, -17, 28, 28), self.image, clue_top=14.0)
        )
        self.assertFalse(
            _clue_roi_within_bounds((1171, 0, 28, 28), self.image, clue_top=14.0)
        )

    def test_digit_candidates_keep_template_seven_when_ocr_says_one(self):
        selected = _rank_digit_candidates({1: 1.0, 7: 0.70}, {7: 0.70})
        self.assertEqual([candidate.value for candidate in selected], [1, 7])

    def test_digit_candidates_keep_template_one_when_ocr_says_seven(self):
        selected = _rank_digit_candidates({7: 1.0, 1: 0.70}, {1: 0.70})
        self.assertEqual([candidate.value for candidate in selected], [7, 1])

    def test_uncertain_one_seven_pair_is_preserved_for_disambiguation(self):
        selected = _rank_digit_candidates({1: 0.66, 7: 0.64}, {1: 0.66, 7: 0.64})
        self.assertEqual({candidate.value for candidate in selected}, {1, 7})

    def test_ordinary_digit_candidate_ranking_is_unchanged(self):
        selected = _rank_digit_candidates({2: 1.0, 3: 0.70}, {3: 0.70})
        self.assertEqual([candidate.value for candidate in selected], [2])
    def test_template_retry_corrects_low_confidence_one_seven_confusion(self):
        candidates = [DigitCandidate(1, 0.898), DigitCandidate(7, 0.623)]
        self.assertEqual(_prefer_template_digit(7, candidates), 1)
        self.assertIsNone(_prefer_template_digit(1, candidates))

    def test_template_retry_can_correct_a_non_one_seven_misread(self):
        candidates = [DigitCandidate(1, 0.898), DigitCandidate(4, 0.706)]
        self.assertEqual(_prefer_template_digit(5, candidates), 1)

    def test_template_retry_refuses_ambiguous_one_seven_candidates(self):
        candidates = [DigitCandidate(1, 0.84), DigitCandidate(7, 0.79)]
        self.assertIsNone(_prefer_template_digit(7, candidates))

    def test_small_clue_guard_covers_recent_cell_size_boundaries(self):
        for cell_size in (37.6, 39.5, 41.3, 42.0):
            with self.subTest(cell_size=cell_size):
                self.assertTrue(_use_small_clue_guard(cell_size, cell_size))
        self.assertFalse(_use_small_clue_guard(42.01, 42.01))

    def test_small_blank_blue_clue_is_skipped_only_when_uniform(self):
        image_path = pathlib.Path(__file__).parent.parent / "debug" / "color_nonogram_live" / "paint_before_20260823_132019_165402.png"
        if not image_path.is_file():
            self.skipTest("recent OCR failure screenshot is not available")
        image = np.asarray(Image.open(image_path).convert("RGB"))
        roi = (528, 258, 32, 32)
        self.assertFalse(_clue_cell_has_mark(image, roi))
        self.assertTrue(_skip_unmarked_small_clue(image, roi, use_small_clue_guard=True))

    def test_real_digit_clue_is_not_silently_skipped(self):
        image_path = pathlib.Path(__file__).parent.parent / "debug" / "color_nonogram_live" / "paint_before_20260823_132019_165402.png"
        if not image_path.is_file():
            self.skipTest("recent clue screenshot is not available")
        image = np.asarray(Image.open(image_path).convert("RGB"))
        roi = (449, 258, 32, 32)
        self.assertTrue(_clue_cell_has_mark(image, roi))
        self.assertFalse(_skip_unmarked_small_clue(image, roi, use_small_clue_guard=True))

    def test_success_regression_keeps_original_path_outside_small_guard(self):
        image_path = pathlib.Path(__file__).parent.parent / "debug" / "color_nonogram_live" / "paint_before_20260823_125844_935180.png"
        if not image_path.is_file():
            self.skipTest("successful regression screenshot is not available")
        image = np.asarray(Image.open(image_path).convert("RGB"))
        roi = (684, 197, 35, 35)
        self.assertTrue(_clue_cell_has_mark(image, roi))
        self.assertFalse(_skip_unmarked_small_clue(image, roi, use_small_clue_guard=False))
    def test_palette_keeps_black_and_gray_swatches(self):
        image = np.full((640, 1280, 3), (240, 220, 170), dtype=np.uint8)
        image[576:640, :640] = (0, 0, 0)
        image[576:640, 640:] = (160, 160, 160)
        image[576:580, :] = (15, 15, 15)
        image[636:, :] = (15, 15, 15)
        image[:, 638:642] = (15, 15, 15)
        palette = _palette_colors(image)
        self.assertEqual(len(palette), 2)
        self.assertEqual(tuple(round(value) for value in palette[0].rgb), (0, 0, 0))
        self.assertEqual(tuple(round(value) for value in palette[1].rgb), (160, 160, 160))

    def test_grid_rejects_known_low_confidence_false_candidates(self):
        for name in ("彩色拼图解题6.png", "彩色拼图解题7.png", "彩色拼图解题13.png", "彩色拼图解题15.png"):
            with self.subTest(name=name):
                image = np.asarray(Image.open(IMAGE_ROOT / name).convert("RGB"))
                self.assertIsNone(detect_color_grid(image))

    def test_grid_candidate_filters_palette_overlap_before_priority_selection(self):
        invalid_high_score = (100.0, 0.01, 2.0, (0.0,), (0.0,))
        valid_aligned_lower_score = (80.0, 0.2, 0.5, (1.0,), (1.0,))
        valid_higher_score = (90.0, 0.4, 0.4, (2.0,), (2.0,))
        selected = _select_color_grid_candidate(
            [invalid_high_score, valid_higher_score, valid_aligned_lower_score],
            (900, 960),
        )
        self.assertEqual(selected, valid_aligned_lower_score)

    def test_grid_candidate_filter_returns_none_when_all_overlap_palette(self):
        candidates = [
            (100.0, 0.1, 1.3, (0.0,), (0.0,)),
            (80.0, 0.2, 1.5, (1.0,), (1.0,)),
        ]
        self.assertIsNone(_select_color_grid_candidate(candidates, (900, 960)))

    def test_black_and_yellow_palette_is_detected(self):
        image = np.asarray(Image.open(IMAGE_ROOT / "彩色拼图解题11.png").convert("RGB"))
        palette = _palette_colors(image)
        self.assertEqual(len(palette), 2)
        self.assertEqual(tuple(round(value) for value in palette[0].rgb), (0, 0, 0))

    def test_live_1080p_grid_prefers_board_below_clue_area(self):
        if not LIVE_SCREENSHOT.is_file():
            self.skipTest("live 1080p regression screenshot is not available")
        image = np.asarray(Image.open(LIVE_SCREENSHOT).convert("RGB"))
        geometry = detect_color_grid(image)
        self.assertIsNotNone(geometry)
        assert geometry is not None
        self.assertEqual((geometry.columns, geometry.rows), (15, 15))
        self.assertAlmostEqual(geometry.x_lines[0], 891, delta=10)
        self.assertAlmostEqual(geometry.x_lines[-1], 1411, delta=10)
        self.assertAlmostEqual(geometry.y_lines[0], 330, delta=10)
        self.assertAlmostEqual(geometry.y_lines[-1], 849, delta=10)
        self.assertIsNotNone(geometry.clue_left)
        self.assertIsNotNone(geometry.clue_top)
        assert geometry.clue_left is not None
        assert geometry.clue_top is not None
        self.assertLess(geometry.clue_left, 530)
        self.assertLess(geometry.clue_top, 30)

        row_slots = set()
        column_slots = set()
        for slot in range(14, -1, -1):
            row_roi = (
                round(geometry.x_lines[0] - (slot + 1) * geometry.cell_width + geometry.cell_width * 0.1),
                round(geometry.y_lines[0] + geometry.cell_height * 0.1),
                round(geometry.cell_width * 0.8),
                round(geometry.cell_height * 0.8),
            )
            if _clue_roi_within_bounds(row_roi, image, clue_left=geometry.clue_left):
                row_slots.add(slot)
            column_roi = (
                round(geometry.x_lines[0] + geometry.cell_width * 0.1),
                round(geometry.y_lines[0] - (slot + 1) * geometry.cell_height + geometry.cell_height * 0.1),
                round(geometry.cell_width * 0.8),
                round(geometry.cell_height * 0.8),
            )
            if _clue_roi_within_bounds(column_roi, image, clue_top=geometry.clue_top):
                column_slots.add(slot)
        self.assertIn(10, row_slots)
        self.assertIn(8, column_slots)


if __name__ == "__main__":
    unittest.main()
