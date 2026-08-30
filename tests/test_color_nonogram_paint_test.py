from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from color_nonogram_paint_test import (
    CellObservation,
    ColorNonogram1080pPaintTest,
    PaintSurfaceError,
    PaintSegment,
    _baseline_mismatches,
    _board_x_constraints,
    _classify_cell,
    _cleanup_success_images,
    _completion_status,
    _estimate_background_rgb,
    _grid_position_status,
    _observe_board,
    _paint_surface_status,
    _paint_segment,
    _paint_action_report,
    _require_paint_surface,
    _palette_editor_detected,
    _select_palette,
    _segments_without_trial,
    _solution_hash,
    _target_segments,
    _trial_segment,
    _target_cells,
    _wait_until_segment_painted,
    _wait_until_segment_painted_with_retry,
    _verify_painted_segment,
    _verify_board,
)
from color_nonogram_core import solve_puzzle
from color_nonogram_solver import (
    PaletteColor,
    _palette_colors,
    _solve_with_candidate_recovery,
    detect_color_grid,
)
from color_nonogram_model import ColorNonogramPuzzle
from nonogram_solver import GridGeometry


PROJECT_ROOT = pathlib.Path(__file__).parent.parent
SAMPLE_REPORT = PROJECT_ROOT / "debug" / "color_nonogram_live" / "report_20260819_115923_547517.json"
PAINT_BEFORE_SAMPLE = (
    PROJECT_ROOT
    / "debug"
    / "color_nonogram_live"
    / "paint_before_20260819_123237_402118.png"
)
PAINT_TRIAL_SAMPLE = (
    PROJECT_ROOT
    / "debug"
    / "color_nonogram_live"
    / "paint_trial_20260819_155751_200603.png"
)
PAINT_EDITOR_SAMPLE = (
    PROJECT_ROOT
    / "debug"
    / "color_nonogram_live"
    / "paint_failure_20260819_171736_249965.png"
)
PAINT_EDITOR_BEFORE_SAMPLE = (
    PROJECT_ROOT
    / "debug"
    / "color_nonogram_live"
    / "paint_before_20260819_171613_139627.png"
)
PAINT_GRID_REFERENCE_SAMPLE = (
    PROJECT_ROOT
    / "debug"
    / "color_nonogram_live"
    / "paint_before_20260819_175654_914328.png"
)
PAINT_GRID_AFTER_SAMPLE = (
    PROJECT_ROOT
    / "debug"
    / "color_nonogram_live"
    / "paint_failure_20260819_175857_964059.png"
)
LATEST_BLANK_SAMPLE = (
    PROJECT_ROOT
    / "debug"
    / "color_nonogram_live"
    / "paint_before_20260822_142212_496246.png"
)


PAINT_CONTENT_COVERAGE_REFERENCE_SAMPLE = (
    PROJECT_ROOT / "debug" / "color_nonogram_live" / "paint_before_20260822_170255_157150.png"
)
PAINT_CONTENT_COVERAGE_SAMPLE = (
    PROJECT_ROOT / "debug" / "color_nonogram_live" / "paint_failure_20260822_170419_836905.png"
)


def _large_geometry() -> GridGeometry:
    return GridGeometry(
        columns=15,
        rows=15,
        x_lines=tuple(929.0 + (481.0 * index / 15.0) for index in range(16)),
        y_lines=tuple(369.0 + (482.0 * index / 15.0) for index in range(16)),
        score=1.0,
    )


def _textured_blank_1080p_board() -> tuple[np.ndarray, GridGeometry]:
    geometry = _large_geometry()
    image = np.full((1080, 1920, 3), (240, 242, 243), dtype=np.int16)
    y, x = np.indices(image.shape[:2])
    texture = ((x + 2 * y) % 9) - 4
    image += texture[:, :, None]
    image = np.clip(image, 0, 255).astype(np.uint8)
    pale_left = round(geometry.x_lines[7]) + 5
    pale_right = round(geometry.x_lines[8]) - 5
    pale_top = round(geometry.y_lines[4]) + 5
    pale_bottom = round(geometry.y_lines[5]) - 5
    image[pale_top:pale_bottom, pale_left:pale_right] = (228, 232, 234)
    for position in geometry.x_lines:
        coordinate = round(position)
        image[:, coordinate - 1 : coordinate + 2] = (70, 70, 70)
    for position in geometry.y_lines:
        coordinate = round(position)
        image[coordinate - 1 : coordinate + 2, :] = (70, 70, 70)
    return image, geometry


def _geometry() -> GridGeometry:
    return GridGeometry(
        columns=2,
        rows=2,
        x_lines=(0.0, 50.0, 100.0),
        y_lines=(0.0, 50.0, 100.0),
        score=1.0,
    )


def _empty_board() -> np.ndarray:
    return np.full((100, 100, 3), (220, 220, 220), dtype=np.uint8)


def _fill_cell(image: np.ndarray, row: int, column: int, color: tuple[int, int, int]) -> None:
    image[11 + row * 50 : 39 + row * 50, 11 + column * 50 : 39 + column * 50] = color


def _fill_x_marker(image: np.ndarray, row: int, column: int) -> None:
    top = 12 + row * 50
    left = 12 + column * 50
    for offset in range(24):
        image[top + offset - 2 : top + offset + 2, left + offset - 2 : left + offset + 2] = (245, 245, 245)
        reverse = 23 - offset
        image[top + offset - 2 : top + offset + 2, left + reverse - 2 : left + reverse + 2] = (245, 245, 245)


class _ScreenshotResult:
    def __init__(self, image):
        self.image = image

    def wait(self):
        return self

    def get(self):
        return self.image


class _ScreenshotController:
    def __init__(self, images):
        self.images = list(images)

    def post_screencap(self):
        return _ScreenshotResult(self.images.pop(0))


class _ScreenshotContext:
    def __init__(self, images):
        self.tasker = SimpleNamespace(controller=_ScreenshotController(images))


class _ActionResult:
    succeeded = True

    def wait(self):
        return self


class _ActionController:
    def __init__(self):
        self.calls = []

    def post_click(self, x, y):
        self.calls.append(("click", x, y))
        return _ActionResult()

    def post_swipe(self, x1, y1, x2, y2, duration):
        self.calls.append(("swipe", x1, y1, x2, y2, duration))
        return _ActionResult()


class ColorNonogramPaintTests(unittest.TestCase):
    def test_grid_position_allows_limited_content_covered_lines(self):
        geometry = GridGeometry(
            columns=15,
            rows=15,
            x_lines=tuple(float(value) for value in range(10, 161, 10)),
            y_lines=tuple(float(value) for value in range(10, 161, 10)),
            score=1.0,
        )
        image = np.full((170, 170, 3), 220, dtype=np.uint8)
        for position in geometry.x_lines:
            coordinate = round(position)
            image[10:161, coordinate - 1 : coordinate + 2] = 70
        for position in geometry.y_lines:
            coordinate = round(position)
            image[coordinate - 1 : coordinate + 2, 10:161] = 70

        for position in geometry.x_lines[:3]:
            coordinate = round(position)
            image[25:161, coordinate - 1 : coordinate + 2] = 220
        coordinate = round(geometry.y_lines[0])
        image[coordinate - 1 : coordinate + 2, 25:161] = 220

        status = _grid_position_status(image, geometry, reference_image=image)

        self.assertTrue(status["detected"])
        self.assertTrue(status["limited_coverage_exception"])
        self.assertEqual(status["low_support_line_count"], 4)
        self.assertLess(status["minimum_line_support"], 0.55)
        self.assertGreaterEqual(status["mean_line_support"], 0.89)

    def test_grid_position_allows_many_local_color_and_x_obstructions(self):
        geometry = GridGeometry(
            columns=15,
            rows=15,
            x_lines=tuple(float(value) for value in range(10, 161, 10)),
            y_lines=tuple(float(value) for value in range(10, 161, 10)),
            score=1.0,
        )
        image = np.full((170, 170, 3), 220, dtype=np.uint8)
        for position in geometry.x_lines:
            coordinate = round(position)
            image[10:161, coordinate - 1 : coordinate + 2] = 70
        for position in geometry.y_lines:
            coordinate = round(position)
            image[coordinate - 1 : coordinate + 2, 10:161] = 70

        color = np.array((220, 60, 60), dtype=np.uint8)
        for row, column in ((1, 1), (1, 4), (1, 7), (1, 10), (1, 13),
                            (4, 2), (4, 5), (4, 8), (4, 11), (4, 14),
                            (7, 1), (7, 4), (7, 7), (7, 10), (7, 13),
                            (10, 2), (10, 5), (10, 8), (10, 11), (10, 14)):
            top = 10 + row * 10
            left = 10 + column * 10
            image[top + 2 : top + 8, left + 2 : left + 8] = color
        for row, column in ((2, 2), (2, 5), (2, 8), (2, 11), (2, 14),
                            (5, 1), (5, 4), (5, 7), (5, 10), (5, 13),
                            (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
                            (11, 1), (11, 4), (11, 7), (11, 10), (11, 13)):
            top = 10 + row * 10
            left = 10 + column * 10
            for offset in range(5):
                image[top + 2 + offset, left + 2 + offset] = (220, 220, 220)
                image[top + 2 + offset, left + 7 - offset] = (220, 220, 220)

        for position in geometry.x_lines[1:-1]:
            coordinate = round(position)
            image[10:161, coordinate - 1 : coordinate + 2][::5] = 220
        for position in geometry.y_lines[1:-1]:
            coordinate = round(position)
            image[coordinate - 1 : coordinate + 2, 10:161][:, ::5] = 220

        status = _grid_position_status(image, geometry, reference_image=image)

        self.assertTrue(status["detected"])
        self.assertGreaterEqual(status["mean_line_support"], 0.85)
        self.assertLess(status["low_support_line_count"], 4)

    def test_grid_position_rejects_whole_board_translation(self):
        geometry = GridGeometry(
            columns=15,
            rows=15,
            x_lines=tuple(float(value) for value in range(10, 161, 10)),
            y_lines=tuple(float(value) for value in range(10, 161, 10)),
            score=1.0,
        )
        reference = np.full((190, 190, 3), 220, dtype=np.uint8)
        for position in geometry.x_lines:
            coordinate = round(position)
            reference[10:161, coordinate - 1 : coordinate + 2] = 70
        for position in geometry.y_lines:
            coordinate = round(position)
            reference[coordinate - 1 : coordinate + 2, 10:161] = 70

        translated = np.full_like(reference, 220)
        translated[20:171, 20:171] = reference[10:161, 10:161]
        status = _grid_position_status(translated, geometry, reference_image=reference)

        self.assertFalse(status["detected"])
        self.assertEqual(status["check"], "anchored_grid_lines")
        self.assertTrue(status["reference_cached"])
    def test_grid_position_rejects_blank_or_structurally_weak_grid(self):
        geometry = GridGeometry(
            columns=15,
            rows=15,
            x_lines=tuple(float(value) for value in range(10, 161, 10)),
            y_lines=tuple(float(value) for value in range(10, 161, 10)),
            score=1.0,
        )
        blank = np.full((170, 170, 3), 220, dtype=np.uint8)
        blank_status = _grid_position_status(blank, geometry, reference_image=blank)
        self.assertFalse(blank_status["detected"])
        self.assertFalse(blank_status["limited_coverage_exception"])

        weak = blank.copy()
        for position in geometry.x_lines[:8]:
            coordinate = round(position)
            weak[10:161, coordinate - 1 : coordinate + 2] = 70
        for position in geometry.y_lines[:8]:
            coordinate = round(position)
            weak[coordinate - 1 : coordinate + 2, 10:161] = 70
        weak_status = _grid_position_status(weak, geometry, reference_image=weak)
        self.assertFalse(weak_status["detected"])
        self.assertFalse(weak_status["limited_coverage_exception"])

    def test_completion_status_accepts_structural_continue_bar(self):
        class CompletionContext:
            def run_recognition_direct(self, *_args):
                return SimpleNamespace(hit=False)

        image = np.full((1080, 1920, 3), 240, dtype=np.uint8)
        image[970:975] = (0, 180, 190)
        image[975:1060] = (20, 20, 20)
        image[1060:1065] = (0, 180, 190)
        status = _completion_status(CompletionContext(), image)
        self.assertTrue(status["detected"])
        self.assertTrue(status["completion_bar_hit"])

    def test_completion_status_accepts_continue_template(self):
        class CompletionContext:
            def __init__(self):
                self.templates = []

            def run_recognition_direct(self, _recognition_type, config, _image):
                if hasattr(config, "template"):
                    self.templates.append(config.template)
                    return SimpleNamespace(hit=config.template == ["继续.png"])
                return None

        context = CompletionContext()
        status = _completion_status(context, np.zeros((1080, 1920, 3), dtype=np.uint8))

        self.assertTrue(status["detected"])
        self.assertFalse(status["template_hit"])
        self.assertTrue(status["continue_hit"])
        self.assertFalse(status["ocr_hit"])
        self.assertEqual(context.templates, [["恭喜过关了.png"], ["继续.png"]])

    def test_success_cleanup_removes_only_recorded_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            before = pathlib.Path(directory) / "paint_before_run.png"
            trial = pathlib.Path(directory) / "paint_trial_run.png"
            report = pathlib.Path(directory) / "paint_after_run.json"
            before.write_bytes(b"before")
            trial.write_bytes(b"trial")
            report.write_bytes(b"report")

            removed = _cleanup_success_images([str(before), str(trial)])

            self.assertEqual(set(removed), {str(before), str(trial)})
            self.assertFalse(before.exists())
            self.assertFalse(trial.exists())
            self.assertTrue(report.exists())

    def test_failure_path_keeps_pngs_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            failure = pathlib.Path(directory) / "paint_failure_run.png"
            failure.write_bytes(b"failure")

            self.assertTrue(failure.exists())

    def test_color_nonogram_keeps_only_paint_entry(self):
        for relative in ("assets/interface.json", "install/interface.json"):
            interface = json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
            names = [task["name"] for task in interface["task"]]
            self.assertNotIn("彩色数织1080p测试", names)
            self.assertNotIn("彩色数织1080p识别DryRun", names)
            self.assertIn("彩色数织1080p安全绘制测试", names)
            paint_task = next(
                task for task in interface["task"]
                if task["name"] == "彩色数织1080p安全绘制测试"
            )
            self.assertIn("AllowColorNonogramPaintTest", paint_task["option"])

    def test_pipeline_keeps_paint_nodes_only(self):
        removed = {
            "彩色数织1080p测试",
            "彩色数织1080p测试完成",
            "彩色数织1080p测试失败",
            "彩色数织1080p识别DryRun",
            "彩色数织1080p识别DryRun完成",
            "彩色数织1080p识别DryRun失败",
        }
        for relative in ("assets/resource/pipeline/Pipeline7.json", "install/resource/pipeline/Pipeline7.json"):
            pipeline = json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
            self.assertTrue(removed.isdisjoint(pipeline))
            self.assertIn("ColorNonogram1080pPaintTest", pipeline)
            self.assertIn("ColorNonogram1080pPaintTestSuccess", pipeline)
            self.assertIn("ColorNonogram1080pPaintTestFailure", pipeline)

    def test_paint_gate_is_closed_without_explicit_allow_paint(self):
        class GateContext:
            def __init__(self):
                self.next_nodes = None

            def override_next(self, node_name, nodes):
                self.next_nodes = (node_name, nodes)

        context = GateContext()
        argv = SimpleNamespace(
            node_name="ColorNonogram1080pPaintTest",
            custom_action_param={"mode": "paint_test", "allow_paint": False},
        )
        with patch("color_nonogram_paint_test._write_report", return_value="test-report"):
            result = ColorNonogram1080pPaintTest().run(context, argv)
        self.assertTrue(result)
        self.assertEqual(context.next_nodes[1], ["ColorNonogram1080pPaintTestFailure"])

    def test_cell_classification_distinguishes_palette_and_background(self):
        image = _empty_board()
        _fill_cell(image, 0, 0, (230, 0, 0))
        palette = [
            PaletteColor((10, 10), (230.0, 0.0, 0.0)),
            PaletteColor((20, 20), (0.0, 0.0, 230.0)),
        ]
        board = _observe_board(image, _geometry(), palette)
        self.assertEqual(board[0][0].state, "palette")
        self.assertEqual(board[0][0].palette_index, 0)
        self.assertEqual(board[1][1].state, "background")

    def test_unknown_cell_is_not_treated_as_background(self):
        image = _empty_board()
        _fill_cell(image, 0, 1, (80, 20, 150))
        palette = [
            PaletteColor((10, 10), (230.0, 0.0, 0.0)),
            PaletteColor((20, 20), (0.0, 0.0, 230.0)),
        ]
        board = _observe_board(image, _geometry(), palette)
        self.assertEqual(board[0][1].state, "unknown")
        self.assertEqual(len(_baseline_mismatches(board)), 1)

    def test_blank_textured_board_grid_lines_and_light_cells_have_no_x_markers(self):
        image, geometry = _textured_blank_1080p_board()
        board = _observe_board(image, geometry, [])

        cells = [cell for line in board for cell in line]
        self.assertTrue(all(cell.state == "background" for cell in cells))
        self.assertTrue(all(cell.x_marker_score == 0.0 for cell in cells))
        self.assertEqual(_board_x_constraints(board)[2:], ([0] * 15, [0] * 15))

    def test_real_x_marker_is_detected_against_background(self):
        image = _empty_board()
        _fill_x_marker(image, 0, 1)
        palette = [PaletteColor((10, 10), (230.0, 0.0, 0.0))]

        board = _observe_board(image, _geometry(), palette)

        self.assertEqual(board[0][1].state, "x_marker")
        self.assertGreaterEqual(board[0][1].x_marker_score, 0.45)
        self.assertEqual(board[0][0].state, "background")

    def test_partial_real_x_markers_are_rejected_as_uncertain(self):
        image = _empty_board()
        _fill_x_marker(image, 0, 1)
        palette = [PaletteColor((10, 10), (230.0, 0.0, 0.0))]
        board = _observe_board(image, _geometry(), palette)

        with self.assertRaisesRegex(ValueError, "partial X markers are uncertain"):
            _board_x_constraints(board)

    @unittest.skipUnless(LATEST_BLANK_SAMPLE.is_file(), "latest blank 1080p screenshot is not available")
    def test_latest_blank_1080p_screenshot_has_no_x_markers(self):
        from PIL import Image

        image = np.asarray(Image.open(LATEST_BLANK_SAMPLE).convert("RGB"))
        geometry = detect_color_grid(image)
        self.assertIsNotNone(geometry)
        assert geometry is not None
        palette = _palette_colors(image)
        board = _observe_board(image, geometry, palette)

        cells = [cell for line in board for cell in line]
        self.assertTrue(all(cell.state == "background" for cell in cells))
        self.assertTrue(all(cell.x_marker_score == 0.0 for cell in cells))
        self.assertEqual(_baseline_mismatches(board), [])
        self.assertEqual(_board_x_constraints(board)[2:], ([0] * 15, [0] * 15))

    @unittest.skipUnless(PAINT_TRIAL_SAMPLE.is_file(), "paint trial regression screenshot is not available")
    def test_live_click_highlight_is_not_classified_as_x_marker(self):
        from PIL import Image

        before = np.asarray(Image.open(PAINT_BEFORE_SAMPLE).convert("RGB"))
        trial = np.asarray(Image.open(PAINT_TRIAL_SAMPLE).convert("RGB"))
        geometry = detect_color_grid(before)
        self.assertIsNotNone(geometry)
        assert geometry is not None
        palette = _palette_colors(before)
        cell = _classify_cell(
            trial,
            geometry,
            0,
            7,
            palette,
            _estimate_background_rgb(before, geometry),
        )
        self.assertEqual(cell.state, "selected")
        self.assertNotEqual(cell.state, "x_marker")

    def test_trial_segment_prefers_high_contrast_short_segment(self):
        segments = [
            PaintSegment(2, 0, 0, 4),
            PaintSegment(0, 4, 3, 3),
            PaintSegment(1, 8, 1, 2),
        ]
        palette = [
            PaletteColor((10, 10), (235.0, 235.0, 235.0)),
            PaletteColor((20, 20), (210.0, 40.0, 40.0)),
            PaletteColor((30, 30), (240.0, 240.0, 240.0)),
        ]
        selected = _trial_segment(segments, palette, (238.0, 238.0, 238.0))
        self.assertEqual(selected, segments[2])

    def test_nonfirst_trial_keeps_first_segment_and_original_order(self):
        segments = [
            PaintSegment(3, 0, 0, 6),
            PaintSegment(5, 3, 6, 6),
            PaintSegment(1, 4, 2, 4),
        ]
        trial_segment = segments[1]
        remaining = _segments_without_trial(segments, trial_segment)
        self.assertIs(remaining[0], segments[0])
        self.assertIs(remaining[1], segments[2])
        self.assertEqual(
            [
                (segment.color, segment.row, segment.start_column, segment.end_column)
                for segment in remaining
            ],
            [(3, 0, 0, 6), (1, 4, 2, 4)],
        )
        self.assertNotIn(trial_segment, remaining)

    def test_target_segments_split_by_color_and_gap(self):
        solution = [[0 for _ in range(15)] for _ in range(15)]
        solution[0][:6] = [1, 1, 2, 0, 2, 2]
        solution[1][4] = 3
        segments = _target_segments(solution, 3)
        self.assertEqual(
            [(item.color, item.row, item.start_column, item.end_column, item.length) for item in segments],
            [(0, 0, 0, 1, 2), (1, 0, 2, 2, 1), (1, 0, 4, 5, 2), (2, 1, 4, 4, 1)],
        )

    def test_paint_segment_uses_click_pair_for_two_cells_without_swipe(self):
        controller = _ActionController()
        context = SimpleNamespace(tasker=SimpleNamespace(controller=controller))
        geometry = _geometry()
        single_report = _paint_action_report(PaintSegment(0, 0, 0, 0), 1)
        pair_report = _paint_action_report(PaintSegment(0, 0, 0, 1), 1)
        _paint_segment(
            context,
            geometry,
            PaintSegment(0, 0, 0, 0),
            action_report=single_report,
        )
        _paint_segment(
            context,
            geometry,
            PaintSegment(0, 0, 0, 1),
            action_report=pair_report,
        )
        self.assertEqual(controller.calls, [
            ("click", 25, 25),
            ("click", 25, 25),
            ("click", 75, 25),
        ])
        self.assertEqual(single_report["operation"], "click")
        self.assertIsNone(single_report["duration_ms"])
        self.assertEqual(pair_report["operation"], "click_pair")
        self.assertIsNone(pair_report["duration_ms"])
        self.assertEqual(
            pair_report["clicks"],
            [
                {"row": 0, "column": 0, "center": [25, 25], "succeeded": True},
                {"row": 0, "column": 1, "center": [75, 25], "succeeded": True},
            ],
        )

    def test_click_pair_waits_without_retry_action_when_unstable(self):
        context = SimpleNamespace(tasker=SimpleNamespace(controller=_ActionController()))
        segment = PaintSegment(0, 0, 0, 1)
        board = {
            (0, 0): CellObservation(0, 0, (25, 25), (245, 245, 245), "x_marker", None, 1.0, 0.0),
            (0, 1): CellObservation(0, 1, (75, 25), (245, 245, 245), "x_marker", None, 1.0, 0.0),
        }
        observations = [{
            "cells": [
                {"state": "x_marker", "palette_index": None},
                {"state": "x_marker", "palette_index": None},
            ],
            "delay_ms": 100,
            "stable": False,
        }]
        palette = [PaletteColor((10, 10), (230.0, 0.0, 0.0))]
        action_report = _paint_action_report(segment, 1)
        with patch(
            "color_nonogram_paint_test._wait_until_segment_painted",
            return_value=(np.zeros((100, 100, 3), dtype=np.uint8), board, observations, False),
        ) as wait, patch("color_nonogram_paint_test._paint_segment") as paint:
            result = _wait_until_segment_painted_with_retry(
                context,
                (100, 100),
                _geometry(),
                palette,
                (100.0, 100.0, 100.0),
                segment,
                initial_delay=0.0,
                poll_interval=0.0,
                timeout=1.2,
                action_report=action_report,
            )
        self.assertFalse(result[3])
        self.assertEqual(wait.call_count, 1)
        paint.assert_not_called()
        self.assertIsNone(result[5])
        self.assertEqual(result[4]["operation"], "click_pair")
        self.assertIsNone(result[4]["duration_ms"])
        self.assertEqual(result[4]["status"], "observed")
        self.assertEqual(len(result[6]), 1)

    def test_wait_never_repaints_unstable_segments_of_any_length(self):
        context = SimpleNamespace(tasker=SimpleNamespace(controller=_ActionController()))
        board = {
            (0, 0): CellObservation(0, 0, (25, 25), (245, 245, 245), "x_marker", None, 1.0, 0.0),
            (0, 1): CellObservation(0, 1, (75, 25), (245, 245, 245), "x_marker", None, 1.0, 0.0),
            (0, 2): CellObservation(0, 2, (125, 25), (245, 245, 245), "x_marker", None, 1.0, 0.0),
        }
        observations = [{
            "cells": [
                {"state": "x_marker", "palette_index": None},
            ],
            "delay_ms": 100,
            "stable": False,
        }]
        palette = [PaletteColor((10, 10), (230.0, 0.0, 0.0))]
        with (
            patch(
                "color_nonogram_paint_test._wait_until_segment_painted",
                return_value=(np.zeros((100, 100, 3), dtype=np.uint8), board, observations, False),
            ) as wait,
            patch("color_nonogram_paint_test._paint_segment") as paint,
        ):
            for end_column in (0, 1, 2):
                with self.subTest(length=end_column + 1):
                    action_report = _paint_action_report(PaintSegment(0, 0, 0, end_column), 1)
                    result = _wait_until_segment_painted_with_retry(
                        context,
                        (100, 100),
                        _geometry(),
                        palette,
                        (100.0, 100.0, 100.0),
                        PaintSegment(0, 0, 0, end_column),
                        initial_delay=0.0,
                        poll_interval=0.0,
                        timeout=1.2,
                        action_report=action_report,
                    )
                    self.assertFalse(result[3])
                    self.assertIsNone(result[5])
                    self.assertEqual(len(result[6]), 1)
                    self.assertEqual(result[4], action_report)
        self.assertEqual(wait.call_count, 3)
        paint.assert_not_called()

    def test_click_pair_waits_for_stable_palette_without_retry_action(self):
        context = SimpleNamespace(tasker=SimpleNamespace(controller=_ActionController()))
        segment = PaintSegment(0, 0, 0, 1)
        board = {
            (0, 0): CellObservation(0, 0, (25, 25), (230, 0, 0), "palette", 0, 0.0, 1.0),
            (0, 1): CellObservation(0, 1, (75, 25), (230, 0, 0), "palette", 0, 0.0, 1.0),
        }
        observations = [{
            "cells": [
                {"state": "palette", "palette_index": 0},
                {"state": "palette", "palette_index": 0},
            ],
            "delay_ms": 200,
            "stable": True,
        }]
        palette = [PaletteColor((10, 10), (230.0, 0.0, 0.0))]
        with patch(
            "color_nonogram_paint_test._wait_until_segment_painted",
            return_value=(np.zeros((100, 100, 3), dtype=np.uint8), board, observations, True),
        ) as wait, patch("color_nonogram_paint_test._paint_segment") as paint:
            result = _wait_until_segment_painted_with_retry(
                context,
                (100, 100),
                _geometry(),
                palette,
                (100.0, 100.0, 100.0),
                segment,
                initial_delay=0.0,
                poll_interval=0.0,
                timeout=1.2,
            )
        self.assertTrue(result[3])
        self.assertEqual(wait.call_count, 1)
        paint.assert_not_called()
        self.assertIsNone(result[5])
        self.assertEqual(result[4]["operation"], "click_pair")
        self.assertIsNone(result[4]["duration_ms"])

    def test_paint_segment_reuses_verified_surface_status_for_same_frame(self):
        controller = _ActionController()
        context = SimpleNamespace(tasker=SimpleNamespace(controller=controller))
        status = {
            "safe": True,
            "palette_editor_detected": False,
            "grid_detected": True,
            "selected_palette_index": 0,
        }
        with patch("color_nonogram_paint_test._require_paint_surface") as require:
            returned = _paint_segment(
                context,
                _geometry(),
                PaintSegment(0, 0, 0, 0),
                expected_resolution=(100, 100),
                palette=[PaletteColor((10, 10), (230.0, 0.0, 0.0))],
                selected_palette_index=0,
                verified_surface_status=status,
            )
        self.assertIs(returned, status)
        require.assert_not_called()
        self.assertEqual(controller.calls, [("click", 25, 25)])

    def test_fast_settle_falls_back_to_full_settle_without_repainting(self):
        context = SimpleNamespace(tasker=SimpleNamespace(controller=_ActionController()))
        segment = PaintSegment(0, 0, 0, 0)
        palette = [PaletteColor((10, 10), (230.0, 0.0, 0.0))]
        board = {(0, 0): CellObservation(0, 0, (25, 25), (245, 245, 245), "x_marker", None, 1.0, 0.0)}
        observations = [{"state": "x_marker", "palette_index": None, "stable": False}]
        stable_board = {(0, 0): CellObservation(0, 0, (25, 25), (230, 0, 0), "palette", 0, 0.0, 1.0)}
        stable_observations = [{"state": "palette", "palette_index": 0, "stable": True}]
        with patch(
            "color_nonogram_paint_test._wait_until_segment_painted",
            side_effect=[
                (np.zeros((100, 100, 3), dtype=np.uint8), board, observations, False),
                (np.zeros((100, 100, 3), dtype=np.uint8), stable_board, stable_observations, True),
            ],
        ) as wait:
            result = _wait_until_segment_painted_with_retry(
                context,
                (100, 100),
                _geometry(),
                palette,
                (100.0, 100.0, 100.0),
                segment,
                initial_delay=0.12,
                poll_interval=0.08,
                timeout=1.2,
                fast_initial_delay=0.06,
                fast_poll_interval=0.04,
                fast_timeout=0.65,
            )
        self.assertTrue(result[3])
        self.assertEqual(wait.call_count, 2)
        self.assertEqual(len(result[6]), 2)
        self.assertEqual(result[6][1]["fallback"], "full_settle")

    def test_same_palette_selection_is_not_clicked_twice(self):
        controller = _ActionController()
        context = SimpleNamespace(tasker=SimpleNamespace(controller=controller))
        palette = [
            PaletteColor((10, 10), (230.0, 0.0, 0.0)),
            PaletteColor((20, 20), (0.0, 0.0, 230.0)),
        ]
        safe_surface = {
            "safe": True,
            "palette_editor_detected": False,
            "grid_detected": True,
            "palette_complete": True,
            "selected_candidates": [],
            "selected_border_detected": False,
            "selected_palette_index": None,
            "palette_buttons": [],
        }
        with (
            patch("color_nonogram_paint_test._capture", return_value=np.zeros((100, 100, 3), dtype=np.uint8)),
            patch("color_nonogram_paint_test._require_paint_surface", return_value=safe_surface),
            patch("color_nonogram_paint_test.time.sleep", return_value=None),
        ):
            selected, first, _ = _select_palette(
                context,
                palette,
                0,
                expected_resolution=(100, 100),
                geometry=_geometry(),
                selected_palette_index=None,
            )
            selected, second, _ = _select_palette(
                context,
                palette,
                0,
                expected_resolution=(100, 100),
                geometry=_geometry(),
                selected_palette_index=selected,
            )
            selected, third, _ = _select_palette(
                context,
                palette,
                1,
                expected_resolution=(100, 100),
                geometry=_geometry(),
                selected_palette_index=selected,
            )
            _, fourth, _ = _select_palette(
                context,
                palette,
                1,
                expected_resolution=(100, 100),
                geometry=_geometry(),
                selected_palette_index=selected,
            )
        self.assertEqual(controller.calls, [("click", 10, 10), ("click", 20, 20)])
        self.assertTrue(first["clicked"])
        self.assertFalse(second["clicked"])
        self.assertTrue(third["clicked"])
        self.assertFalse(fourth["clicked"])

    def test_editor_surface_is_rejected_before_board_action(self):
        controller = _ActionController()
        context = SimpleNamespace(tasker=SimpleNamespace(controller=controller))
        error = PaintSurfaceError("palette editor detected", {"safe": False})
        palette = [PaletteColor((10, 10), (230.0, 0.0, 0.0))]
        with patch("color_nonogram_paint_test._require_paint_surface", side_effect=error):
            with self.assertRaises(PaintSurfaceError):
                _paint_segment(
                    context,
                    _geometry(),
                    PaintSegment(0, 0, 0, 0),
                    expected_resolution=(100, 100),
                    palette=palette,
                    selected_palette_index=0,
                )
        self.assertEqual(controller.calls, [])

    @unittest.skipUnless(
        PAINT_EDITOR_SAMPLE.is_file() and PAINT_EDITOR_BEFORE_SAMPLE.is_file(),
        "palette editor regression screenshots are not available",
    )
    def test_palette_editor_detection_invalidates_surface(self):
        from PIL import Image

        before = np.asarray(Image.open(PAINT_EDITOR_BEFORE_SAMPLE).convert("RGB"))
        editor = np.asarray(Image.open(PAINT_EDITOR_SAMPLE).convert("RGB"))
        geometry = detect_color_grid(before)
        self.assertIsNotNone(geometry)
        assert geometry is not None
        palette = _palette_colors(before)
        self.assertFalse(_palette_editor_detected(before))
        self.assertTrue(_paint_surface_status(before, geometry, palette)["safe"])
        self.assertTrue(_palette_editor_detected(editor))
        editor_status = _paint_surface_status(editor, geometry, palette)
        self.assertTrue(editor_status["palette_editor_detected"])
        self.assertFalse(editor_status["safe"])

    def test_palette_editor_detection_rejects_blank_board_white_cells(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image[120:900, 480:1440] = (235, 245, 246)
        for x in range(480, 1441, 64):
            image[120:900, x : x + 2] = (50, 60, 60)
        for y in range(120, 901, 64):
            image[y : y + 2, 480:1441] = (50, 60, 60)
        self.assertFalse(_palette_editor_detected(image))

    def test_palette_editor_detection_requires_modal_top_and_bottom_edges(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image[5:1075, 480:1440] = 255
        image[20:1060, 575:1275] = (20, 30, 100)
        self.assertTrue(_palette_editor_detected(image))
        image[990:1080, 480:1440] = 0
        self.assertFalse(_palette_editor_detected(image))


    @unittest.skipUnless(
        PAINT_GRID_REFERENCE_SAMPLE.is_file() and PAINT_GRID_AFTER_SAMPLE.is_file(),
        "cached grid regression screenshots are not available",
    )
    def test_cached_grid_position_survives_board_color_changes(self):
        from PIL import Image

        reference = np.asarray(Image.open(PAINT_GRID_REFERENCE_SAMPLE).convert("RGB"))
        current = np.asarray(Image.open(PAINT_GRID_AFTER_SAMPLE).convert("RGB"))
        geometry = detect_color_grid(reference)
        self.assertIsNotNone(geometry)
        assert geometry is not None
        palette = _palette_colors(reference)
        status = _paint_surface_status(current, geometry, palette, reference)
        self.assertTrue(status["grid_detected"])
        self.assertEqual(status["grid_detection_method"], "anchored")
        self.assertEqual(status["grid_position"]["check"], "anchored_grid_lines")
        self.assertTrue(status["grid_position"]["reference_cached"])

    def test_paint_surface_uses_full_detection_fallback_once_when_anchor_check_fails(self):
        geometry = _geometry()
        palette = [PaletteColor((10, 10), (0.0, 0.0, 0.0))]
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        incomplete_status = {
            "safe": False,
            "palette_editor_detected": False,
            "grid_detected": False,
            "grid": None,
            "palette_complete": True,
            "selected_candidates": [],
        }
        with (
            patch("color_nonogram_paint_test._paint_surface_status", return_value=incomplete_status),
            patch("color_nonogram_paint_test.detect_color_grid", return_value=geometry) as detect,
        ):
            status = _require_paint_surface(
                SimpleNamespace(),
                (100, 100),
                geometry,
                palette,
                image=image,
            )
        self.assertTrue(status["safe"])
        self.assertTrue(status["grid_detected"])
        self.assertEqual(status["grid_detection_method"], "full_detect_fallback")
        self.assertEqual(detect.call_count, 1)

    def test_palette_selection_reuses_supplied_surface_image(self):
        controller = _ActionController()
        context = SimpleNamespace(tasker=SimpleNamespace(controller=controller))
        palette = [PaletteColor((10, 10), (230.0, 0.0, 0.0))]
        safe_surface = {
            "safe": True,
            "palette_editor_detected": False,
            "grid_detected": True,
            "palette_complete": True,
            "selected_candidates": [],
            "selected_border_detected": False,
            "selected_palette_index": None,
            "palette_buttons": [],
        }
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with (
            patch("color_nonogram_paint_test._capture") as capture,
            patch("color_nonogram_paint_test._require_paint_surface", return_value=safe_surface),
        ):
            selected, selection, returned_image = _select_palette(
                context,
                palette,
                0,
                expected_resolution=(100, 100),
                geometry=_geometry(),
                selected_palette_index=0,
                surface_image=image,
            )
        self.assertEqual(selected, 0)
        self.assertFalse(selection["clicked"])
        self.assertIs(returned_image, image)
        capture.assert_not_called()
        self.assertEqual(controller.calls, [])

    def test_wait_until_segment_painted_ignores_highlight_until_two_stable_palette_reads(self):
        geometry = GridGeometry(
            columns=2,
            rows=2,
            x_lines=(0.0, 50.0, 100.0),
            y_lines=(0.0, 50.0, 100.0),
            score=1.0,
        )
        palette = [
            PaletteColor((10, 10), (230.0, 0.0, 0.0)),
            PaletteColor((20, 20), (0.0, 0.0, 230.0)),
        ]
        background = np.full((100, 100, 3), (100, 100, 100), dtype=np.uint8)
        highlight = background.copy()
        _fill_cell(highlight, 0, 0, (172, 200, 178))
        painted = background.copy()
        _fill_cell(painted, 0, 0, (230, 0, 0))
        context = _ScreenshotContext([highlight, painted, painted])
        segment = PaintSegment(0, 0, 0, 0)
        with patch("color_nonogram_paint_test.time.sleep", return_value=None):
            _, board, observations, stable = _wait_until_segment_painted(
                context,
                (100, 100),
                geometry,
                palette,
                (100.0, 100.0, 100.0),
                segment,
                initial_delay=0.0,
                poll_interval=0.0,
                timeout=2.0,
            )
        self.assertTrue(stable)
        self.assertEqual(set(board), {(0, 0)})
        self.assertEqual(board[(0, 0)].state, "palette")
        self.assertEqual([item["state"] for item in observations], ["selected", "palette", "palette"])
        self.assertTrue(observations[-1]["stable"])

    @unittest.skipUnless(PAINT_BEFORE_SAMPLE.is_file(), "paint regression screenshot is not available")
    def test_live_paint_baseline_recognizes_background_and_first_column_x(self):
        from PIL import Image

        image = np.asarray(Image.open(PAINT_BEFORE_SAMPLE).convert("RGB"))
        geometry = detect_color_grid(image)
        self.assertIsNotNone(geometry)
        assert geometry is not None
        palette = _palette_colors(image)
        board = _observe_board(image, geometry, palette)
        self.assertTrue(all(board[row][0].state == "x_marker" for row in range(15)))
        self.assertTrue(
            all(
                board[row][column].state == "background"
                for row in range(15)
                for column in range(1, 15)
            )
        )
        self.assertEqual(_baseline_mismatches(board), [])
        empty_rows, empty_columns, row_counts, column_counts = _board_x_constraints(board)
        self.assertEqual(empty_rows, set())
        self.assertEqual(empty_columns, {0})
        self.assertEqual(row_counts, [1] * 15)
        self.assertEqual(column_counts[0], 15)

    def test_white_palette_fill_wins_over_x_marker_heuristic(self):
        palette = [
            PaletteColor((10, 10), (240.0, 242.0, 243.0)),
            PaletteColor((20, 20), (230.0, 0.0, 0.0)),
        ]
        image = np.full((100, 100, 3), (130, 130, 130), dtype=np.uint8)
        _fill_cell(image, 0, 0, (240, 242, 243))

        observation = _classify_cell(image, _geometry(), 0, 0, palette, (130, 130, 130))

        self.assertEqual(observation.state, "palette")
        self.assertEqual(observation.palette_index, 0)
        self.assertEqual(observation.background_ratio, 0.0)

    def test_x_line_constraints_support_arbitrary_full_row_and_column(self):
        board = [
            [SimpleNamespace(state="background") for _ in range(15)]
            for _ in range(15)
        ]
        for column in range(15):
            board[6][column].state = "x_marker"
        for row in range(15):
            board[row][11].state = "x_marker"
        empty_rows, empty_columns, _, _ = _board_x_constraints(board)
        self.assertEqual(empty_rows, {6})
        self.assertEqual(empty_columns, {11})

    def test_x_markers_in_a_full_column_do_not_make_each_row_uncertain(self):
        board = [
            [SimpleNamespace(state="x_marker" if column == 3 else "background") for column in range(15)]
            for _ in range(15)
        ]
        empty_rows, empty_columns, row_counts, column_counts = _board_x_constraints(board)
        self.assertEqual(empty_rows, set())
        self.assertEqual(empty_columns, {3})
        self.assertEqual(row_counts, [1] * 15)
        self.assertEqual(column_counts[3], 15)

    def test_partial_x_line_is_rejected_as_uncertain(self):
        board = [
            [SimpleNamespace(state="background") for _ in range(15)]
            for _ in range(15)
        ]
        for column in range(14):
            board[4][column].state = "x_marker"
        with self.assertRaisesRegex(ValueError, "partial X markers are uncertain"):
            _board_x_constraints(board)

    def test_forced_empty_lines_produce_zero_solution_lines(self):
        rows = [[(0, 1)], [(0, 1)], []]
        columns = [[(0, 1)], [], [(0, 1)]]
        _, _, solution = _solve_with_candidate_recovery(
            None,
            None,
            None,
            [PaletteColor((10, 10), (230.0, 0.0, 0.0))],
            rows,
            columns,
            empty_rows={1},
            empty_columns={2},
        )
        self.assertEqual(solution[1], [0, 0, 0])
        self.assertEqual([line[2] for line in solution], [0, 0, 0])

    def test_full_board_verification_rejects_unexpected_empty_cell_paint(self):
        palette = [
            PaletteColor((10, 10), (230.0, 0.0, 0.0)),
            PaletteColor((20, 20), (0.0, 0.0, 230.0)),
        ]
        before = _empty_board()
        baseline = _observe_board(before, _geometry(), palette)
        after = before.copy()
        _fill_cell(after, 0, 0, (230, 0, 0))
        _fill_cell(after, 1, 1, (0, 0, 230))
        observed = _observe_board(after, _geometry(), palette)
        mismatches = _verify_board(observed, baseline, [[1, 0], [0, 0]])
        self.assertEqual([(item["row"], item["column"]) for item in mismatches], [(1, 1)])

    def test_zero_solution_allows_automatic_x_marker(self):
        palette = [
            PaletteColor((10, 10), (230.0, 0.0, 0.0)),
            PaletteColor((20, 20), (0.0, 0.0, 230.0)),
        ]
        before = _empty_board()
        baseline = _observe_board(before, _geometry(), palette)
        after = before.copy()
        _fill_x_marker(after, 0, 1)
        observed = _observe_board(
            after,
            _geometry(),
            palette,
            _estimate_background_rgb(before, _geometry()),
        )
        self.assertEqual(observed[0][1].state, "x_marker")
        self.assertEqual(_verify_board(observed, baseline, [[0, 0], [0, 0]]), [])

    def test_unpainted_nonzero_cell_rejects_automatic_x_marker(self):
        palette = [
            PaletteColor((10, 10), (230.0, 0.0, 0.0)),
            PaletteColor((20, 20), (0.0, 0.0, 230.0)),
        ]
        before = _empty_board()
        baseline = _observe_board(before, _geometry(), palette)
        after = before.copy()
        _fill_cell(after, 0, 0, (230, 0, 0))
        _fill_x_marker(after, 0, 1)
        observed = _observe_board(
            after,
            _geometry(),
            palette,
            _estimate_background_rgb(before, _geometry()),
        )
        mismatches = _verify_painted_segment(
            observed,
            baseline,
            [[1, 2], [0, 0]],
            PaintSegment(0, 0, 0, 0),
        )
        self.assertEqual([(item["row"], item["column"]) for item in mismatches], [(0, 1)])

    def test_already_painted_cell_must_keep_solution_color(self):
        palette = [
            PaletteColor((10, 10), (230.0, 0.0, 0.0)),
            PaletteColor((20, 20), (0.0, 0.0, 230.0)),
        ]
        before = _empty_board()
        baseline = _observe_board(before, _geometry(), palette)
        after = before.copy()
        _fill_cell(after, 0, 0, (0, 0, 230))
        _fill_cell(after, 0, 1, (0, 0, 230))
        observed = _observe_board(
            after,
            _geometry(),
            palette,
            _estimate_background_rgb(before, _geometry()),
        )
        mismatches = _verify_painted_segment(
            observed,
            baseline,
            [[1, 2], [0, 0]],
            PaintSegment(1, 0, 1, 1),
            already_painted={(0, 0)},
        )
        self.assertEqual([(item["row"], item["column"]) for item in mismatches], [(0, 0)])

    def test_unknown_solution_empty_cell_is_rejected(self):
        palette = [
            PaletteColor((10, 10), (230.0, 0.0, 0.0)),
            PaletteColor((20, 20), (0.0, 0.0, 230.0)),
        ]
        before = _empty_board()
        baseline = _observe_board(before, _geometry(), palette)
        after = before.copy()
        _fill_cell(after, 1, 1, (80, 20, 150))
        observed = _observe_board(after, _geometry(), palette)
        self.assertEqual(observed[1][1].state, "unknown")
        mismatches = _verify_board(observed, baseline, [[0, 0], [0, 0]])
        self.assertEqual([(item["row"], item["column"]) for item in mismatches], [(1, 1)])

    def test_target_cells_and_solution_hash_are_deterministic(self):
        solution = [[0 for _ in range(15)] for _ in range(15)]
        solution[0][0] = 1
        solution[14][14] = 2
        targets = _target_cells(solution, 2)
        self.assertEqual(targets, [(0, 0, 0), (1, 14, 14)])
        self.assertEqual(len(_solution_hash(solution)), 64)
        self.assertEqual(_solution_hash(solution), _solution_hash([line[:] for line in solution]))

    @unittest.skipUnless(SAMPLE_REPORT.is_file(), "random Dry Run report is not available")
    def test_random_dry_run_report_is_a_15x15_paint_input(self):
        report = json.loads(SAMPLE_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["phase"], "complete")
        self.assertEqual(len(report["solution"]), 15)
        self.assertTrue(all(len(line) == 15 for line in report["solution"]))
        self.assertGreater(sum(value > 0 for line in report["solution"] for value in line), 0)
        puzzle = ColorNonogramPuzzle.from_clues(report["rows"], report["columns"])
        puzzle.validate_color_totals()
        solved = solve_puzzle(puzzle, require_unique=True)
        self.assertEqual([list(line) for line in solved.grid], report["solution"])


if __name__ == "__main__":
    unittest.main()
