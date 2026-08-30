import pathlib
import unittest

import numpy as np
from PIL import Image

from color_nonogram_answer import (
    EMPTY_CELL,
    AnswerBoardError,
    PaintSegment,
    answer_grid_to_segments,
    detect_answer_grid_roi,
    extract_answer_board,
    map_answer_colors,
    remap_answer_grid,
    validate_answer_grid,
)


PROJECT_ROOT = pathlib.Path(__file__).parent.parent
ANSWER_IMAGE = PROJECT_ROOT / "install" / "resource" / "image" / "案例答案.png"
BEFORE_IMAGE = PROJECT_ROOT / "debug" / "color_nonogram_live" / "paint_before_20260824_220242_518709.png"
ANSWER_POPUP_IMAGE = PROJECT_ROOT / "debug" / "color_nonogram_live" / "paint_answer_20260824_224300_870896.png"


def _fixture_grid() -> tuple[tuple[int, ...], ...]:
    grid = [[EMPTY_CELL for _ in range(15)] for _ in range(15)]
    for column in range(2, 5):
        grid[1][column] = 0
    for column in range(7, 10):
        grid[1][column] = 1
    grid[6][5] = 2
    grid[6][6] = 2
    grid[6][7] = 2
    return tuple(tuple(row) for row in grid)


def _fixture_image(grid: tuple[tuple[int, ...], ...]) -> np.ndarray:
    colors = np.asarray(
        [
            (255, 210, 0),
            (0, 0, 0),
            (235, 30, 30),
        ],
        dtype=np.uint8,
    )
    image = np.full((15 * 12 + 4, 15 * 12 + 4, 3), 245, dtype=np.uint8)
    image[2:-2, 2:-2] = 250
    for row, line in enumerate(grid):
        for column, value in enumerate(line):
            if value != EMPTY_CELL:
                y = 2 + row * 12
                x = 2 + column * 12
                image[y + 2 : y + 10, x + 2 : x + 10] = colors[value]
    return image


class ColorNonogramAnswerTests(unittest.TestCase):
    def test_extracts_synthetic_board_with_palette_independent_labels(self):
        expected = _fixture_grid()
        board = extract_answer_board(_fixture_image(expected), grid_roi=(2, 2, 180, 180))
        self.assertEqual(board.grid, expected)
        self.assertEqual(board.color_count, 3)

    def test_extracts_local_answer_image_and_detects_square(self):
        if not ANSWER_IMAGE.is_file():
            self.skipTest("local answer image is not present")
        image = np.asarray(Image.open(ANSWER_IMAGE).convert("RGB"))
        roi = detect_answer_grid_roi(image)
        board = extract_answer_board(image, grid_roi=roi)
        self.assertEqual(len(board.grid), 15)
        self.assertEqual(board.color_count, 7)
        self.assertEqual(sum(value != EMPTY_CELL for row in board.grid for value in row), 102)

    def test_extracts_nested_answer_popup_inner_board(self):
        if not ANSWER_POPUP_IMAGE.is_file():
            self.skipTest("local answer popup image is not present")
        image = np.asarray(Image.open(ANSWER_POPUP_IMAGE).convert("RGB"))
        roi = detect_answer_grid_roi(image)
        board = extract_answer_board(image, grid_roi=roi)
        self.assertEqual(roi[:2], (645, 228))
        self.assertEqual(roi[2:], (629, 629))
        self.assertEqual(board.grid_roi, roi)
        self.assertEqual(board.color_count, 7)
        self.assertEqual(sum(value != EMPTY_CELL for row in board.grid for value in row), 102)
    def test_rejects_normal_puzzle_screenshot_as_answer_page(self):
        if not BEFORE_IMAGE.is_file():
            self.skipTest("local before screenshot is not present")
        image = np.asarray(Image.open(BEFORE_IMAGE).convert("RGB"))
        with self.assertRaises(AnswerBoardError):
            detect_answer_grid_roi(image)

    def test_maps_answer_colors_one_to_one(self):
        mapped = map_answer_colors(
            [(250, 20, 20), (10, 10, 10)],
            [type("Palette", (), {"rgb": (12, 12, 12)})(), type("Palette", (), {"rgb": (248, 18, 18)})()],
            maximum_distance=20,
        )
        self.assertEqual(mapped, (1, 0))
        self.assertEqual(
            remap_answer_grid(_fixture_grid(), mapped + (2,)),
            tuple(tuple(EMPTY_CELL if value == EMPTY_CELL else (mapped + (2,))[value] for value in row) for row in _fixture_grid()),
        )

    def test_generates_horizontal_paint_segments(self):
        segments = answer_grid_to_segments(_fixture_grid())
        self.assertEqual(
            segments,
            (
                PaintSegment(0, 1, 2, 4),
                PaintSegment(1, 1, 7, 9),
                PaintSegment(2, 6, 5, 7),
            ),
        )

    def test_rejects_wrong_size_and_out_of_range_color(self):
        with self.assertRaises(AnswerBoardError):
            validate_answer_grid(((EMPTY_CELL,),))
        invalid = [list(row) for row in _fixture_grid()]
        invalid[0][0] = 3
        with self.assertRaises(AnswerBoardError):
            validate_answer_grid(invalid, color_count=3)

    def test_rejects_blurred_board(self):
        grid = _fixture_grid()
        image = _fixture_image(grid).astype(np.float32)
        image[2 + 6 * 12 + 2 : 2 + 6 * 12 + 10, 2 + 5 * 12 + 2 : 2 + 5 * 12 + 10] = 128
        image[2 + 6 * 12 + 2 : 2 + 6 * 12 + 6, 2 + 5 * 12 + 2 : 2 + 5 * 12 + 10] = 0
        image[2 + 6 * 12 + 6 : 2 + 6 * 12 + 10, 2 + 5 * 12 + 2 : 2 + 5 * 12 + 10] = 255
        with self.assertRaises(AnswerBoardError):
            extract_answer_board(image.astype(np.uint8), grid_roi=(2, 2, 180, 180))

    def test_rejects_unmappable_and_ambiguous_colors(self):
        with self.assertRaises(AnswerBoardError):
            map_answer_colors([(255, 0, 0)], [(0, 0, 255)], maximum_distance=10)
        with self.assertRaises(AnswerBoardError):
            map_answer_colors([(100, 100, 100)], [(95, 95, 95), (105, 105, 105)], maximum_distance=20)


if __name__ == "__main__":
    unittest.main()
