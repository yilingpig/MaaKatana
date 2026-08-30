import json
import pathlib
import unittest

import numpy as np
from PIL import Image

from color_nonogram_core import solve_puzzle
from color_nonogram_truth import clues_from_grid, extract_board_truth, iter_clue_truth
from color_nonogram_vision import GridCalibration


PROJECT_ROOT = pathlib.Path(__file__).parent.parent
CALIBRATIONS_PATH = PROJECT_ROOT / "tests" / "fixtures" / "color_nonogram_screenshots.json"
IMAGE_ROOT = PROJECT_ROOT / "assets" / "resource" / "image"


class ColorNonogramTruthTests(unittest.TestCase):
    def test_clues_are_derived_from_colored_grid(self):
        rows, columns = clues_from_grid(
            (
                (1, 1, 0),
                (0, 2, 2),
                (1, 0, 2),
            )
        )
        self.assertEqual(rows, (((0, 2),), ((1, 2),), ((0, 1), (1, 1))))
        self.assertEqual(columns, (((0, 1), (0, 1)), ((0, 1), (1, 1)), ((1, 2),)))

    def test_only_solid_completed_screenshots_produce_truth(self):
        data = json.loads(CALIBRATIONS_PATH.read_text(encoding="utf-8"))
        completed = {}
        for item in data:
            calibration = GridCalibration(
                source=item["source"],
                grid=tuple(item["grid"]),
                clue_left=float(item["clue_left"]),
                clue_top=float(item["clue_top"]),
                size=int(item.get("size", 15)),
            )
            image = np.asarray(Image.open(IMAGE_ROOT / calibration.source).convert("RGB"))
            truth = extract_board_truth(image, calibration)
            if truth.completed:
                solution = solve_puzzle(truth.puzzle(), require_unique=True)
                self.assertEqual(solution.grid, truth.grid)
                completed[truth.source] = len(iter_clue_truth(truth))
        self.assertEqual(
            completed,
            {
                "彩色拼图解题12.png": 105,
                "彩色拼图解题13.png": 174,
                "彩色拼图解题14.png": 71,
                "彩色拼图解题15.png": 110,
            },
        )


if __name__ == "__main__":
    unittest.main()
