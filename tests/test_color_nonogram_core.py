import json
import pathlib
import time
import unittest
from unittest import mock

from color_nonogram_core import (
    find_solutions,
    line_patterns,
    runs,
    solve_puzzle,
    SolutionSearchTimeout,
    validate_solution,
)
from color_nonogram_model import ColorNonogramPuzzle, ColorNonogramSolution


FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "color_nonogram_cases.json"


class ColorNonogramCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_fixture_solutions_are_valid(self):
        for case in self.cases:
            with self.subTest(case=case["name"]):
                puzzle = ColorNonogramPuzzle.from_clues(case["rows"], case["columns"])
                puzzle.validate_color_totals()
                solution = solve_puzzle(puzzle, require_unique=True)
                validate_solution(puzzle, solution)
                self.assertEqual([list(row) for row in solution.grid], case["expected"])

    def test_unique_search_reports_first_solution_and_unique_proof(self):
        puzzle = ColorNonogramPuzzle.from_clues([[(0, 1)]], [[(0, 1)]])
        events = []
        solution = solve_puzzle(
            puzzle,
            require_unique=True,
            progress_callback=lambda stage, elapsed, nodes, count: events.append(
                (stage, elapsed, nodes, count)
            ),
        )
        self.assertEqual(solution.grid, ((1,),))
        stages = [event[0] for event in events]
        self.assertIn("first_solution_found", stages)
        self.assertIn("unique_proven", stages)
        self.assertLessEqual(events[-1][2], 1)

    def test_root_propagation_is_reused_for_trivially_solved_puzzle(self):
        puzzle = ColorNonogramPuzzle.from_clues([[(0, 1)]], [[(0, 1)]])
        with mock.patch(
            "color_nonogram_core._propagate",
            wraps=__import__("color_nonogram_core")._propagate,
        ) as propagate:
            solve_puzzle(puzzle, require_unique=True)
        self.assertEqual(propagate.call_count, 1)
    def test_different_colors_can_touch(self):
        self.assertIn((1, 2), line_patterns(2, [(0, 1), (1, 1)]))

    def test_same_colors_require_an_empty_cell(self):
        patterns = line_patterns(3, [(0, 1), (0, 1)])
        self.assertIn((1, 0, 1), patterns)
        self.assertNotIn((1, 1, 0), patterns)

    def test_runs_are_zero_based_by_palette_color(self):
        solution = ColorNonogramSolution(((1, 1, 0), (0, 2, 2)))
        self.assertEqual(runs(solution), [(0, 0, 0, 1), (1, 1, 1, 2)])

    def test_invalid_clue_length_is_rejected(self):
        with self.assertRaises(ValueError):
            ColorNonogramPuzzle.from_clues([[(0, 0)]], [[(0, 1)]])

    def test_multiple_solutions_are_reported(self):
        puzzle = ColorNonogramPuzzle.from_clues([[(0, 1)], [(0, 1)]], [[(0, 1)], [(0, 1)]])
        self.assertEqual(len(find_solutions(puzzle, limit=2)), 2)
        with self.assertRaisesRegex(ValueError, "multiple solutions"):
            solve_puzzle(puzzle, require_unique=True)

    def test_pattern_generation_timeout_is_interruptible(self):
        with self.assertRaises(SolutionSearchTimeout):
            line_patterns(30, [(0, 1)] * 10, deadline=time.monotonic() - 1)

    def test_solution_search_timeout_is_interruptible(self):
        puzzle = ColorNonogramPuzzle.from_clues(
            [[(0, 1)], [(0, 1)]],
            [[(0, 1)], [(0, 1)]],
        )
        with self.assertRaises(SolutionSearchTimeout):
            solve_puzzle(puzzle, require_unique=True, maximum_seconds=0.000001)
    def test_no_solution_is_reported(self):
        puzzle = ColorNonogramPuzzle.from_clues([[(0, 2)]], [[(0, 1)], []])
        self.assertEqual(find_solutions(puzzle), ())
        with self.assertRaisesRegex(ValueError, "no solution"):
            solve_puzzle(puzzle)

    def test_mismatched_color_totals_are_rejected(self):
        puzzle = ColorNonogramPuzzle.from_clues([[(0, 1)]], [[(1, 1)]])
        with self.assertRaisesRegex(ValueError, "totals do not match"):
            puzzle.validate_color_totals()


if __name__ == "__main__":
    unittest.main()
