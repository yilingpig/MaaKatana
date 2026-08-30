import unittest
from unittest.mock import patch

import color_nonogram_disambiguation as disambiguation

from color_nonogram_disambiguation import (
    ClueCandidate,
    ClueObservation,
    LineCombinationTimeout,
    LineCombinationBudgetExceeded,
    build_line_options,
    disambiguate_puzzle,
)


class ColorNonogramDisambiguationTests(unittest.TestCase):
    def test_missing_line_observations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing observations"):
            build_line_options([], "row", 1, 3, 1)

    def test_explicit_empty_line_is_allowed(self):
        options = build_line_options([], "row", 1, 3, 1, empty_lines={0})
        self.assertEqual(options[0][0].clues, ())

    def test_line_combination_budget_fails_before_full_enumeration(self):
        observations = [
            ClueObservation(
                "row", 0, slot,
                tuple(ClueCandidate(value, value % 2, 1.0) for value in (1, 2, 3)),
            )
            for slot in range(10)
        ]
        with self.assertRaisesRegex(
            LineCombinationBudgetExceeded,
            "row line=0 exceeded candidate combination budget=100",
        ):
            build_line_options(
                observations,
                "row",
                1,
                30,
                2,
                maximum_combinations=100,
            )

    def test_line_generation_timeout_fails_before_option_sorting(self):
        observations = [
            ClueObservation(
                "row", 0, slot,
                tuple(ClueCandidate(value, value % 2, 1.0) for value in (1, 2, 3)),
            )
            for slot in range(8)
        ]
        with self.assertRaisesRegex(LineCombinationTimeout, "row line=0"):
            build_line_options(
                observations,
                "row",
                1,
                24,
                2,
                maximum_combinations=1_000_000,
                maximum_seconds=0.000001,
            )
    def test_low_combination_budget_preserves_existing_line_options(self):
        observations = [
            ClueObservation(
                "row", 0, 0,
                (ClueCandidate(1, 0, 1.0), ClueCandidate(2, 0, 0.5)),
            ),
            ClueObservation(
                "row", 0, 1,
                (ClueCandidate(1, 0, 1.0),),
            ),
        ]
        default_options = build_line_options(observations, "row", 1, 3, 1)
        bounded_options = build_line_options(
            observations, "row", 1, 3, 1, maximum_combinations=2,
        )
        self.assertEqual(bounded_options, default_options)

    def test_disambiguation_propagates_line_combination_budget(self):
        rows = [[
            ClueObservation(
                "row", 0, slot,
                tuple(ClueCandidate(value, value % 2, 1.0) for value in (1, 2, 3)),
            )
            for slot in range(6)
        ]]
        columns = [
            [ClueObservation("column", column, 0, (ClueCandidate(1, 0, 1.0),))]
            for column in range(5)
        ]
        with self.assertRaises(LineCombinationBudgetExceeded):
            disambiguate_puzzle(
                rows,
                columns,
                5,
                1,
                2,
                maximum_line_combinations=100,
            )
    def test_line_options_enforce_same_color_gap(self):
        observations = [
            ClueObservation(
                "row", 0, 0,
                (ClueCandidate(1, 0, 1.0), ClueCandidate(2, 0, 0.5)),
            ),
            ClueObservation(
                "row", 0, 1,
                (ClueCandidate(1, 0, 1.0),),
            ),
        ]
        options = build_line_options(observations, "row", 1, 3, 1)
        self.assertEqual([option.clues for option in options[0]], [((0, 1), (0, 1))])

    def test_ambiguous_candidate_puzzles_are_rejected(self):
        rows = [[
            ClueObservation(
                "row", 0, 0,
                (ClueCandidate(1, 0, 1.0), ClueCandidate(1, 1, 0.9)),
            ),
            ClueObservation(
                "row", 0, 1,
                (ClueCandidate(1, 1, 1.0), ClueCandidate(1, 0, 0.9)),
            ),
        ]]
        columns = [
            [ClueObservation("column", 0, 0, (ClueCandidate(1, 0, 1.0), ClueCandidate(1, 1, 0.9)))],
            [ClueObservation("column", 1, 0, (ClueCandidate(1, 1, 1.0), ClueCandidate(1, 0, 0.9)))],
        ]
        result = disambiguate_puzzle(rows, columns, 2, 1, 2)
        self.assertEqual(result.status, "multiple")
        self.assertIsNone(result.solution)

    def test_candidate_timeout_is_recorded_and_search_continues(self):
        rows = [[ClueObservation("row", 0, 0, (ClueCandidate(1, 0, 1.0),))]]
        columns = [[ClueObservation("column", 0, 0, (ClueCandidate(1, 0, 1.0),))]]
        with patch.object(
            disambiguation,
            "solve_puzzle",
            side_effect=disambiguation.SolutionSearchTimeout("candidate timeout"),
        ):
            result = disambiguation.disambiguate_puzzle(
                rows,
                columns,
                1,
                1,
                1,
                maximum_seconds=1.0,
                maximum_candidate_seconds=0.5,
            )
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.candidate_timeouts, 1)
        self.assertEqual(result.row_option_counts, (1,))
        self.assertEqual(result.column_option_counts, (1,))
        self.assertEqual(result.status, "timeout")
        self.assertIsNone(result.solution)
        self.assertIn("candidate_timeouts=1", result.reason)

    def test_solution_with_unresolved_candidate_timeout_is_not_unique(self):
        rows = [[
            ClueObservation(
                "row", 0, 0,
                (ClueCandidate(1, 0, 1.0), ClueCandidate(1, 1, 0.9)),
            ),
        ]]
        columns = [[ClueObservation(
            "column", 0, 0,
            (ClueCandidate(1, 0, 1.0), ClueCandidate(1, 1, 0.9)),
        )]]
        solution = disambiguation.ColorNonogramSolution(((1,),))
        calls = 0

        def solve_candidate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return solution
            raise disambiguation.SolutionSearchTimeout("candidate timeout")

        with patch.object(disambiguation, "solve_puzzle", side_effect=solve_candidate):
            result = disambiguation.disambiguate_puzzle(
                rows, columns, 1, 1, 2, maximum_candidate_seconds=0.5
            )

        self.assertGreaterEqual(calls, 2)
        self.assertEqual(result.status, "timeout")
        self.assertIsNotNone(result.solution)
        self.assertEqual(result.solution.grid, ((1,),))
        self.assertIn("fully resolved", result.reason)

    def test_true_no_solution_remains_no_solution(self):
        rows = [[ClueObservation("row", 0, 0, (ClueCandidate(1, 0, 1.0),))]]
        columns = [[ClueObservation("column", 0, 0, (ClueCandidate(1, 0, 1.0),))]]
        with patch.object(
            disambiguation,
            "solve_puzzle",
            side_effect=ValueError("no solution"),
        ):
            result = disambiguation.disambiguate_puzzle(rows, columns, 1, 1, 1)

        self.assertEqual(result.status, "no_solution")
        self.assertIsNone(result.solution)
        self.assertEqual(result.candidate_timeouts, 0)

    def test_default_candidate_budget_is_passed_to_solver(self):
        rows = [[ClueObservation("row", 0, 0, (ClueCandidate(1, 0, 1.0),))]]
        columns = [[ClueObservation("column", 0, 0, (ClueCandidate(1, 0, 1.0),))]]
        with patch.object(
            disambiguation,
            "solve_puzzle",
            return_value=disambiguation.ColorNonogramSolution(((1,),)),
        ) as solve_mock:
            result = disambiguation.disambiguate_puzzle(rows, columns, 1, 1, 1)
        self.assertEqual(result.status, "unique")
        self.assertEqual(solve_mock.call_args.kwargs["maximum_seconds"], 3.0)
        self.assertNotIn("global_timeout=True", result.reason)
    def test_unique_candidate_puzzle_returns_solution(self):
        rows = [[
            ClueObservation("row", 0, 0, (ClueCandidate(1, 1, 1.0),)),
            ClueObservation("row", 0, 1, (ClueCandidate(1, 0, 1.0),)),
        ]]
        columns = [
            [ClueObservation("column", 0, 0, (ClueCandidate(1, 0, 1.0),))],
            [ClueObservation("column", 1, 0, (ClueCandidate(1, 1, 1.0),))],
        ]
        result = disambiguate_puzzle(rows, columns, 2, 1, 2)
        self.assertEqual(result.status, "unique")
        self.assertIsNotNone(result.solution)
        self.assertEqual(result.solution.grid, ((1, 2),))


if __name__ == "__main__":
    unittest.main()
