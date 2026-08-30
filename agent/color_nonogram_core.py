from __future__ import annotations

from collections.abc import Callable
from typing import Sequence
import time

from color_nonogram_model import Clue, ColorNonogramPuzzle, ColorNonogramSolution


class SolutionSearchTimeout(TimeoutError):
    pass


def line_minimum(clues: Sequence[Clue]) -> int:
    return sum(length for _, length in clues) + sum(
        clues[index][0] == clues[index + 1][0]
        for index in range(len(clues) - 1)
    )


def line_patterns(
    length: int,
    clues: Sequence[Clue],
    deadline: float | None = None,
) -> list[tuple[int, ...]]:
    if length < 0:
        raise ValueError("line length cannot be negative")
    if not clues:
        return [tuple(0 for _ in range(length))]
    if line_minimum(clues) > length:
        return []

    patterns: list[tuple[int, ...]] = []

    def place(index: int, start: int, cells: list[int]) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise SolutionSearchTimeout("solution search exceeded its deadline")
        if index == len(clues):
            patterns.append(tuple(cells + [0] * (length - len(cells))))
            return
        color, run_length = clues[index]
        tail = clues[index + 1 :]
        gap = 1 if tail and tail[0][0] == color else 0
        remaining = sum(item[1] for item in tail)
        remaining += sum(
            tail[offset][0] == tail[offset + 1][0]
            for offset in range(len(tail) - 1)
        )
        max_start = length - (run_length + gap + remaining)
        for position in range(start, max_start + 1):
            if deadline is not None and time.monotonic() >= deadline:
                raise SolutionSearchTimeout("solution search exceeded its deadline")
            prefix = cells + [0] * (position - len(cells)) + [color + 1] * run_length
            place(index + 1, position + run_length + gap, prefix + [0] * gap)

    place(0, 0, [])
    return patterns


def _propagate(
    puzzle: ColorNonogramPuzzle,
    current: list[list[int]],
    row_options: list[list[tuple[int, ...]]],
    column_options: list[list[tuple[int, ...]]],
) -> tuple[list[list[int]], list[list[tuple[int, ...]]], list[list[tuple[int, ...]]]] | None:
    current = [line[:] for line in current]
    while True:
        changed = False
        row_options = []
        for row, patterns in enumerate(row_options or [line_patterns(puzzle.width, clues) for clues in puzzle.rows]):
            options = [
                pattern
                for pattern in patterns
                if all(current[row][column] in (-1, pattern[column]) for column in range(puzzle.width))
            ]
            if not options:
                return None
            row_options.append(options)
            for column in range(puzzle.width):
                values = {pattern[column] for pattern in options}
                if len(values) == 1 and current[row][column] != next(iter(values)):
                    current[row][column] = next(iter(values))
                    changed = True

        column_options = []
        for column, patterns in enumerate(column_options or [line_patterns(puzzle.height, clues) for clues in puzzle.columns]):
            options = [
                pattern
                for pattern in patterns
                if all(current[row][column] in (-1, pattern[row]) for row in range(puzzle.height))
            ]
            if not options:
                return None
            column_options.append(options)
            for row in range(puzzle.height):
                values = {pattern[row] for pattern in options}
                if len(values) == 1 and current[row][column] != next(iter(values)):
                    current[row][column] = next(iter(values))
                    changed = True

        if not changed:
            return current, row_options, column_options


def find_solutions(
    puzzle: ColorNonogramPuzzle,
    limit: int = 2,
    maximum_seconds: float | None = None,
    progress_callback: Callable[[str, float, int, int], None] | None = None,
) -> tuple[ColorNonogramSolution, ...]:
    puzzle.validate()
    if limit <= 0:
        raise ValueError("solution limit must be positive")
    if maximum_seconds is not None and maximum_seconds <= 0:
        raise ValueError("maximum_seconds must be positive when provided")
    started_at = time.monotonic()
    deadline = started_at + maximum_seconds if maximum_seconds is not None else None
    last_progress_at = started_at
    search_nodes = 0
    results: list[ColorNonogramSolution] = []

    def report(stage: str, *, force: bool = False) -> None:
        nonlocal last_progress_at
        elapsed = time.monotonic() - started_at
        if not force and elapsed - last_progress_at < 1.0:
            return
        last_progress_at = time.monotonic()
        solution_count = len(results)
        if progress_callback is not None:
            progress_callback(stage, elapsed, search_nodes, solution_count)
        else:
            print(
                f"color nonogram solve phase={stage} elapsed={elapsed:.2f}s "
                f"nodes={search_nodes} solutions={solution_count}",
                flush=True,
            )

    row_patterns = [line_patterns(puzzle.width, clues, deadline) for clues in puzzle.rows]
    column_patterns = [line_patterns(puzzle.height, clues, deadline) for clues in puzzle.columns]
    if any(not options for options in [*row_patterns, *column_patterns]):
        report("no_line_patterns", force=True)
        return ()

    report("patterns_ready", force=True)
    propagated_root = _propagate(
        puzzle,
        [[-1] * puzzle.width for _ in range(puzzle.height)],
        row_patterns,
        column_patterns,
    )
    if propagated_root is None:
        report("contradiction_after_propagation", force=True)
        return ()

    def search(
        current: list[list[int]],
        available_rows: list[list[tuple[int, ...]]],
        available_columns: list[list[tuple[int, ...]]],
        *,
        already_propagated: bool = False,
    ) -> None:
        nonlocal search_nodes
        search_nodes += 1
        if deadline is not None and time.monotonic() >= deadline:
            phase = "uniqueness" if results else "first_solution"
            report(f"{phase}_timeout", force=True)
            raise SolutionSearchTimeout(
                f"solution search exceeded timeout={maximum_seconds:.3f}s phase={phase} "
                f"nodes={search_nodes} solutions={len(results)}"
            )
        if len(results) >= limit:
            return
        if already_propagated:
            propagated = (current, available_rows, available_columns)
        else:
            propagated = _propagate(puzzle, current, available_rows, available_columns)
        if propagated is None:
            report("branch_contradiction")
            return
        current, rows, columns = propagated
        if all(len(options) == 1 for options in rows) and all(len(options) == 1 for options in columns):
            solution = ColorNonogramSolution(tuple(tuple(row) for row in current))
            validate_solution(puzzle, solution)
            if solution not in results:
                results.append(solution)
                report("first_solution_found" if len(results) == 1 else "second_solution_found", force=True)
            return

        report("uniqueness_search" if results else "first_solution_search")
        choices = [(len(options), "row", index, options) for index, options in enumerate(rows) if len(options) > 1]
        choices.extend((len(options), "column", index, options) for index, options in enumerate(columns) if len(options) > 1)
        _, axis, index, options = min(choices, key=lambda item: item[0])
        for pattern in options:
            branch = [line[:] for line in current]
            if axis == "row":
                branch[index] = list(pattern)
            else:
                for row, value in enumerate(pattern):
                    branch[row][index] = value
            search(branch, rows, columns)
            if len(results) >= limit:
                return

    search(*propagated_root, already_propagated=True)
    report("unique_proven" if len(results) == 1 else "multiple_solutions_found" if len(results) > 1 else "no_solution", force=True)
    return tuple(results)


def solve_puzzle(
    puzzle: ColorNonogramPuzzle,
    require_unique: bool = False,
    maximum_seconds: float | None = None,
    progress_callback: Callable[[str, float, int, int], None] | None = None,
) -> ColorNonogramSolution:
    solutions = find_solutions(
        puzzle,
        limit=2 if require_unique else 1,
        maximum_seconds=maximum_seconds,
        progress_callback=progress_callback,
    )
    if not solutions:
        raise ValueError("colored puzzle has no solution")
    if require_unique and len(solutions) > 1:
        raise ValueError("colored puzzle has multiple solutions")
    return solutions[0]

def validate_solution(puzzle: ColorNonogramPuzzle, solution: ColorNonogramSolution) -> None:
    if solution.height != puzzle.height or solution.width != puzzle.width:
        raise ValueError("solution dimensions do not match puzzle")
    for row, clues in zip(solution.grid, puzzle.rows):
        if tuple(row) not in line_patterns(puzzle.width, clues):
            raise ValueError("solution violates row clues")
    for column, clues in enumerate(puzzle.columns):
        values = tuple(solution.grid[row][column] for row in range(puzzle.height))
        if values not in line_patterns(puzzle.height, clues):
            raise ValueError("solution violates column clues")


def runs(solution: ColorNonogramSolution) -> list[tuple[int, int, int, int]]:
    result = []
    for row, line in enumerate(solution.grid):
        start = 0
        while start < len(line):
            color = line[start]
            if color == 0:
                start += 1
                continue
            end = start
            while end + 1 < len(line) and line[end + 1] == color:
                end += 1
            result.append((color - 1, row, start, end))
            start = end + 1
    return result
