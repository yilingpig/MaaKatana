from __future__ import annotations

from dataclasses import dataclass
import heapq
import time
from typing import Iterable, Sequence

from color_nonogram_core import SolutionSearchTimeout, solve_puzzle
from color_nonogram_model import ColorNonogramPuzzle, ColorNonogramSolution, Line

DEFAULT_MAXIMUM_CANDIDATE_SECONDS = 3.0


@dataclass(frozen=True)
class ClueCandidate:
    value: int
    color: int
    score: float


@dataclass(frozen=True)
class ClueObservation:
    axis: str
    line: int
    slot: int
    candidates: tuple[ClueCandidate, ...]


@dataclass(frozen=True)
class LineOption:
    clues: Line
    score: float
    color_totals: tuple[int, ...]


class LineCombinationTimeout(RuntimeError):
    def __init__(self, axis: str, line: int, timeout: float):
        self.axis = axis
        self.line = line
        self.timeout = timeout
        super().__init__(f"{axis} line={line} exceeded combination generation timeout={timeout:.3f}s")


class LineCombinationBudgetExceeded(RuntimeError):
    def __init__(self, axis: str, line: int, budget: int):
        self.axis = axis
        self.line = line
        self.budget = budget
        super().__init__(
            f"{axis} line={line} exceeded candidate combination budget={budget}"
        )


@dataclass(frozen=True)
class DisambiguationResult:
    status: str
    puzzle: ColorNonogramPuzzle | None
    solution: ColorNonogramSolution | None
    attempts: int
    candidate_timeouts: int
    row_option_counts: tuple[int, ...]
    column_option_counts: tuple[int, ...]
    reason: str


def build_line_options(
    observations: Iterable[ClueObservation],
    axis: str,
    line_count: int,
    line_length: int,
    color_count: int,
    top_k: int = 3,
    maximum_options: int = 4096,
    empty_lines: set[int] | None = None,
    maximum_combinations: int = 100_000,
    maximum_seconds: float | None = 3.0,
    _deadline: float | None = None,
) -> tuple[tuple[LineOption, ...], ...]:
    if axis not in {"row", "column"}:
        raise ValueError("axis must be row or column")
    if line_count <= 0 or line_length <= 0 or color_count <= 0:
        raise ValueError("line dimensions and color count must be positive")
    if top_k <= 0 or maximum_options <= 0 or maximum_combinations <= 0:
        raise ValueError("top_k, maximum_options, and maximum_combinations must be positive")
    if maximum_seconds is not None and maximum_seconds <= 0:
        raise ValueError("maximum_seconds must be positive when provided")
    empty_lines = set(empty_lines or ())
    if any(not 0 <= line < line_count for line in empty_lines):
        raise ValueError("empty line index is outside the puzzle")
    grouped: list[dict[int, ClueObservation]] = [dict() for _ in range(line_count)]
    for observation in observations:
        if observation.axis != axis:
            raise ValueError(f"expected {axis} observations")
        if not 0 <= observation.line < line_count or observation.slot < 0:
            raise ValueError("observation position is outside the puzzle")
        if observation.slot in grouped[observation.line]:
            raise ValueError("duplicate observation slot")
        if not observation.candidates:
            raise ValueError("observation must contain candidates")
        grouped[observation.line][observation.slot] = observation
    result = []
    for line_index, line_observations in enumerate(grouped):
        if not line_observations:
            if line_index not in empty_lines:
                raise ValueError(f"missing observations for {axis} line={line_index}")
            result.append((LineOption((), 0.0, (0,) * color_count),))
            continue
        highest_slot = max(line_observations)
        if set(line_observations) != set(range(highest_slot + 1)):
            raise ValueError("missing clue slots")
        ordered = [line_observations[slot] for slot in range(highest_slot, -1, -1)]
        options = _line_options_for_observations(
            ordered, line_length, color_count, top_k, maximum_options,
            maximum_combinations, axis, line_index,
            _deadline, maximum_seconds,
        )
        if not options:
            raise ValueError("line has no length-compatible clue options")
        result.append(tuple(options))
    return tuple(result)


def disambiguate_puzzle(
    rows: Sequence[Sequence[ClueObservation]],
    columns: Sequence[Sequence[ClueObservation]],
    width: int,
    height: int,
    color_count: int,
    top_k: int = 3,
    maximum_line_options: int = 64,
    maximum_attempts: int = 10000,
    require_unique: bool = True,
    maximum_seconds: float | None = None,
    empty_rows: set[int] | None = None,
    empty_columns: set[int] | None = None,
    maximum_line_combinations: int = 100_000,
    maximum_line_generation_seconds: float | None = 3.0,
    maximum_candidate_seconds: float = DEFAULT_MAXIMUM_CANDIDATE_SECONDS,
) -> DisambiguationResult:
    if len(rows) != height or len(columns) != width:
        raise ValueError("row and column counts do not match puzzle dimensions")
    if maximum_line_combinations <= 0:
        raise ValueError("maximum_line_combinations must be positive")
    if maximum_line_generation_seconds is not None and maximum_line_generation_seconds <= 0:
        raise ValueError("maximum_line_generation_seconds must be positive when provided")
    if maximum_candidate_seconds <= 0:
        raise ValueError("maximum_candidate_seconds must be positive")
    started_at = time.monotonic()
    deadline = started_at + maximum_seconds if maximum_seconds is not None else None
    row_options = build_line_options(
        [item for line in rows for item in line], "row", height, width,
        color_count, top_k, maximum_line_options, empty_rows,
        maximum_line_combinations,
        maximum_line_generation_seconds, deadline,
    )
    column_options = build_line_options(
        [item for line in columns for item in line], "column", width, height,
        color_count, top_k, maximum_line_options, empty_columns,
        maximum_line_combinations,
        maximum_line_generation_seconds, deadline,
    )
    row_indices = (0,) * height
    column_indices = (0,) * width
    queue = [(-_assignment_score(row_options, row_indices) - _assignment_score(column_options, column_indices), row_indices, column_indices)]
    visited = {(row_indices, column_indices)}
    attempts = 0
    candidate_timeouts = 0
    candidate_search_incomplete = False
    first_puzzle = None
    first_solution = None
    while queue and attempts < maximum_attempts:
        if deadline is not None and time.monotonic() >= deadline:
            candidate_search_incomplete = True
            break
        _, current_rows, current_columns = heapq.heappop(queue)
        row_assignment = tuple(options[index] for options, index in zip(row_options, current_rows))
        column_assignment = tuple(options[index] for options, index in zip(column_options, current_columns))
        if _assignment_totals(row_assignment) == _assignment_totals(column_assignment):
            attempts += 1
            puzzle = ColorNonogramPuzzle(
                rows=tuple(option.clues for option in row_assignment),
                columns=tuple(option.clues for option in column_assignment),
            )
            try:
                if deadline is not None and time.monotonic() >= deadline:
                    candidate_search_incomplete = True
                    break
                remaining_seconds = (
                    maximum_candidate_seconds
                    if deadline is None
                    else min(maximum_candidate_seconds, max(0.001, deadline - time.monotonic()))
                )
                solution = solve_puzzle(
                    puzzle,
                    require_unique=require_unique,
                    maximum_seconds=remaining_seconds,
                )
            except SolutionSearchTimeout:
                candidate_timeouts += 1
                candidate_search_incomplete = True
                solution = None
            except ValueError:
                solution = None
            if solution is not None:
                if first_puzzle is None:
                    first_puzzle, first_solution = puzzle, solution
                elif puzzle != first_puzzle or solution != first_solution:
                    return DisambiguationResult(
                        "multiple", None, None, attempts, candidate_timeouts,
                        tuple(len(item) for item in row_options),
                        tuple(len(item) for item in column_options),
                        (
                            "multiple candidate puzzles have valid solutions; "
                            f"attempts={attempts}; candidate_timeouts={candidate_timeouts}; "
                            f"row_option_counts={tuple(len(item) for item in row_options)}; "
                            f"column_option_counts={tuple(len(item) for item in column_options)}"
                        ),
                    )
        for axis, options, current in (
            ("row", row_options, current_rows),
            ("column", column_options, current_columns),
        ):
            for line_index in range(len(options)):
                next_index = current[line_index] + 1
                if next_index >= len(options[line_index]):
                    continue
                if axis == "row":
                    next_rows = current_rows[:line_index] + (next_index,) + current_rows[line_index + 1:]
                    next_columns = current_columns
                else:
                    next_rows = current_rows
                    next_columns = current_columns[:line_index] + (next_index,) + current_columns[line_index + 1:]
                state = (next_rows, next_columns)
                if state in visited:
                    continue
                visited.add(state)
                score = _assignment_score(row_options, next_rows) + _assignment_score(column_options, next_columns)
                heapq.heappush(queue, (-score, next_rows, next_columns))
    if candidate_search_incomplete:
        reason = (
            "candidate solution search timed out before the candidate set was "
            "fully resolved"
        )
        status = "timeout"
    elif first_puzzle is None:
        reason = "no candidate puzzle has a valid solution"
        status = "no_solution" if not queue else "inconclusive"
    elif queue:
        reason = "maximum puzzle attempts reached before candidate search was exhausted"
        status = "inconclusive"
    else:
        reason = "one candidate puzzle has a valid solution"
        status = "unique"
    reason = (
        f"{reason}; attempts={attempts}; candidate_timeouts={candidate_timeouts}; "
        f"row_option_counts={tuple(len(item) for item in row_options)}; "
        f"column_option_counts={tuple(len(item) for item in column_options)}"
    )
    return DisambiguationResult(
        status, first_puzzle, first_solution, attempts, candidate_timeouts,
        tuple(len(item) for item in row_options),
        tuple(len(item) for item in column_options),
        reason,
    )


def _assignment_score(options: Sequence[Sequence[LineOption]], indices: tuple[int, ...]) -> float:
    return sum(options[line_index][option_index].score for line_index, option_index in enumerate(indices))


def _assignment_totals(options: Sequence[LineOption]) -> tuple[int, ...]:
    if not options:
        return ()
    totals = [0] * len(options[0].color_totals)
    for option in options:
        for color, value in enumerate(option.color_totals):
            totals[color] += value
    return tuple(totals)

def _line_options_for_observations(
    observations: Sequence[ClueObservation],
    line_length: int,
    color_count: int,
    top_k: int,
    maximum_options: int,
    maximum_combinations: int,
    axis: str,
    line: int,
    deadline: float | None,
    maximum_seconds: float | None,
) -> list[LineOption]:
    candidates = [observation.candidates[:top_k] for observation in observations]
    options: dict[Line, float] = {}
    line_deadline = time.monotonic() + maximum_seconds if maximum_seconds is not None else None
    if deadline is not None:
        line_deadline = deadline if line_deadline is None else min(line_deadline, deadline)
    for combination_index, combination in enumerate(_candidate_combinations(candidates), start=1):
        if combination_index > maximum_combinations:
            raise LineCombinationBudgetExceeded(axis, line, maximum_combinations)
        if line_deadline is not None and time.monotonic() >= line_deadline:
            timeout = maximum_seconds if maximum_seconds is not None else 0.0
            raise LineCombinationTimeout(axis, line, timeout)
        clues = tuple((candidate.color, candidate.value) for candidate in combination)
        if _minimum_line_length(clues) > line_length:
            continue
        score = sum(candidate.score for candidate in combination)
        options[clues] = max(score, options.get(clues, float("-inf")))
    ranked = sorted(options.items(), key=lambda item: item[1], reverse=True)
    return [LineOption(clues, score, _color_totals(clues, color_count)) for clues, score in ranked[:maximum_options]]


def _candidate_combinations(candidates: Sequence[Sequence[ClueCandidate]]) -> Iterable[tuple[ClueCandidate, ...]]:
    if not candidates:
        yield ()
        return
    for candidate in candidates[0]:
        for suffix in _candidate_combinations(candidates[1:]):
            yield (candidate, *suffix)


def _assignments_by_total(
    line_options: Sequence[Sequence[LineOption]],
    color_count: int,
    maximum_per_total: int,
) -> dict[tuple[int, ...], list[tuple[LineOption, ...]]]:
    states: dict[tuple[int, ...], list[tuple[LineOption, ...]]] = {(0,) * color_count: [()]}
    for options in line_options:
        next_states: dict[tuple[int, ...], list[tuple[LineOption, ...]]] = {}
        for total, assignments in states.items():
            for option in options:
                updated = tuple(left + right for left, right in zip(total, option.color_totals))
                bucket = next_states.setdefault(updated, [])
                for assignment in assignments:
                    if len(bucket) >= maximum_per_total:
                        break
                    bucket.append((*assignment, option))
        states = next_states
        if not states:
            break
    return states


def _color_totals(clues: Line, color_count: int) -> tuple[int, ...]:
    totals = [0] * color_count
    for color, value in clues:
        if not 0 <= color < color_count:
            raise ValueError("clue color is outside the configured palette")
        totals[color] += value
    return tuple(totals)


def _minimum_line_length(clues: Line) -> int:
    length = sum(value for _, value in clues)
    for previous, current in zip(clues, clues[1:]):
        if previous[0] == current[0]:
            length += 1
    return length
