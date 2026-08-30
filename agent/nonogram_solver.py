import json
import re
import time
from dataclasses import dataclass

import numpy as np

SUPPORTED_GRID_SIZES = (10, 15, 20)

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JOCR, JRecognitionType


@dataclass(frozen=True)
class GridGeometry:
    columns: int
    rows: int
    x_lines: tuple[float, ...]
    y_lines: tuple[float, ...]
    score: float
    clue_left: float | None = None
    clue_top: float | None = None

    @property
    def cell_width(self) -> float:
        return (self.x_lines[-1] - self.x_lines[0]) / self.columns

    @property
    def cell_height(self) -> float:
        return (self.y_lines[-1] - self.y_lines[0]) / self.rows

    def center(self, row: int, column: int) -> tuple[int, int]:
        x = (self.x_lines[column] + self.x_lines[column + 1]) / 2
        y = (self.y_lines[row] + self.y_lines[row + 1]) / 2
        return round(x), round(y)


def _load_param(raw: str) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _line_patterns(length: int, clues: list[int]) -> list[int]:
    if not clues:
        return [0]
    if sum(clues) + len(clues) - 1 > length:
        return []
    patterns: list[int] = []

    def place(clue_index: int, next_start: int, mask: int) -> None:
        if clue_index == len(clues):
            patterns.append(mask)
            return
        remaining = sum(clues[clue_index:]) + len(clues) - clue_index - 1
        latest_start = length - remaining
        run_length = clues[clue_index]
        run_mask = (1 << run_length) - 1
        for start in range(next_start, latest_start + 1):
            next_mask = mask | (run_mask << start)
            place(clue_index + 1, start + run_length + 1, next_mask)

    place(0, 0, 0)
    return patterns


def _runs(line: list[int] | tuple[int, ...]) -> list[int]:
    output = []
    length = 0
    for value in [*line, 0]:
        if value:
            length += 1
        elif length:
            output.append(length)
            length = 0
    return output


def _propagate(
    grid: list[list[int]],
    row_options: list[list[int]],
    column_options: list[list[int]],
) -> bool:
    row_count = len(grid)
    column_count = len(grid[0]) if grid else 0
    row_mask = (1 << column_count) - 1
    column_mask = (1 << row_count) - 1
    while True:
        changed = False
        for row in range(row_count):
            known_ones = 0
            known_zeros = 0
            for column in range(column_count):
                if grid[row][column] == 1:
                    known_ones |= 1 << column
                elif grid[row][column] == 0:
                    known_zeros |= 1 << column
            options = [
                pattern
                for pattern in row_options[row]
                if pattern & known_ones == known_ones and not pattern & known_zeros
            ]
            if not options:
                return False
            row_options[row] = options
            common_ones = row_mask
            possible_ones = 0
            for pattern in options:
                common_ones &= pattern
                possible_ones |= pattern
            common_zeros = row_mask ^ possible_ones
            for column in range(column_count):
                bit = 1 << column
                if grid[row][column] == -1 and common_ones & bit:
                    grid[row][column] = 1
                    changed = True
                elif grid[row][column] == -1 and common_zeros & bit:
                    grid[row][column] = 0
                    changed = True
        for column in range(column_count):
            known_ones = 0
            known_zeros = 0
            for row in range(row_count):
                if grid[row][column] == 1:
                    known_ones |= 1 << row
                elif grid[row][column] == 0:
                    known_zeros |= 1 << row
            options = [
                pattern
                for pattern in column_options[column]
                if pattern & known_ones == known_ones and not pattern & known_zeros
            ]
            if not options:
                return False
            column_options[column] = options
            common_ones = column_mask
            possible_ones = 0
            for pattern in options:
                common_ones &= pattern
                possible_ones |= pattern
            common_zeros = column_mask ^ possible_ones
            for row in range(row_count):
                bit = 1 << row
                if grid[row][column] == -1 and common_ones & bit:
                    grid[row][column] = 1
                    changed = True
                elif grid[row][column] == -1 and common_zeros & bit:
                    grid[row][column] = 0
                    changed = True
        if not changed:
            return True


def solve_nonogram(
    row_clues: list[list[int]],
    column_clues: list[list[int]],
    max_branches: int = 50000,
) -> list[list[int]]:
    row_count = len(row_clues)
    column_count = len(column_clues)
    if row_count != column_count or row_count not in SUPPORTED_GRID_SIZES:
        raise ValueError("only square 10x10, 15x15, and 20x20 puzzles are supported")
    for clues in row_clues:
        if any(value <= 0 for value in clues):
            raise ValueError(f"invalid row clue line: {clues}")
        if sum(clues) + max(0, len(clues) - 1) > column_count:
            raise ValueError(f"row clue line exceeds board width: {clues}")
    for clues in column_clues:
        if any(value <= 0 for value in clues):
            raise ValueError(f"invalid column clue line: {clues}")
        if sum(clues) + max(0, len(clues) - 1) > row_count:
            raise ValueError(f"column clue line exceeds board height: {clues}")
    row_options = [_line_patterns(column_count, clues) for clues in row_clues]
    column_options = [_line_patterns(row_count, clues) for clues in column_clues]
    grid = [[-1] * column_count for _ in range(row_count)]
    branches = 0

    def search(
        state: list[list[int]],
        rows: list[list[int]],
        columns: list[list[int]],
    ) -> list[list[int]] | None:
        nonlocal branches
        branches += 1
        if branches > max_branches:
            raise RuntimeError("nonogram branch limit exceeded")
        state = [line[:] for line in state]
        rows = [options[:] for options in rows]
        columns = [options[:] for options in columns]
        if not _propagate(state, rows, columns):
            return None
        if all(value != -1 for line in state for value in line):
            return state
        choices = [
            (len(options), "row", index)
            for index, options in enumerate(rows)
            if len(options) > 1
        ]
        choices.extend(
            (len(options), "column", index)
            for index, options in enumerate(columns)
            if len(options) > 1
        )
        _, axis, index = min(choices)
        patterns = rows[index] if axis == "row" else columns[index]
        for pattern in patterns:
            next_state = [line[:] for line in state]
            if axis == "row":
                next_state[index] = [
                    1 if pattern & (1 << column) else 0
                    for column in range(column_count)
                ]
            else:
                for row in range(row_count):
                    next_state[row][index] = 1 if pattern & (1 << row) else 0
            solution = search(next_state, rows, columns)
            if solution is not None:
                return solution
        return None

    solution = search(grid, row_options, column_options)
    if solution is None:
        raise ValueError("nonogram has no solution")
    if any(_runs(solution[row]) != row_clues[row] for row in range(row_count)):
        raise ValueError("row clue verification failed")
    if any(
        _runs([solution[row][column] for row in range(row_count)])
        != column_clues[column]
        for column in range(column_count)
    ):
        raise ValueError("column clue verification failed")
    return solution


def solution_runs(solution: list[list[int]]) -> list[tuple[int, int, int]]:
    actions = []
    for row, line in enumerate(solution):
        start = None
        for column, value in enumerate([*line, 0]):
            if value and start is None:
                start = column
            elif not value and start is not None:
                actions.append((row, start, column - 1))
                start = None
    return actions


def _to_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    channels = array[..., :3].astype(np.float32)
    return channels[..., 0] * 0.114 + channels[..., 1] * 0.587 + channels[..., 2] * 0.299


def _peak_positions(scores: np.ndarray, min_score: float) -> list[tuple[int, float]]:
    candidates = []
    for index in range(1, len(scores) - 1):
        value = float(scores[index])
        if value >= min_score and value >= scores[index - 1] and value >= scores[index + 1]:
            candidates.append((index, value))
    candidates.sort(key=lambda item: item[1], reverse=True)
    selected: list[tuple[int, float]] = []
    for position, value in candidates:
        if all(abs(position - other) > 4 for other, _ in selected):
            selected.append((position, value))
        if len(selected) >= 100:
            break
    return sorted(selected)


def _regular_sequences(scores: np.ndarray, count: int) -> list[tuple[float, tuple[float, ...]]]:
    minimum = max(70.0, float(scores.max()) * 0.16)
    peaks = _peak_positions(scores, minimum)
    if len(peaks) < count:
        return []
    maximum = max(value for _, value in peaks)
    candidates = []
    for start_index, (start, _) in enumerate(peaks):
        for end, _ in peaks[start_index + count - 1 :]:
            step = (end - start) / (count - 1)
            if not 20 <= step <= 60:
                continue
            tolerance = max(3.0, step * 0.12)
            positions = []
            strength = 0.0
            error = 0.0
            used = set()
            for offset in range(count):
                expected = start + offset * step
                nearest_index, nearest = min(
                    enumerate(peaks),
                    key=lambda item: abs(item[1][0] - expected),
                )
                distance = abs(nearest[0] - expected)
                if distance > tolerance or nearest_index in used:
                    positions = []
                    break
                used.add(nearest_index)
                positions.append(float(nearest[0]))
                strength += nearest[1] / maximum
                error += distance / tolerance
            if positions:
                score = strength - error * 0.25 + start / max(1, len(scores)) * 2.0
                candidates.append((score, tuple(positions)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    output = []
    seen = set()
    for score, positions in candidates:
        key = (round(positions[0]), round(positions[-1]))
        if key not in seen:
            seen.add(key)
            output.append((score, positions))
        if len(output) >= 12:
            break
    return output


def _grid_content_penalty(
    gray: np.ndarray,
    x_lines: tuple[float, ...],
    y_lines: tuple[float, ...],
    columns: int,
    rows: int,
) -> float:
    suspicious = 0
    for row in range(rows):
        for column in range(columns):
            left = round(x_lines[column] * 0.7 + x_lines[column + 1] * 0.3)
            right = round(x_lines[column] * 0.3 + x_lines[column + 1] * 0.7)
            top = round(y_lines[row] * 0.7 + y_lines[row + 1] * 0.3)
            bottom = round(y_lines[row] * 0.3 + y_lines[row + 1] * 0.7)
            patch = gray[
                max(0, top) : max(top + 1, bottom),
                max(0, left) : max(left + 1, right),
            ]
            dark_ratio = float(np.mean(patch < 110)) if patch.size else 1.0
            if 0.025 < dark_ratio < 0.45:
                suspicious += 1
    return suspicious * 0.6


def _patch_chroma(
    image: np.ndarray, center_x: float, center_y: float, width: float, height: float
) -> float:
    array = np.asarray(image)
    x_start = max(0, round(center_x - width / 2))
    x_end = min(array.shape[1], round(center_x + width / 2))
    y_start = max(0, round(center_y - height / 2))
    y_end = min(array.shape[0], round(center_y + height / 2))
    patch = array[y_start:y_end, x_start:x_end, :3].astype(np.float32)
    if not patch.size:
        return 0.0
    return float(np.mean(patch.max(axis=2) - patch.min(axis=2)))


def _grid_boundary_transition_score(
    image: np.ndarray,
    x_lines: tuple[float, ...],
    y_lines: tuple[float, ...],
    columns: int,
    rows: int,
) -> float:
    cell_width = (x_lines[-1] - x_lines[0]) / columns
    cell_height = (y_lines[-1] - y_lines[0]) / rows
    left_transitions = []
    for row in range(rows):
        center_y = (y_lines[row] + y_lines[row + 1]) / 2
        outside = _patch_chroma(
            image,
            x_lines[0] - cell_width * 0.5,
            center_y,
            cell_width * 0.3,
            cell_height * 0.3,
        )
        inside = _patch_chroma(
            image,
            x_lines[0] + cell_width * 0.5,
            center_y,
            cell_width * 0.3,
            cell_height * 0.3,
        )
        left_transitions.append(outside - inside)
    top_transitions = []
    for column in range(columns):
        center_x = (x_lines[column] + x_lines[column + 1]) / 2
        outside = _patch_chroma(
            image,
            center_x,
            y_lines[0] - cell_height * 0.5,
            cell_width * 0.3,
            cell_height * 0.3,
        )
        inside = _patch_chroma(
            image,
            center_x,
            y_lines[0] + cell_height * 0.5,
            cell_width * 0.3,
            cell_height * 0.3,
        )
        top_transitions.append(outside - inside)
    return float(np.median(left_transitions) + np.median(top_transitions))


def _find_clue_boundary(
    scores: np.ndarray,
    origin: float,
    step: float,
    max_slots: int,
    tolerance_ratio: float = 0.12,
) -> float | None:
    minimum = max(70.0, float(scores.max()) * 0.45)
    peaks = _peak_positions(scores, minimum)
    tolerance = max(3.0, step * tolerance_ratio)
    boundary = None
    for slot in range(1, max_slots + 1):
        expected = origin - slot * step
        if expected < 0:
            break
        nearest = min(peaks, key=lambda item: abs(item[0] - expected), default=None)
        if nearest is None or abs(nearest[0] - expected) > tolerance:
            break
        boundary = float(nearest[0])
    return boundary


def detect_grid(image: np.ndarray) -> GridGeometry | None:
    gray = _to_gray(image)
    dark = gray < 185
    x_scores = dark.sum(axis=0)
    y_scores = dark.sum(axis=1)
    x_candidates = []
    y_candidates = []
    for columns in SUPPORTED_GRID_SIZES:
        for score, lines in _regular_sequences(x_scores, columns + 1):
            step = (lines[-1] - lines[0]) / columns
            x_candidates.append((score, columns, lines, step))
    for rows in SUPPORTED_GRID_SIZES:
        for score, lines in _regular_sequences(y_scores, rows + 1):
            step = (lines[-1] - lines[0]) / rows
            y_candidates.append((score, rows, lines, step))
    x_candidates.sort(key=lambda item: item[0], reverse=True)
    y_candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = []
    for x_score, columns, x_lines, x_step in x_candidates[:32]:
        for y_score, rows, y_lines, y_step in y_candidates[:32]:
            if columns != rows:
                continue
            step_error = abs(x_step - y_step) / max(x_step, y_step)
            if step_error > 0.12:
                continue
            content_penalty = _grid_content_penalty(
                gray, x_lines, y_lines, columns, rows
            )
            transition_score = _grid_boundary_transition_score(
                image, x_lines, y_lines, columns, rows
            )
            score = (
                x_score
                + y_score
                - step_error * 5
                - content_penalty * 0.15
                + transition_score * 0.2
            )
            candidates.append(GridGeometry(columns, rows, x_lines, y_lines, score))
    best = max(candidates, key=lambda item: item.score, default=None)
    if best is None:
        return None
    row_clue_slots = (best.columns + 1) // 2
    column_clue_slots = (best.rows + 1) // 2
    y_start = max(0, round(best.y_lines[0]))
    y_end = min(dark.shape[0], round(best.y_lines[-1]) + 1)
    x_start = max(0, round(best.x_lines[0]))
    x_end = min(dark.shape[1], round(best.x_lines[-1]) + 1)
    clue_x_scores = dark[y_start:y_end, :].sum(axis=0)
    clue_y_scores = dark[:, x_start:x_end].sum(axis=1)
    clue_left = _find_clue_boundary(
        clue_x_scores, best.x_lines[0], best.cell_width, row_clue_slots
    )
    clue_top = _find_clue_boundary(
        clue_y_scores, best.y_lines[0], best.cell_height, column_clue_slots
    )
    return GridGeometry(
        best.columns,
        best.rows,
        best.x_lines,
        best.y_lines,
        best.score,
        clue_left=clue_left,
        clue_top=clue_top,
    )


def _clip_roi(roi: tuple[int, int, int, int], image: np.ndarray) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x, y, roi_width, roi_height = roi
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    right = max(x + 1, min(width, int(x + roi_width)))
    bottom = max(y + 1, min(height, int(y + roi_height)))
    return x, y, right - x, bottom - y


def _clue_cell_has_digit(
    image: np.ndarray, roi: tuple[int, int, int, int]
) -> bool:
    x, y, width, height = _clip_roi(roi, image)
    gray = _to_gray(image[y : y + height, x : x + width])
    margin_x = max(2, round(width * 0.14))
    margin_y = max(2, round(height * 0.14))
    core = gray[margin_y : max(margin_y + 1, height - margin_y), margin_x : max(margin_x + 1, width - margin_x)]
    if not core.size:
        return False
    dark_pixels = int(np.count_nonzero(core < 150))
    return dark_pixels >= max(10, round(core.size * 0.025))


def _ocr_results(detail) -> list:
    if detail is None:
        return []
    results = [
        getattr(detail, "best_result", None),
        *getattr(detail, "all_results", []),
        *getattr(detail, "filtered_results", []),
    ]
    output = []
    seen = set()
    for result in results:
        if result is None:
            continue
        box = getattr(result, "box", None)
        key = (
            str(getattr(result, "text", "")),
            getattr(box, "x", -1),
            getattr(box, "y", -1),
            getattr(box, "w", -1),
            getattr(box, "h", -1),
        )
        if key not in seen:
            seen.add(key)
            output.append(result)
    return output


_CLUE_OCR_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "I": "1",
        "i": "1",
        "l": "1",
        "L": "1",
        "|": "1",
        "!": "1",
        "Z": "2",
        "z": "2",
        "E": "3",
        "A": "4",
        "S": "5",
        "s": "5",
        "G": "6",
        "b": "6",
        "T": "7",
        "B": "8",
        "g": "9",
        "q": "9",
    }
)


def _normalize_clue_text(text: str) -> str:
    return str(text).translate(_CLUE_OCR_TRANSLATION)


def _numbers_from_text(text: str, size: int) -> list[int]:
    normalized = _normalize_clue_text(text)
    values = []
    for token in re.findall(r"\d+", normalized):
        value = int(token)
        if value == 0:
            continue
        if value <= size:
            values.append(value)
        elif all(character != "0" for character in token):
            values.extend(int(character) for character in token)
    return values


def _number_from_clue_cell(texts: list[str], size: int) -> list[int]:
    digits = "".join(
        token
        for text in texts
        for token in re.findall(r"\d+", _normalize_clue_text(text))
    )
    if not digits:
        return []
    value = int(digits)
    return [value] if 1 <= value <= size else []


def _read_clue(
    context: Context,
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    size: int,
    vertical: bool,
) -> tuple[list[int], list[str], bool]:
    detail = context.run_recognition_direct(
        JRecognitionType.OCR,
        JOCR(expected=[], roi=_clip_roi(roi, image), threshold=0.2),
        image,
    )
    results = _ocr_results(detail)
    primary = "y" if vertical else "x"
    secondary = "x" if vertical else "y"
    results.sort(
        key=lambda result: (
            getattr(getattr(result, "box", None), primary, 0),
            getattr(getattr(result, "box", None), secondary, 0),
        )
    )
    texts = [str(getattr(result, "text", "")) for result in results]
    values = []
    recognized = False
    for text in texts:
        normalized = _normalize_clue_text(text)
        recognized = recognized or bool(re.search(r"\d", normalized))
        values.extend(_numbers_from_text(text, size))
    return values, texts, recognized


def _read_clue_cells(
    context: Context,
    image: np.ndarray,
    geometry: GridGeometry,
    line_index: int,
    vertical: bool,
) -> tuple[list[int], list[list[str]], bool]:
    line_length = geometry.rows if vertical else geometry.columns
    max_slots = (line_length + 1) // 2
    cell_width = geometry.cell_width
    cell_height = geometry.cell_height
    if vertical:
        available = geometry.y_lines[0] - (geometry.clue_top or 0)
        slot_count = round(available / cell_height)
    else:
        available = geometry.x_lines[0] - (geometry.clue_left or 0)
        slot_count = round(available / cell_width)
    slot_count = max(1, min(max_slots, slot_count))
    values = []
    raw = []
    recognized = True
    for slot in range(slot_count - 1, -1, -1):
        if vertical:
            left = geometry.x_lines[line_index] + cell_width * 0.12
            right = geometry.x_lines[line_index + 1] - cell_width * 0.12
            top = geometry.y_lines[0] - (slot + 1) * cell_height + cell_height * 0.1
            bottom = geometry.y_lines[0] - slot * cell_height - cell_height * 0.1
        else:
            left = geometry.x_lines[0] - (slot + 1) * cell_width + cell_width * 0.1
            right = geometry.x_lines[0] - slot * cell_width - cell_width * 0.1
            top = geometry.y_lines[line_index] + cell_height * 0.1
            bottom = geometry.y_lines[line_index + 1] - cell_height * 0.1
        roi = (round(left), round(top), round(right - left), round(bottom - top))
        if not _clue_cell_has_digit(image, roi):
            raw.append([])
            continue
        _, texts, _ = _read_clue(
            context,
            image,
            roi,
            line_length,
            vertical,
        )
        clues = _number_from_clue_cell(texts, line_length)
        raw.append(texts)
        if clues:
            values.extend(clues)
        else:
            recognized = False
    return values, raw, recognized


def extract_clues(
    context: Context,
    image: np.ndarray,
    geometry: GridGeometry,
    split_cells: bool = False,
) -> tuple[list[list[int]], list[list[int]]] | None:
    row_count = geometry.rows
    column_count = geometry.columns
    row_clue_slots = (column_count + 1) // 2
    column_clue_slots = (row_count + 1) // 2
    cell_width = geometry.cell_width
    cell_height = geometry.cell_height
    row_left = geometry.clue_left
    if row_left is None:
        row_left = geometry.x_lines[0] - cell_width * (row_clue_slots + 0.25)
    column_top = geometry.clue_top
    if column_top is None:
        column_top = geometry.y_lines[0] - cell_height * (column_clue_slots + 0.25)
    row_clues = []
    column_clues = []
    raw_rows = []
    raw_columns = []
    recognized_rows = []
    recognized_columns = []
    for row in range(row_count):
        if split_cells:
            clues, texts, recognized = _read_clue_cells(
                context, image, geometry, row, False
            )
        else:
            top = geometry.y_lines[row] + cell_height * 0.1
            bottom = geometry.y_lines[row + 1] - cell_height * 0.1
            clues, texts, recognized = _read_clue(
                context,
                image,
                (
                    round(row_left + cell_width * 0.08),
                    round(top),
                    round(geometry.x_lines[0] - row_left - cell_width * 0.16),
                    round(bottom - top),
                ),
                column_count,
                False,
            )
        row_clues.append(clues)
        raw_rows.append(texts)
        recognized_rows.append(recognized)
    for column in range(column_count):
        if split_cells:
            clues, texts, recognized = _read_clue_cells(
                context, image, geometry, column, True
            )
        else:
            left = geometry.x_lines[column] + cell_width * 0.1
            right = geometry.x_lines[column + 1] - cell_width * 0.1
            clues, texts, recognized = _read_clue(
                context,
                image,
                (
                    round(left),
                    round(column_top + cell_height * 0.08),
                    round(right - left),
                    round(geometry.y_lines[0] - column_top - cell_height * 0.16),
                ),
                row_count,
                True,
            )
        column_clues.append(clues)
        raw_columns.append(texts)
        recognized_columns.append(recognized)
    mode = "cells" if split_cells else "lines"
    print(f"nonogram OCR mode={mode} rows={raw_rows!r} parsed={row_clues!r}")
    print(f"nonogram OCR mode={mode} columns={raw_columns!r} parsed={column_clues!r}")
    if not all([*recognized_rows, *recognized_columns]):
        return None
    if any(
        sum(clues) + max(0, len(clues) - 1) > column_count
        for clues in row_clues
    ):
        return None
    if any(
        sum(clues) + max(0, len(clues) - 1) > row_count
        for clues in column_clues
    ):
        return None
    return row_clues, column_clues


def _filled_cells(image: np.ndarray, geometry: GridGeometry) -> list[list[bool]]:
    gray = _to_gray(image)
    filled = []
    for row in range(geometry.rows):
        line = []
        for column in range(geometry.columns):
            left = round(geometry.x_lines[column] + geometry.cell_width * 0.28)
            right = round(geometry.x_lines[column + 1] - geometry.cell_width * 0.28)
            top = round(geometry.y_lines[row] + geometry.cell_height * 0.28)
            bottom = round(geometry.y_lines[row + 1] - geometry.cell_height * 0.28)
            patch = gray[
                max(0, top) : max(top + 1, bottom),
                max(0, left) : max(left + 1, right),
            ]
            line.append(bool(patch.size and np.mean(patch < 90) > 0.55))
        filled.append(line)
    return filled


def _paint_solution(
    context: Context,
    image: np.ndarray,
    geometry: GridGeometry,
    solution: list[list[int]],
    pause: float,
) -> bool:
    current = _filled_cells(image, geometry)
    for row in range(geometry.rows):
        for column in range(geometry.columns):
            if current[row][column] and not solution[row][column]:
                print(f"nonogram unexpected filled cell at row={row + 1}, column={column + 1}")
                return False
    pending = [
        [bool(solution[row][column] and not current[row][column]) for column in range(geometry.columns)]
        for row in range(geometry.rows)
    ]
    actions = solution_runs([[int(value) for value in line] for line in pending])
    print(f"nonogram paint actions={[(r + 1, s + 1, e + 1) for r, s, e in actions]}")
    for row, start, end in actions:
        start_x, y = geometry.center(row, start)
        end_x, _ = geometry.center(row, end)
        if start == end:
            result = context.tasker.controller.post_click(start_x, y).wait()
            action_name = "click"
        else:
            duration = max(250, 140 + (end - start + 1) * 65)
            result = context.tasker.controller.post_swipe(
                start_x, y, end_x, y, duration
            ).wait()
            action_name = "swipe"
        print(
            f"nonogram {action_name}: row={row + 1} columns={start + 1}-{end + 1} "
            f"from=({start_x},{y}) to=({end_x},{y})"
        )
        if not result.succeeded:
            return False
        time.sleep(pause)
    return True


def _solution_text(solution: list[list[int]]) -> str:
    return "\n".join("".join("#" if value else "." for value in line) for line in solution)


@AgentServer.custom_action("识别并完成黑白数织")
class SolveNonogramAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        param = _load_param(argv.custom_action_param)
        success_node = str(param.get("success_node", "等待继续或通关"))
        failure_node = str(param.get("failure_node", "人工解谜失败等待"))
        pause = max(0.03, float(param.get("pause", 0.08)))
        close_overlay_with_back = bool(param.get("close_overlay_with_back", True))
        try:
            image = context.tasker.controller.post_screencap().wait().get()
            if image is None:
                raise RuntimeError("screenshot returned no image")
            image = np.asarray(image)
            geometry = detect_grid(image)
            if geometry is None and close_overlay_with_back:
                print("nonogram grid not found; pressing Android back once to close the tool overlay")
                context.tasker.controller.post_click_key(4).wait()
                time.sleep(0.8)
                image = context.tasker.controller.post_screencap().wait().get()
                if image is None:
                    raise RuntimeError("screenshot after closing overlay returned no image")
                image = np.asarray(image)
                geometry = detect_grid(image)
            if geometry is None:
                raise ValueError("unable to locate a supported 10x10, 15x15, or 20x20 grid")
            print(
                f"nonogram grid: size={geometry.rows}x{geometry.columns} "
                f"x={geometry.x_lines[0]:.1f}-{geometry.x_lines[-1]:.1f} "
                f"y={geometry.y_lines[0]:.1f}-{geometry.y_lines[-1]:.1f} "
                f"clue_left={geometry.clue_left} clue_top={geometry.clue_top} "
                f"score={geometry.score:.2f}"
            )
            if geometry.clue_left is None or geometry.clue_top is None:
                raise ValueError(
                    "clue area boundaries not found; refusing a possible unsupported oversized grid"
                )
            cell_first = max(geometry.rows, geometry.columns) > 10
            if cell_first:
                print("nonogram large-clue mode: reading one clue cell at a time")
            clues = extract_clues(context, image, geometry, split_cells=cell_first)
            solution = None
            if clues is not None:
                row_clues, column_clues = clues
                row_total = sum(map(sum, row_clues))
                column_total = sum(map(sum, column_clues))
                if row_total == column_total:
                    try:
                        solution = solve_nonogram(row_clues, column_clues)
                    except ValueError as exc:
                        mode = "cell" if cell_first else "line"
                        print(f"nonogram {mode} OCR clues rejected: {exc}")
                else:
                    mode = "cell" if cell_first else "line"
                    print(
                        f"nonogram {mode} OCR totals differ: "
                        f"rows={row_total} columns={column_total}"
                    )
            if solution is None and not cell_first:
                print("nonogram retrying OCR one clue cell at a time")
                clues = extract_clues(context, image, geometry, split_cells=True)
                if clues is None:
                    raise ValueError("cell OCR did not produce a complete valid clue set")
                row_clues, column_clues = clues
                row_total = sum(map(sum, row_clues))
                column_total = sum(map(sum, column_clues))
                if row_total != column_total:
                    raise ValueError(
                        f"cell OCR totals differ: rows={row_total} columns={column_total}"
                    )
                solution = solve_nonogram(row_clues, column_clues)
            if solution is None:
                raise ValueError("cell OCR did not produce a solvable clue set")
            print(f"nonogram solution:\n{_solution_text(solution)}")
            succeeded = _paint_solution(context, image, geometry, solution, pause)
            next_node = success_node if succeeded else failure_node
        except Exception as exc:
            print(f"nonogram solve failed: {exc}")
            next_node = failure_node
        context.override_next(argv.node_name, [next_node] if next_node else [])
        return True
