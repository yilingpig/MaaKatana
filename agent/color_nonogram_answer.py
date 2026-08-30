from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


ANSWER_GRID_SIZE = 15
EMPTY_CELL = -1
Rgb = tuple[int, int, int]
Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PaintSegment:
    color: int
    row: int
    start_column: int
    end_column: int

    @property
    def length(self) -> int:
        return self.end_column - self.start_column + 1

    @property
    def cells(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (self.row, column)
            for column in range(self.start_column, self.end_column + 1)
        )


@dataclass(frozen=True)
class AnswerBoard:
    grid: Grid
    colors: tuple[Rgb, ...]
    grid_roi: tuple[int, int, int, int]
    cell_variances: tuple[float, ...]

    @property
    def color_count(self) -> int:
        return len(self.colors)


class AnswerBoardError(ValueError):
    pass


def detect_answer_grid_roi(
    image: np.ndarray,
    *,
    minimum_side: int = 200,
    maximum_aspect_error: float = 0.025,
) -> tuple[int, int, int, int]:
    """Find the smallest complete, nearly square 15x15 answer frame."""
    rgb = _as_rgb_image(image)
    height, width = rgb.shape[:2]
    dark = np.max(rgb, axis=2) <= 90

    def horizontal_lines() -> list[tuple[int, int, int]]:
        lines = []
        for row in range(height):
            runs = _longest_runs(dark[row], minimum_side)
            if not runs:
                continue
            start, end = max(runs, key=lambda item: item[1] - item[0])
            if start <= 2 and end >= width - 2:
                continue
            lines.append((row, start, end))
        return lines

    def vertical_lines() -> list[tuple[int, int, int]]:
        lines = []
        for column in range(width):
            runs = _longest_runs(dark[:, column], minimum_side)
            if not runs:
                continue
            start, end = max(runs, key=lambda item: item[1] - item[0])
            if start <= 2 and end >= height - 2:
                continue
            lines.append((column, start, end))
        return lines

    horizontal = horizontal_lines()
    vertical = vertical_lines()
    rectangles = []
    for first_index, (top, left, right) in enumerate(horizontal):
        for bottom, other_left, other_right in horizontal[first_index + 1 :]:
            if bottom - top < minimum_side:
                continue
            if abs(left - other_left) > 5 or abs(right - other_right) > 5:
                continue
            for first_vertical, (left_edge, top_edge, bottom_edge) in enumerate(vertical):
                for right_edge, other_top, other_bottom in vertical[first_vertical + 1 :]:
                    if right_edge - left_edge < minimum_side:
                        continue
                    if abs(top - other_top) > 5 or abs(bottom - other_bottom) > 5:
                        continue
                    side_x = right_edge - left_edge
                    side_y = bottom - top
                    if abs(side_x - side_y) / max(side_x, side_y) > maximum_aspect_error:
                        continue
                    if not 20.0 <= side_x / ANSWER_GRID_SIZE <= 70.0:
                        continue
                    if not 20.0 <= side_y / ANSWER_GRID_SIZE <= 70.0:
                        continue
                    roi = (left_edge + 1, top + 1, side_x, side_y)
                    score = _answer_roi_score(rgb, roi, dark)
                    if score is None or score[1] < -30.0 or score[2] > 1.0:
                        continue
                    color_count, filled_ratio = _answer_roi_content_score(rgb, roi)
                    if color_count < 3 or not 0.15 <= filled_ratio <= 0.90:
                        continue
                    rectangles.append((side_x * side_y, -color_count, -filled_ratio, roi))
    if not rectangles:
        raise AnswerBoardError("could not detect a square 15x15 answer board")
    _, _, _, roi = max(rectangles, key=lambda item: item[0])
    _validate_roi(roi, rgb.shape[:2])
    return roi

def _answer_roi_content_score(
    rgb: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[int, float]:
    x, y, width, height = roi
    samples = []
    for row in range(ANSWER_GRID_SIZE):
        for column in range(ANSWER_GRID_SIZE):
            left = round(x + (column + 0.20) * width / ANSWER_GRID_SIZE)
            right = round(x + (column + 0.80) * width / ANSWER_GRID_SIZE)
            top = round(y + (row + 0.20) * height / ANSWER_GRID_SIZE)
            bottom = round(y + (row + 0.80) * height / ANSWER_GRID_SIZE)
            patch = rgb[top:bottom, left:right]
            if not patch.size:
                return 0, 0.0
            samples.append(np.median(patch.reshape(-1, 3), axis=0))
    empty = _find_empty_color(samples, 35.0)
    centers: list[np.ndarray] = []
    filled = 0
    for sample in samples:
        if _distance(sample, empty) <= 45.0:
            continue
        filled += 1
        nearest = _nearest(sample, centers)
        if nearest is None or nearest[1] > 45.0:
            centers.append(sample.copy())
    return len(centers), filled / len(samples)

def _answer_roi_score(
    rgb: np.ndarray,
    roi: tuple[int, int, int, int],
    dark: np.ndarray,
) -> tuple[float, float, float] | None:
    """Score frame completeness and stable 15x15 cell interiors."""
    x, y, width, height = roi
    border = max(2, round(min(width, height) * 0.006))
    horizontal_coverage = min(
        float(np.mean(dark[max(0, y - border):min(dark.shape[0], y + border + 1), x:x + width])),
        float(np.mean(dark[max(0, y + height - border):min(dark.shape[0], y + height + border + 1), x:x + width])),
    )
    vertical_coverage = min(
        float(np.mean(dark[y:y + height, max(0, x - border):min(dark.shape[1], x + border + 1)])),
        float(np.mean(dark[y:y + height, max(0, x + width - border):min(dark.shape[1], x + width + border + 1)])),
    )
    if min(horizontal_coverage, vertical_coverage) < 0.05:
        return None

    variances = []
    for row in range(ANSWER_GRID_SIZE):
        for column in range(ANSWER_GRID_SIZE):
            left = round(x + (column + 0.20) * width / ANSWER_GRID_SIZE)
            right = round(x + (column + 0.80) * width / ANSWER_GRID_SIZE)
            top = round(y + (row + 0.20) * height / ANSWER_GRID_SIZE)
            bottom = round(y + (row + 0.80) * height / ANSWER_GRID_SIZE)
            patch = rgb[top:bottom, left:right].astype(np.float32)
            if not patch.size:
                return None
            variances.append(float(np.mean(np.std(patch, axis=(0, 1)))))
    stable_cells = sum(value <= 135.0 for value in variances)
    if stable_cells < 210:
        return None
    return (stable_cells / (ANSWER_GRID_SIZE * ANSWER_GRID_SIZE), -float(np.mean(variances)), horizontal_coverage + vertical_coverage)


def extract_answer_board(
    image: np.ndarray,
    *,
    grid_roi: tuple[int, int, int, int] | None = None,
    size: int = ANSWER_GRID_SIZE,
    maximum_cell_variance: float = 125.0,
    empty_saturation: float = 35.0,
    color_distance: float = 45.0,
) -> AnswerBoard:
    """Extract a deterministic, palette-independent answer matrix.

    Matrix values are ``EMPTY_CELL`` for blank cells and zero-based color
    cluster ids for filled cells.  No OCR or puzzle search is performed.
    """
    if size != ANSWER_GRID_SIZE:
        raise AnswerBoardError(f"answer board must be {ANSWER_GRID_SIZE}x{ANSWER_GRID_SIZE}")
    rgb = _as_rgb_image(image)
    roi = grid_roi or detect_answer_grid_roi(rgb)
    _validate_roi(roi, rgb.shape[:2])
    x, y, width, height = roi
    if width < size or height < size:
        raise AnswerBoardError("answer board ROI is too small for 15x15 cells")

    samples: list[np.ndarray] = []
    variances: list[float] = []
    for row in range(size):
        for column in range(size):
            left = round(x + (column + 0.20) * width / size)
            right = round(x + (column + 0.80) * width / size)
            top = round(y + (row + 0.20) * height / size)
            bottom = round(y + (row + 0.80) * height / size)
            patch = rgb[top:bottom, left:right].astype(np.float32)
            if not patch.size:
                raise AnswerBoardError(f"empty cell patch at row={row}, column={column}")
            samples.append(np.median(patch.reshape(-1, 3), axis=0))
            variances.append(float(np.mean(np.std(patch, axis=(0, 1)))))

    if max(variances, default=0.0) > maximum_cell_variance:
        raise AnswerBoardError("answer board contains blurred or unstable cell samples")
    empty = _find_empty_color(samples, empty_saturation)
    labels: list[int] = []
    color_centers: list[np.ndarray] = []
    for sample in samples:
        if _distance(sample, empty) <= color_distance:
            labels.append(EMPTY_CELL)
            continue
        nearest = _nearest(sample, color_centers)
        if nearest is None or nearest[1] > color_distance:
            color_centers.append(sample.copy())
            labels.append(len(color_centers) - 1)
        else:
            labels.append(nearest[0])
            color_centers[nearest[0]] = (color_centers[nearest[0]] + sample) / 2.0
    if not color_centers:
        raise AnswerBoardError("answer board contains no filled colors")
    grid = tuple(
        tuple(labels[row * size : (row + 1) * size])
        for row in range(size)
    )
    colors = tuple(_rgb(center) for center in color_centers)
    validate_answer_grid(grid, color_count=len(colors))
    return AnswerBoard(grid, colors, roi, tuple(variances))


def map_answer_colors(
    answer_colors: Sequence[Sequence[float]],
    palette_colors: Sequence[object | Sequence[float]],
    *,
    maximum_distance: float = 70.0,
    minimum_margin: float = 8.0,
) -> tuple[int, ...]:
    """Map each answer color to a distinct current-palette index."""
    palette_rgb = tuple(_palette_rgb(color) for color in palette_colors)
    if not answer_colors or not palette_rgb:
        raise AnswerBoardError("answer and current palettes must not be empty")
    if len(answer_colors) > len(palette_rgb):
        raise AnswerBoardError("answer uses more colors than the current palette")
    distances = np.asarray(
        [[_distance(answer, palette) for palette in palette_rgb] for answer in answer_colors],
        dtype=np.float32,
    )
    result: list[int] = []
    used: set[int] = set()
    for row in distances:
        order = np.argsort(row)
        index = int(order[0])
        if float(row[index]) > maximum_distance:
            raise AnswerBoardError("answer color does not match the current palette")
        if len(order) > 1 and float(row[order[1]] - row[index]) < minimum_margin:
            raise AnswerBoardError("answer color has an ambiguous palette match")
        if index in used:
            raise AnswerBoardError("answer colors do not map one-to-one to the current palette")
        used.add(index)
        result.append(index)
    return tuple(result)


def remap_answer_grid(grid: Sequence[Sequence[int]], color_mapping: Sequence[int]) -> Grid:
    validate_answer_grid(grid, color_count=len(color_mapping))
    return tuple(
        tuple(EMPTY_CELL if value == EMPTY_CELL else int(color_mapping[value]) for value in row)
        for row in grid
    )


def answer_grid_to_segments(
    grid: Sequence[Sequence[int]],
    *,
    segment_factory: Callable[[int, int, int, int], object] | None = None,
) -> tuple[object, ...]:
    """Convert each horizontal filled run into one PaintSegment-like object."""
    validate_answer_grid(grid)
    factory = segment_factory or PaintSegment
    segments = []
    for row_index, line in enumerate(grid):
        column = 0
        while column < ANSWER_GRID_SIZE:
            if line[column] == EMPTY_CELL:
                column += 1
                continue
            color = int(line[column])
            start = column
            while column + 1 < ANSWER_GRID_SIZE and line[column + 1] == color:
                column += 1
            segments.append(factory(color, row_index, start, column))
            column += 1
    return tuple(segments)


def validate_answer_grid(
    grid: Sequence[Sequence[int]],
    *,
    color_count: int | None = None,
    rows: Sequence[Sequence[tuple[int, int]]] | None = None,
    columns: Sequence[Sequence[tuple[int, int]]] | None = None,
) -> None:
    if len(grid) != ANSWER_GRID_SIZE or any(len(row) != ANSWER_GRID_SIZE for row in grid):
        raise AnswerBoardError("answer matrix must be exactly 15x15")
    values = [int(value) for row in grid for value in row]
    if any(value < EMPTY_CELL for value in values):
        raise AnswerBoardError("answer matrix contains an invalid negative color")
    if color_count is not None and any(value >= color_count for value in values):
        raise AnswerBoardError("answer matrix contains a color outside color_count")
    if rows is not None or columns is not None:
        if rows is None or columns is None or len(rows) != 15 or len(columns) != 15:
            raise AnswerBoardError("row and column clues must both contain 15 lines")
        actual_rows, actual_columns = _grid_clues(grid)
        if tuple(tuple(line) for line in rows) != actual_rows or tuple(tuple(line) for line in columns) != actual_columns:
            raise AnswerBoardError("answer matrix does not match the supplied row/column clues")
    if color_count is not None:
        row_counts = _grid_color_counts(grid)
        if any(color not in row_counts for color in range(color_count)):
            raise AnswerBoardError("answer matrix does not contain every mapped color")


def _as_rgb_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise AnswerBoardError("answer screenshot must be an RGB/RGBA image")
    return array[:, :, :3].astype(np.uint8, copy=False)


def _validate_roi(roi: tuple[int, int, int, int], shape: tuple[int, int]) -> None:
    if len(roi) != 4 or any(int(value) != value for value in roi):
        raise AnswerBoardError("grid ROI must be an integer (x, y, width, height)")
    x, y, width, height = roi
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > shape[1] or y + height > shape[0]:
        raise AnswerBoardError("grid ROI is outside the screenshot")
    if abs(width - height) > max(4, round(max(width, height) * 0.05)):
        raise AnswerBoardError("answer board ROI must be approximately square")


def _find_empty_color(samples: Sequence[np.ndarray], empty_saturation: float) -> np.ndarray:
    low_saturation = [sample for sample in samples if _saturation(sample) <= empty_saturation]
    if not low_saturation:
        raise AnswerBoardError("could not identify the blank answer-cell color")
    return np.median(np.asarray(low_saturation), axis=0)


def _nearest(sample: np.ndarray, centers: Sequence[np.ndarray]) -> tuple[int, float] | None:
    if not centers:
        return None
    distances = [_distance(sample, center) for center in centers]
    index = int(np.argmin(distances))
    return index, distances[index]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)))


def _saturation(color: Sequence[float]) -> float:
    values = np.asarray(color, dtype=np.float32)
    return float(np.max(values) - np.min(values))


def _rgb(color: Sequence[float]) -> Rgb:
    return tuple(int(round(float(value))) for value in color[:3])


def _palette_rgb(color: object | Sequence[float]) -> Rgb:
    value = getattr(color, "rgb", color)
    if len(value) < 3:
        raise AnswerBoardError("palette color must provide three RGB values")
    return _rgb(value)


def _grid_clues(grid: Sequence[Sequence[int]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    rows = tuple(_line_clues(row) for row in grid)
    columns = tuple(_line_clues(tuple(grid[row][column] for row in range(15))) for column in range(15))
    return rows, columns


def _line_clues(line: Iterable[int]) -> tuple[tuple[int, int], ...]:
    result = []
    index = 0
    values = tuple(line)
    while index < len(values):
        if values[index] == EMPTY_CELL:
            index += 1
            continue
        color = values[index]
        end = index + 1
        while end < len(values) and values[end] == color:
            end += 1
        result.append((int(color), end - index))
        index = end
    return tuple(result)


def _grid_color_counts(grid: Sequence[Sequence[int]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in grid:
        for value in row:
            if value != EMPTY_CELL:
                counts[int(value)] = counts.get(int(value), 0) + 1
    return counts


def _longest_run_by_row(mask: np.ndarray, *, minimum_length: int) -> list[tuple[int, int]]:
    return _line_extents(mask, axis=1, minimum_length=minimum_length)


def _longest_run_by_column(mask: np.ndarray, *, minimum_length: int) -> list[tuple[int, int]]:
    return _line_extents(mask, axis=0, minimum_length=minimum_length)


def _line_extents(mask: np.ndarray, *, axis: int, minimum_length: int) -> list[tuple[int, int]]:
    lines = np.moveaxis(mask, axis, 0)
    extents = []
    for line in lines:
        runs = _longest_runs(line, minimum_length)
        if runs:
            extents.append(max(runs, key=lambda item: item[1] - item[0]))
    return extents


def _longest_runs(values: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate(np.r_[values, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum_length:
                runs.append((start, index))
            start = None
    return runs


def _pairs(runs: Sequence[tuple[int, int]]) -> Iterable[tuple[int, int]]:
    for first_index, first in enumerate(runs):
        for second in runs[first_index + 1 :]:
            yield first[0], second[1]
