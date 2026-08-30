from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JRecognitionType, JTemplateMatch, JOCR

from color_nonogram_solver import (
    COLOR_GRID_SIZE,
    PaletteColor,
    _colored_clues,
    _load_param,
    _palette_colors,
    _project_root,
    _solve_with_candidate_recovery,
    _verify_clue_color_match,
    detect_color_grid,
)
from color_nonogram_answer import (
    EMPTY_CELL,
    AnswerBoardError,
    extract_answer_board,
    map_answer_colors,
    remap_answer_grid,
)
from nonogram_solver import GridGeometry, _ocr_results


PROJECT_ROOT = _project_root(__file__)
DEFAULT_SUCCESS_NODE = "ColorNonogram1080pPaintTestSuccess"
DEFAULT_FAILURE_NODE = "ColorNonogram1080pPaintTestFailure"
CELL_COLOR_TOLERANCE = 90.0
CELL_MATCH_RATIO = 0.55
BASELINE_RGB_TOLERANCE = 45.0
BACKGROUND_PIXEL_TOLERANCE = 24.0
BACKGROUND_MIN_SUPPORT = 0.12
X_MARKER_MIN_DIAGONAL_SUPPORT = 0.45
X_MARKER_MIN_BRIGHT_RATIO = 0.16
X_MARKER_MIN_LUMINANCE_DELTA = 18.0
X_MARKER_INTERIOR_MARGIN_RATIO = 0.08
HIGHLIGHT_MIN_BRIGHT_RATIO = 0.85
HIGHLIGHT_MIN_LUMINANCE_DELTA = 35.0
STABILITY_MEAN_DELTA = 2.0
TARGET_STABILITY_MEAN_DELTA = 2.0
DEFAULT_PAINT_SETTLE_INITIAL = 0.12
DEFAULT_PAINT_SETTLE_POLL = 0.08
DEFAULT_PAINT_SETTLE_TIMEOUT = 1.2
DEFAULT_FAST_PAINT_SETTLE_INITIAL = 0.06
DEFAULT_FAST_PAINT_SETTLE_POLL = 0.04
DEFAULT_FAST_PAINT_SETTLE_TIMEOUT = 0.65
ANCHORED_GRID_DARK_THRESHOLD = 135.0
ANCHORED_GRID_LINE_SEARCH_RADIUS = 3
ANCHORED_GRID_MIN_LINE_SUPPORT = 0.55
ANCHORED_GRID_MIN_MEAN_SUPPORT = 0.85
ANCHORED_GRID_EXCEPTION_MIN_SUPPORT = 0.05
ANCHORED_GRID_EXCEPTION_MIN_MEAN_SUPPORT = 0.89
ANCHORED_GRID_EXCEPTION_MAX_LOW_SUPPORT_LINES = 4
ANCHORED_GRID_EXCEPTION_MAX_LOW_SUPPORT_RATIO = 0.125
ANCHORED_GRID_EXCEPTION_MIN_STRONG_LINE_RATIO = 0.75
ANCHORED_GRID_EXCEPTION_MAX_LOW_SUPPORT_LINES_PER_AXIS = 3
REFERENCE_FRAME_SEARCH_RADIUS = 8
REFERENCE_FRAME_STRIP_WIDTH = 8
REFERENCE_FRAME_MAX_MEAN_DELTA = 18.0
REFERENCE_FRAME_MIN_MARGIN = 1.25
REFERENCE_FRAME_MAX_OFFSET = 6
BASELINE_ALLOWED_STATES = frozenset(("background", "x_marker"))
PALETTE_RGB_TOLERANCE = 55.0
PALETTE_SOLID_MATCH_RATIO = 0.8
PALETTE_SOLID_DISTANCE = 35.0
PALETTE_SELECTED_BORDER_THRESHOLD = 0.15
EDITOR_MODAL_WHITE_THRESHOLD = 235
EDITOR_MODAL_MIN_EDGE_WIDTH_RATIO = 0.40
EDITOR_MODAL_EDGE_SCAN_RATIO = 0.08


@dataclass(frozen=True)
class CellObservation:
    row: int
    column: int
    center: tuple[int, int]
    rgb: tuple[int, int, int]
    state: str
    palette_index: int | None
    match_ratio: float
    distance: float
    background_ratio: float = 0.0
    background_distance: float = 0.0
    x_marker_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "column": self.column,
            "center": [self.center[0], self.center[1]],
            "rgb": list(self.rgb),
            "state": self.state,
            "palette_index": self.palette_index,
            "match_ratio": round(self.match_ratio, 6),
            "distance": round(self.distance, 6),
            "background_ratio": round(self.background_ratio, 6),
            "background_distance": round(self.background_distance, 6),
            "x_marker_score": round(self.x_marker_score, 6),
        }


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

    def to_dict(self) -> dict:
        return {
            "color": self.color,
            "row": self.row,
            "start_column": self.start_column,
            "end_column": self.end_column,
            "length": self.length,
            "operation": (
                "click"
                if self.length == 1
                else "click_pair"
                if self.length == 2
                else "swipe"
            ),
        }


class PaintSurfaceError(RuntimeError):
    def __init__(self, message: str, status: dict):
        super().__init__(message)
        self.status = status


def _artifact_dir() -> pathlib.Path:
    output_dir = PROJECT_ROOT / "debug" / "color_nonogram_live"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _save_image(image: np.ndarray, prefix: str, timestamp: str | None = None) -> str:
    timestamp = timestamp or _stamp()
    path = _artifact_dir() / f"{prefix}_{timestamp}.png"
    Image.fromarray(image[:, :, :3].astype(np.uint8)).save(path)
    return str(path)


def _cleanup_success_images(paths) -> list[str]:
    removed = []
    for value in paths:
        try:
            path = pathlib.Path(value)
            path.unlink()
            removed.append(str(path))
        except (FileNotFoundError, OSError):
            continue
    return removed


def _write_report(report: dict, prefix: str, timestamp: str | None = None) -> str:
    timestamp = timestamp or _stamp()
    path = _artifact_dir() / f"{prefix}_{timestamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _geometry_report(geometry: GridGeometry | None) -> dict | None:
    if geometry is None:
        return None
    return {
        "columns": int(geometry.columns),
        "rows": int(geometry.rows),
        "x_lines": [float(value) for value in geometry.x_lines],
        "y_lines": [float(value) for value in geometry.y_lines],
        "score": float(geometry.score),
        "cell_width": float(geometry.cell_width),
        "cell_height": float(geometry.cell_height),
        "clue_left": None if geometry.clue_left is None else float(geometry.clue_left),
        "clue_top": None if geometry.clue_top is None else float(geometry.clue_top),
    }


def _palette_report(palette: list[PaletteColor]) -> list[dict]:
    return [
        {
            "center": [int(color.center[0]), int(color.center[1])],
            "rgb": [int(round(value)) for value in color.rgb],
        }
        for color in palette
    ]


def _clues_report(lines: list[list[tuple[int, int]]]) -> list[list[list[int]]]:
    return [[[int(color), int(value)] for color, value in line] for line in lines]


def _solution_hash(solution: list[list[int]]) -> str:
    payload = json.dumps(solution, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _target_cells(solution: list[list[int]], color_count: int) -> list[tuple[int, int, int]]:
    if len(solution) != COLOR_GRID_SIZE or any(len(line) != COLOR_GRID_SIZE for line in solution):
        raise ValueError("solution must be a 15x15 matrix")
    cells = []
    for row, line in enumerate(solution):
        for column, value in enumerate(line):
            value = int(value)
            if value < 0 or value > color_count:
                raise ValueError(f"solution color is outside palette at row={row} column={column}")
            if value:
                cells.append((value - 1, row, column))
    if not 0 < len(cells) <= COLOR_GRID_SIZE * COLOR_GRID_SIZE:
        raise ValueError(f"expected 1-225 paint cells, found {len(cells)}")
    return cells


def _trial_segment(segments: list[PaintSegment], palette: list[PaletteColor], background_rgb: tuple[float, float, float]) -> PaintSegment:
    if not segments:
        raise ValueError("expected at least one paint segment")
    background = np.asarray(background_rgb, dtype=np.float32)
    return max(
        segments,
        key=lambda segment: (
            float(np.linalg.norm(np.asarray(palette[segment.color].rgb, dtype=np.float32) - background)),
            -segment.length,
            -segment.row,
            -segment.start_column,
        ),
    )


def _target_segments(solution: list[list[int]], color_count: int) -> list[PaintSegment]:
    if len(solution) != COLOR_GRID_SIZE or any(len(line) != COLOR_GRID_SIZE for line in solution):
        raise ValueError("solution must be a 15x15 matrix")
    segments = []
    for row, line in enumerate(solution):
        column = 0
        while column < COLOR_GRID_SIZE:
            value = int(line[column])
            if value == 0:
                column += 1
                continue
            if not 1 <= value <= color_count:
                raise ValueError(f"solution color is outside palette at row={row} column={column}")
            start_column = column
            while column + 1 < COLOR_GRID_SIZE and int(line[column + 1]) == value:
                column += 1
            segments.append(PaintSegment(value - 1, row, start_column, column))
            column += 1
    if not segments:
        raise ValueError("expected at least one paint segment")
    return segments


def _parse_cells(raw, geometry: GridGeometry) -> list[tuple[int, int]]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("erase_cells must be a list of row/column pairs")
    cells = []
    seen = set()
    for item in raw:
        if isinstance(item, dict):
            row, column = item.get("row"), item.get("column")
        else:
            try:
                row, column = item
            except (TypeError, ValueError):
                raise ValueError(f"invalid erase cell: {item!r}") from None
        try:
            row, column = int(row), int(column)
        except (TypeError, ValueError):
            raise ValueError(f"invalid erase cell: {item!r}") from None
        if not 0 <= row < geometry.rows or not 0 <= column < geometry.columns:
            raise ValueError(f"erase cell is outside board: row={row} column={column}")
        if (row, column) not in seen:
            seen.add((row, column))
            cells.append((row, column))
    return cells


def _cell_roi(geometry: GridGeometry, row: int, column: int) -> tuple[int, int, int, int]:
    margin_x = geometry.cell_width * 0.22
    margin_y = geometry.cell_height * 0.22
    left = round(geometry.x_lines[column] + margin_x)
    right = round(geometry.x_lines[column + 1] - margin_x)
    top = round(geometry.y_lines[row] + margin_y)
    bottom = round(geometry.y_lines[row + 1] - margin_y)
    return left, top, max(1, right - left), max(1, bottom - top)


def _estimate_background_rgb(
    image: np.ndarray,
    geometry: GridGeometry,
) -> tuple[float, float, float]:
    medians = []
    for row in range(geometry.rows):
        for column in range(geometry.columns):
            left, top, width, height = _cell_roi(geometry, row, column)
            patch = image[top : top + height, left : left + width, :3]
            if patch.size:
                medians.append(np.median(patch.reshape(-1, 3), axis=0))
    if not medians:
        raise ValueError("unable to estimate board background from empty cells")
    values = np.asarray(medians, dtype=np.float32)
    distances = np.linalg.norm(values[:, np.newaxis, :] - values[np.newaxis, :, :], axis=2)
    neighbor_counts = (distances <= 32.0).sum(axis=1)
    representative = values[int(np.argmax(neighbor_counts))]
    inliers = values[np.linalg.norm(values - representative, axis=1) <= 40.0]
    if not inliers.size:
        inliers = values
    return tuple(float(value) for value in np.median(inliers, axis=0))


def _bright_threshold(background_rgb: tuple[float, float, float]) -> float:
    return max(170.0, min(240.0, float(np.mean(background_rgb)) + 55.0))


def _x_marker_score(patch: np.ndarray, background_rgb: tuple[float, float, float]) -> float:
    pixels = patch[:, :, :3].astype(np.float32)
    height, width = pixels.shape[:2]
    if height < 3 or width < 3:
        return 0.0

    background = np.asarray(background_rgb, dtype=np.float32)
    luminance = pixels.mean(axis=2)
    background_luminance = float(background.mean())
    background_distance = np.linalg.norm(pixels - background, axis=2)
    edge_margin = max(1, round(min(height, width) * X_MARKER_INTERIOR_MARGIN_RATIO))
    interior = np.zeros((height, width), dtype=bool)
    interior[edge_margin : height - edge_margin, edge_margin : width - edge_margin] = True
    foreground = (
        (background_distance > BACKGROUND_PIXEL_TOLERANCE)
        & (luminance - background_luminance >= X_MARKER_MIN_LUMINANCE_DELTA)
        & interior
    )
    if not foreground.any():
        return 0.0

    y, x = np.indices((height, width), dtype=np.float32)
    scale = max(1.0, height - 1.0)
    diagonal_width = max(1.0, min(height, width) * 0.12)
    main_diagonal = (
        np.abs(x - y * (width - 1.0) / scale) <= diagonal_width
    ) & interior
    anti_diagonal = (
        np.abs(x - (width - 1.0 - y * (width - 1.0) / scale)) <= diagonal_width
    ) & interior
    main_support = float((foreground & main_diagonal).sum() / max(1, main_diagonal.sum()))
    anti_support = float((foreground & anti_diagonal).sum() / max(1, anti_diagonal.sum()))
    off_diagonal = interior & ~(main_diagonal | anti_diagonal)
    off_support = float((foreground & off_diagonal).sum() / max(1, off_diagonal.sum()))
    diagonal_support = min(main_support, anti_support)
    return diagonal_support * max(0.0, 1.0 - off_support)


def _classify_cell(
    image: np.ndarray,
    geometry: GridGeometry,
    row: int,
    column: int,
    palette: list[PaletteColor],
    background_rgb: tuple[float, float, float] | None = None,
) -> CellObservation:
    left, top, width, height = _cell_roi(geometry, row, column)
    patch = image[top : top + height, left : left + width, :3].astype(np.float32)
    if not patch.size:
        raise ValueError(f"empty board cell at row={row} column={column}")
    if background_rgb is None:
        background_rgb = _estimate_background_rgb(image, geometry)
    pixels = patch.reshape(-1, 3)
    median = np.median(pixels, axis=0)
    targets = np.asarray([color.rgb for color in palette], dtype=np.float32)
    if len(palette):
        distances = np.linalg.norm(pixels[:, np.newaxis, :] - targets[np.newaxis, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        counts = np.bincount(nearest, minlength=len(palette))
        best_index = int(np.argmax(counts))
        match_ratio = float(counts[best_index] / max(1, pixels.shape[0]))
        median_distances = np.linalg.norm(targets - median, axis=1)
        distance = float(np.min(median_distances))
    else:
        best_index = 0
        match_ratio = 0.0
        distance = float("inf")
    background = np.asarray(background_rgb, dtype=np.float32)
    background_distances = np.linalg.norm(pixels - background, axis=1)
    background_ratio = float(
        np.mean(background_distances <= BACKGROUND_PIXEL_TOLERANCE)
    )
    background_distance = float(np.linalg.norm(median - background))
    x_score = _x_marker_score(patch, background_rgb)
    palette_hit = match_ratio >= CELL_MATCH_RATIO and distance <= CELL_COLOR_TOLERANCE
    solid_palette_hit = (
        palette_hit
        and match_ratio >= PALETTE_SOLID_MATCH_RATIO
        and distance <= PALETTE_SOLID_DISTANCE
        and background_ratio < BACKGROUND_MIN_SUPPORT
    )
    bright_ratio = float(np.mean(pixels.mean(axis=1) >= _bright_threshold(background_rgb)))
    median_luminance_delta = float(np.mean(median) - np.mean(background))
    highlight = (
        not palette_hit
        and bright_ratio >= HIGHLIGHT_MIN_BRIGHT_RATIO
        and median_luminance_delta >= HIGHLIGHT_MIN_LUMINANCE_DELTA
    )
    x_candidate = (
        x_score >= X_MARKER_MIN_DIAGONAL_SUPPORT
        and bright_ratio >= X_MARKER_MIN_BRIGHT_RATIO
    )
    if highlight:
        state = "selected"
        palette_index = None
    elif solid_palette_hit:
        state = "palette"
        palette_index = best_index
    elif background_ratio >= BACKGROUND_MIN_SUPPORT and not x_candidate:
        state = "background"
        palette_index = None
    elif x_candidate:
        state = "x_marker"
        palette_index = None
    elif palette_hit:
        state = "palette"
        palette_index = best_index
    else:
        state = "unknown"
        palette_index = None
    center = geometry.center(row, column)
    return CellObservation(
        row=row,
        column=column,
        center=center,
        rgb=tuple(int(round(value)) for value in median),
        state=state,
        palette_index=palette_index,
        match_ratio=match_ratio,
        distance=distance,
        background_ratio=background_ratio,
        background_distance=background_distance,
        x_marker_score=x_score,
    )


def _observe_board(
    image: np.ndarray,
    geometry: GridGeometry,
    palette: list[PaletteColor],
    background_rgb: tuple[float, float, float] | None = None,
) -> list[list[CellObservation]]:
    if background_rgb is None:
        background_rgb = _estimate_background_rgb(image, geometry)
    return [
        [
            _classify_cell(
                image,
                geometry,
                row,
                column,
                palette,
                background_rgb,
            )
            for column in range(geometry.columns)
        ]
        for row in range(geometry.rows)
    ]


def _observe_cells(
    image: np.ndarray,
    geometry: GridGeometry,
    palette: list[PaletteColor],
    background_rgb: tuple[float, float, float],
    cells: tuple[tuple[int, int], ...] | list[tuple[int, int]] | set[tuple[int, int]],
) -> dict[tuple[int, int], CellObservation]:
    return {
        (row, column): _classify_cell(
            image,
            geometry,
            row,
            column,
            palette,
            background_rgb,
        )
        for row, column in cells
    }


def _board_report(board: list[list[CellObservation]]) -> list[list[dict]]:
    return [[cell.to_dict() for cell in line] for line in board]


def _rgb_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)))


def _baseline_mismatches(board: list[list[CellObservation]]) -> list[dict]:
    return [
        cell.to_dict()
        for line in board
        for cell in line
        if cell.state not in BASELINE_ALLOWED_STATES
    ]


def _board_x_constraints(
    board: list[list[CellObservation]],
) -> tuple[set[int], set[int], list[int], list[int]]:
    if not board or not board[0] or any(len(line) != len(board[0]) for line in board):
        raise ValueError("board observations are not rectangular")
    row_x_counts = [sum(cell.state == "x_marker" for cell in line) for line in board]
    column_x_counts = [
        sum(board[row][column].state == "x_marker" for row in range(len(board)))
        for column in range(len(board[0]))
    ]
    empty_rows = {
        row for row, count in enumerate(row_x_counts) if count == len(board[row])
    }
    empty_columns = {
        column
        for column, count in enumerate(column_x_counts)
        if count == len(board)
    }
    uncertain = [
        (row, column)
        for row, line in enumerate(board)
        for column, cell in enumerate(line)
        if cell.state == "x_marker"
        and row not in empty_rows
        and column not in empty_columns
    ]
    if uncertain:
        sample = ", ".join(f"({row},{column})" for row, column in uncertain[:8])
        raise ValueError(f"partial X markers are uncertain: {sample}")
    return empty_rows, empty_columns, row_x_counts, column_x_counts


def _preserves_baseline_cell(
    actual: CellObservation,
    expected: CellObservation,
) -> bool:
    if expected.state == "background":
        return (
            actual.state == "background"
            and _rgb_distance(actual.rgb, expected.rgb) <= BASELINE_RGB_TOLERANCE
        )
    if expected.state == "x_marker":
        return actual.state == "x_marker"
    return False


def _segments_without_trial(
    segments: list[PaintSegment], trial_segment: PaintSegment
) -> list[PaintSegment]:
    trial_index = next(
        (
            index
            for index, segment in enumerate(segments)
            if segment is trial_segment
        ),
        None,
    )
    if trial_index is None:
        raise ValueError("trial segment is not part of the paint segment list")
    return [
        segment
        for index, segment in enumerate(segments)
        if index != trial_index
    ]

def _verify_trial_impact(
    board: list[list[CellObservation]],
    baseline: list[list[CellObservation]],
    solution: list[list[int]],
    target: tuple[int, int] | set[tuple[int, int]],
    already_painted: set[tuple[int, int]] | None = None,
) -> list[dict]:
    targets = {target} if isinstance(target, tuple) else set(target)
    painted = set(already_painted or ())
    mismatches = []
    for row, line in enumerate(board):
        for column, actual in enumerate(line):
            if (row, column) in targets:
                continue
            expected = baseline[row][column]
            expected_value = int(solution[row][column])
            if (row, column) in painted:
                expected_color = expected_value - 1 if expected_value else None
                valid = (
                    expected_color is not None
                    and actual.state == "palette"
                    and actual.palette_index == expected_color
                )
            elif expected_value == 0:
                # The game may automatically mark solved-empty cells with X.
                valid = actual.state in BASELINE_ALLOWED_STATES
            else:
                # A future non-empty target must not be auto-marked before painting.
                valid = _preserves_baseline_cell(actual, expected)
            if not valid:
                mismatches.append(
                    {
                        "row": row,
                        "column": column,
                        "expected_solution": expected_value,
                        "expected": expected.to_dict(),
                        "actual": actual.to_dict(),
                    }
                )
    return mismatches


def _verify_segment_targets(
    board: list[list[CellObservation]] | dict[tuple[int, int], CellObservation],
    segment: PaintSegment,
) -> list[dict]:
    mismatches = []
    for row, column in segment.cells:
        actual = board[(row, column)] if isinstance(board, dict) else board[row][column]
        if actual.state == "palette" and actual.palette_index == segment.color:
            continue
        mismatches.append(
            {
                "row": row,
                "column": column,
                "expected_color": segment.color,
                "actual": actual.to_dict(),
            }
        )
    return mismatches


def _verify_painted_segment(
    board: list[list[CellObservation]],
    baseline: list[list[CellObservation]],
    solution: list[list[int]],
    segment: PaintSegment,
    already_painted: set[tuple[int, int]] | None = None,
) -> list[dict]:
    return [
        *_verify_segment_targets(board, segment),
        *_verify_trial_impact(
            board,
            baseline,
            solution,
            set(segment.cells),
            already_painted=already_painted,
        ),
    ]


def _verify_board(
    board: list[list[CellObservation]],
    baseline: list[list[CellObservation]],
    solution: list[list[int]],
) -> list[dict]:
    mismatches = []
    for row, line in enumerate(board):
        for column, actual in enumerate(line):
            expected_value = int(solution[row][column])
            expected_color = expected_value - 1 if expected_value else None
            if expected_color is None:
                # Empty solution cells may remain blank or receive the game's X marker.
                valid = actual.state in BASELINE_ALLOWED_STATES
            else:
                valid = actual.state == "palette" and actual.palette_index == expected_color
            if not valid:
                mismatches.append(
                    {
                        "row": row,
                        "column": column,
                        "expected_color": expected_color,
                        "actual": actual.to_dict(),
                        "baseline": baseline[row][column].to_dict(),
                    }
                )
    return mismatches


def _capture(context: Context, expected_resolution: tuple[int, int]) -> np.ndarray:
    image = context.tasker.controller.post_screencap().wait().get()
    if image is None:
        raise RuntimeError("screenshot returned no image")
    image = np.asarray(image)
    actual = (int(image.shape[1]), int(image.shape[0]))
    if actual != expected_resolution:
        raise ValueError(f"unexpected screenshot resolution: expected={expected_resolution}, actual={actual}")
    return image


def _capture_stable(
    context: Context,
    expected_resolution: tuple[int, int],
    delay: float,
) -> tuple[np.ndarray, float]:
    first = _capture(context, expected_resolution)
    time.sleep(delay)
    second = _capture(context, expected_resolution)
    delta = float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))
    if delta > STABILITY_MEAN_DELTA:
        raise RuntimeError(f"screen is not stable before paint: mean_delta={delta:.3f}")
    return second, delta


def _click_cell(context: Context, geometry: GridGeometry, row: int, column: int) -> None:
    x, y = _validate_cell_center(geometry, row, column)
    result = context.tasker.controller.post_click(x, y).wait()
    if not result.succeeded:
        raise RuntimeError(f"cell click failed: row={row} column={column}")


def _validate_cell_center(geometry: GridGeometry, row: int, column: int) -> tuple[int, int]:
    x, y = geometry.center(row, column)
    if not (
        geometry.x_lines[0] < x < geometry.x_lines[-1]
        and geometry.y_lines[0] < y < geometry.y_lines[-1]
    ):
        raise ValueError(f"cell center is outside board: row={row} column={column} center=({x},{y})")
    return x, y


def _segment_rgb_delta(
    previous: np.ndarray,
    current: np.ndarray,
    geometry: GridGeometry,
    segment: PaintSegment,
) -> float:
    previous_values = []
    current_values = []
    for row, column in segment.cells:
        left, top, width, height = _cell_roi(geometry, row, column)
        previous_patch = previous[top : top + height, left : left + width, :3]
        current_patch = current[top : top + height, left : left + width, :3]
        if not previous_patch.size or not current_patch.size:
            return float("inf")
        previous_values.append(np.median(previous_patch.reshape(-1, 3), axis=0))
        current_values.append(np.median(current_patch.reshape(-1, 3), axis=0))
    return float(
        np.mean(
            np.abs(
                np.asarray(previous_values, dtype=np.float32)
                - np.asarray(current_values, dtype=np.float32)
            )
        )
    )


def _segment_observation(
    board: list[list[CellObservation]] | dict[tuple[int, int], CellObservation],
    segment: PaintSegment,
    delay_ms: int,
    stable_delta: float | None,
    stable: bool,
) -> dict:
    cells = [
        (board[(row, column)] if isinstance(board, dict) else board[row][column]).to_dict()
        for row, column in segment.cells
    ]
    if segment.length == 1:
        observation = dict(cells[0])
        observation.update(
            {
                "expected_color": segment.color,
                "delay_ms": delay_ms,
                "stable_delta": None if stable_delta is None else round(stable_delta, 6),
                "stable": stable,
            }
        )
        return observation
    return {
        **segment.to_dict(),
        "delay_ms": delay_ms,
        "stable_delta": None if stable_delta is None else round(stable_delta, 6),
        "stable": stable,
        "cells": cells,
    }


def _wait_until_segment_painted(
    context: Context,
    expected_resolution: tuple[int, int],
    geometry: GridGeometry,
    palette: list[PaletteColor],
    background_rgb: tuple[float, float, float],
    segment: PaintSegment,
    *,
    initial_delay: float,
    poll_interval: float,
    timeout: float,
) -> tuple[np.ndarray, dict[tuple[int, int], CellObservation], list[dict], bool]:
    started = time.monotonic()
    time.sleep(initial_delay)
    previous_image = None
    previous_valid = False
    observations: list[dict] = []
    while True:
        current = _capture(context, expected_resolution)
        board = _observe_cells(current, geometry, palette, background_rgb, segment.cells)
        target_mismatches = _verify_segment_targets(board, segment)
        valid = not target_mismatches
        stable_delta = (
            None
            if previous_image is None
            else _segment_rgb_delta(previous_image, current, geometry, segment)
        )
        stable = (
            valid
            and previous_valid
            and stable_delta is not None
            and stable_delta <= TARGET_STABILITY_MEAN_DELTA
        )
        delay_ms = round((time.monotonic() - started) * 1000)
        observations.append(
            _segment_observation(board, segment, delay_ms, stable_delta, stable)
        )
        if stable:
            return current, board, observations, True
        if time.monotonic() - started >= timeout:
            return current, board, observations, False
        previous_image = current
        previous_valid = valid
        time.sleep(poll_interval)


def _paint_duration(segment: PaintSegment) -> int:
    if segment.length == 2:
        return 520
    return max(250, 140 + segment.length * 65)


def _paint_action_report(segment: PaintSegment, attempt: int) -> dict:
    return {
        "attempt": attempt,
        "operation": segment.to_dict()["operation"],
        "duration_ms": None if segment.length <= 2 else _paint_duration(segment),
    }


def _observed_state_summary(observations: list[dict]) -> list[dict]:
    summary = []
    for observation in observations:
        cells = observation.get("cells")
        if cells is None:
            cells = [observation]
        summary.append(
            {
                "delay_ms": observation.get("delay_ms"),
                "states": [cell.get("state") for cell in cells],
                "palette_indices": [cell.get("palette_index") for cell in cells],
                "stable": bool(observation.get("stable")),
            }
        )
    return summary


def _wait_until_segment_painted_with_retry(
    context: Context,
    expected_resolution: tuple[int, int],
    geometry: GridGeometry,
    palette: list[PaletteColor],
    background_rgb: tuple[float, float, float],
    segment: PaintSegment,
    *,
    initial_delay: float,
    poll_interval: float,
    timeout: float,
    reference_image: np.ndarray | None = None,
    action_report: dict | None = None,
    fast_initial_delay: float | None = None,
    fast_poll_interval: float | None = None,
    fast_timeout: float | None = None,
) -> tuple[
    np.ndarray,
    dict[tuple[int, int], CellObservation],
    list[dict],
    bool,
    dict,
    dict | None,
    list[dict],
]:
    current, board, observations, stable = _wait_until_segment_painted(
        context,
        expected_resolution,
        geometry,
        palette,
        background_rgb,
        segment,
        initial_delay=initial_delay if fast_initial_delay is None else fast_initial_delay,
        poll_interval=poll_interval if fast_poll_interval is None else fast_poll_interval,
        timeout=timeout if fast_timeout is None else fast_timeout,
    )
    attempt_observations = [
        {
            "attempt": 1,
            "stable": stable,
            "observations": observations,
            "observed_states": _observed_state_summary(observations),
        }
    ]
    if not stable and fast_timeout is not None:
        current, board, fallback_observations, stable = _wait_until_segment_painted(
            context,
            expected_resolution,
            geometry,
            palette,
            background_rgb,
            segment,
            initial_delay=initial_delay,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        observations = observations + fallback_observations
        attempt_observations.append(
            {
                "attempt": 2,
                "stable": stable,
                "fallback": "full_settle",
                "observations": fallback_observations,
                "observed_states": _observed_state_summary(fallback_observations),
            }
        )
    action_attempt = action_report or _paint_action_report(segment, 1)
    action_attempt["status"] = "observed"
    return current, board, observations, stable, action_attempt, None, attempt_observations


def _paint_segment(
    context: Context,
    geometry: GridGeometry,
    segment: PaintSegment,
    *,
    expected_resolution: tuple[int, int] | None = None,
    palette: list[PaletteColor] | None = None,
    selected_palette_index: int | None = None,
    surface_image: np.ndarray | None = None,
    reference_image: np.ndarray | None = None,
    action_report: dict | None = None,
    verified_surface_status: dict | None = None,
) -> dict | None:
    surface_status = None
    if expected_resolution is not None:
        if palette is None or selected_palette_index is None:
            raise ValueError("palette and selected_palette_index are required for a guarded paint action")
        if selected_palette_index != segment.color:
            raise RuntimeError(
                f"selected palette does not match paint segment: "
                f"selected={selected_palette_index} expected={segment.color}"
            )
        if (
            verified_surface_status is not None
            and verified_surface_status.get("safe") is True
            and verified_surface_status.get("grid_detected") is True
            and verified_surface_status.get("palette_editor_detected") is False
            and (
                verified_surface_status.get("selected_palette_index") is None
                or verified_surface_status.get("selected_palette_index") == selected_palette_index
            )
        ):
            surface_status = verified_surface_status
        else:
            surface_status = _require_paint_surface(
                context,
                expected_resolution,
                geometry,
                palette,
                expected_selected=selected_palette_index,
                segment=segment,
                image=surface_image,
                reference_image=reference_image,
            )
    if segment.start_column < 0 or segment.end_column >= geometry.columns:
        raise ValueError(f"paint segment is outside board: {segment}")
    start_x, start_y = _validate_cell_center(geometry, segment.row, segment.start_column)
    end_x, end_y = _validate_cell_center(geometry, segment.row, segment.end_column)
    if start_y != end_y:
        raise ValueError(f"paint segment is not horizontal: {segment}")
    if segment.length == 1:
        result = context.tasker.controller.post_click(start_x, start_y).wait()
        if action_report is not None:
            action_report["clicks"] = [
                {
                    "row": segment.row,
                    "column": segment.start_column,
                    "center": [start_x, start_y],
                    "succeeded": bool(result.succeeded),
                }
            ]
    elif segment.length == 2:
        clicks = []
        for row, column in segment.cells:
            x, y = _validate_cell_center(geometry, row, column)
            result = context.tasker.controller.post_click(x, y).wait()
            clicks.append(
                {
                    "row": row,
                    "column": column,
                    "center": [x, y],
                    "succeeded": bool(result.succeeded),
                }
            )
            if not result.succeeded:
                if action_report is not None:
                    action_report["clicks"] = clicks
                raise RuntimeError(f"paint segment action failed: {segment}")
        if action_report is not None:
            action_report["clicks"] = clicks
        result = None
    else:
        duration = _paint_duration(segment)
        result = context.tasker.controller.post_swipe(
            start_x,
            start_y,
            end_x,
            end_y,
            duration,
        ).wait()
    if result is not None and not result.succeeded:
        raise RuntimeError(f"paint segment action failed: {segment}")
    return surface_status


def _cell_mean_delta(
    before: np.ndarray,
    after: np.ndarray,
    geometry: GridGeometry,
    row: int,
    column: int,
) -> float:
    top, left, bottom, right = _cell_roi(geometry, row, column)
    before_patch = before[top:bottom, left:right, :3].astype(np.float32)
    after_patch = after[top:bottom, left:right, :3].astype(np.float32)
    if before_patch.shape != after_patch.shape or not before_patch.size:
        return float("inf")
    return float(np.mean(np.abs(before_patch - after_patch)))


def _safe_missing_cells(
    before: np.ndarray,
    after: np.ndarray,
    geometry: GridGeometry,
    palette: list[PaletteColor],
    background_rgb: tuple[float, float, float],
    segment: PaintSegment,
    board: dict[tuple[int, int], CellObservation],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    expected_contrast = float(
        np.linalg.norm(
            np.asarray(palette[segment.color].rgb, dtype=np.float32)
            - np.asarray(background_rgb, dtype=np.float32)
        )
    )
    missing = []
    blocked = []
    for row, column in segment.cells:
        observation = board[(row, column)]
        if observation.state == "palette" and observation.palette_index == segment.color:
            continue
        unchanged = _cell_mean_delta(before, after, geometry, row, column) <= 2.5
        if observation.state == "background" and unchanged and expected_contrast >= 25.0:
            missing.append((row, column))
        else:
            blocked.append((row, column))
    return missing, blocked


def _repair_unchanged_segment_cells(
    context: Context,
    expected_resolution: tuple[int, int],
    geometry: GridGeometry,
    palette: list[PaletteColor],
    background_rgb: tuple[float, float, float],
    segment: PaintSegment,
    board: dict[tuple[int, int], CellObservation],
    *,
    selected_palette_index: int,
    before_image: np.ndarray,
    after_image: np.ndarray,
    reference_image: np.ndarray | None,
) -> dict:
    missing, blocked = _safe_missing_cells(
        before_image,
        after_image,
        geometry,
        palette,
        background_rgb,
        segment,
        board,
    )
    result = {
        "attempted": False,
        "missing_cells": [[row, column] for row, column in missing],
        "blocked_cells": [[row, column] for row, column in blocked],
        "clicked_cells": [],
        "status": "skipped",
    }
    if not missing or blocked:
        return result
    if len(missing) != 1:
        result["status"] = "multiple_missing_cells"
        return result
    if selected_palette_index != segment.color:
        result["status"] = "palette_mismatch"
        return result

    row, column = missing[0]
    settled_before = _capture(context, expected_resolution)
    settled_board = _observe_cells(
        settled_before, geometry, palette, background_rgb, segment.cells
    )
    if any(
        observation.state not in {"background", "palette"}
        or (
            observation.state == "palette"
            and observation.palette_index != segment.color
        )
        for observation in settled_board.values()
    ) or settled_board[(row, column)].state != "background":
        result["status"] = "post_action_state_changed"
        return result

    time.sleep(0.08)
    settled_after = _capture(context, expected_resolution)
    settled_after_board = _observe_cells(
        settled_after, geometry, palette, background_rgb, segment.cells
    )
    if any(
        observation.state not in {"background", "palette"}
        or (
            observation.state == "palette"
            and observation.palette_index != segment.color
        )
        for observation in settled_after_board.values()
    ) or settled_after_board[(row, column)].state != "background":
        result["status"] = "post_action_state_changed"
        return result
    if _cell_mean_delta(before_image, settled_after, geometry, row, column) > 2.5:
        result["status"] = "pixels_changed"
        return result

    _require_paint_surface(
        context,
        expected_resolution,
        geometry,
        palette,
        expected_selected=segment.color,
        segment=PaintSegment(segment.color, row, column, column),
        reference_image=reference_image,
    )
    _click_cell(context, geometry, row, column)
    result["attempted"] = True
    result["clicked_cells"].append([row, column])
    result["status"] = "clicked_single_unchanged_cell"
    return result

def _recognition_box(detail) -> tuple[int, int, int, int] | None:
    if detail is None:
        return None
    results = [
        getattr(detail, "best_result", None),
        *getattr(detail, "all_results", []),
        *getattr(detail, "filtered_results", []),
    ]
    for result in results:
        if result is None:
            continue
        box = getattr(result, "box", None)
        if isinstance(box, (tuple, list)) and len(box) >= 4:
            return tuple(int(value) for value in box[:4])
        values = [getattr(box, name, None) for name in ("x", "y", "w", "h")]
        if all(value is not None for value in values):
            return tuple(int(value) for value in values)
        values = [getattr(box, name, None) for name in ("x", "y", "width", "height")]
        if all(value is not None for value in values):
            return tuple(int(value) for value in values)
    return None


def _template_hit(
    context: Context,
    image: np.ndarray,
    template: str,
    threshold: float = 0.75,
) -> bool:
    detail = context.run_recognition_direct(
        JRecognitionType.TemplateMatch,
        JTemplateMatch(
            [template],
            roi=(0, 0, image.shape[1], image.shape[0]),
            threshold=[threshold],
        ),
        image,
    )
    return bool(detail is not None and detail.hit)


def _palette_editor_detected(image: np.ndarray) -> bool:
    height, width = image.shape[:2]
    if height < 2 or width < 2:
        return False
    white = np.all(image[..., :3] >= EDITOR_MODAL_WHITE_THRESHOLD, axis=2)
    scan_height = max(1, int(round(height * EDITOR_MODAL_EDGE_SCAN_RATIO)))
    min_edge_width = int(round(width * EDITOR_MODAL_MIN_EDGE_WIDTH_RATIO))

    def longest_run(row: np.ndarray) -> tuple[int, int, int]:
        transitions = np.diff(np.r_[False, row, False].astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        if not len(starts):
            return 0, 0, 0
        index = int(np.argmax(ends - starts))
        return int(ends[index] - starts[index]), int(starts[index]), int(ends[index])

    top_edge = max((longest_run(row) for row in white[:scan_height]), default=(0, 0, 0))
    bottom_edge = max((longest_run(row) for row in white[-scan_height:]), default=(0, 0, 0))
    if top_edge[0] < min_edge_width or bottom_edge[0] < min_edge_width:
        return False
    if top_edge[1] == 0 or bottom_edge[1] == 0 or top_edge[2] == width or bottom_edge[2] == width:
        return False
    left = max(top_edge[1], bottom_edge[1])
    right = min(top_edge[2], bottom_edge[2])
    return right - left >= min_edge_width * 0.75


def _grid_matches(expected: GridGeometry, actual: GridGeometry | None) -> bool:
    if actual is None:
        return False
    if actual.rows != expected.rows or actual.columns != expected.columns:
        return False
    if len(actual.x_lines) != len(expected.x_lines) or len(actual.y_lines) != len(expected.y_lines):
        return False
    tolerance = max(8.0, min(expected.cell_width, expected.cell_height) * 0.25)
    return bool(
        np.max(np.abs(np.asarray(actual.x_lines) - np.asarray(expected.x_lines))) <= tolerance
        and np.max(np.abs(np.asarray(actual.y_lines) - np.asarray(expected.y_lines))) <= tolerance
    )


def _reference_frame_status(
    image: np.ndarray,
    geometry: GridGeometry,
    reference_image: np.ndarray | None,
) -> dict:
    if reference_image is None or image.shape != reference_image.shape:
        return {
            "available": False,
            "detected": False,
            "best_offset": None,
            "best_mean_delta": None,
            "second_mean_delta": None,
            "margin": None,
        }
    height, width = image.shape[:2]
    left = round(geometry.x_lines[0])
    right = round(geometry.x_lines[-1])
    top = round(geometry.y_lines[0])
    bottom = round(geometry.y_lines[-1])
    strip = REFERENCE_FRAME_STRIP_WIDTH
    if not (strip <= left < right < width - strip and strip <= top < bottom < height - strip):
        return {
            "available": False,
            "detected": False,
            "best_offset": None,
            "best_mean_delta": None,
            "second_mean_delta": None,
            "margin": None,
        }
    mask = np.zeros((height, width), dtype=bool)
    mask[top - strip : top + 3, left - strip : right + strip + 1] = True
    mask[bottom - 2 : bottom + strip + 1, left - strip : right + strip + 1] = True
    mask[top - strip : bottom + strip + 1, left - strip : left + 3] = True
    mask[top - strip : bottom + strip + 1, right - 2 : right + strip + 1] = True
    current = image[:, :, :3].astype(np.float32).mean(axis=2)
    reference = reference_image[:, :, :3].astype(np.float32).mean(axis=2)
    ys, xs = np.nonzero(mask)
    if len(ys) > 1024:
        sample = np.linspace(0, len(ys) - 1, 1024, dtype=np.int64)
        ys, xs = ys[sample], xs[sample]
    delta = np.abs(current[ys, xs] - reference[ys, xs])
    scores = [(float(np.mean(delta)), 0, 0)]
    if not scores:
        return {
            "available": True,
            "detected": False,
            "best_offset": None,
            "best_mean_delta": None,
            "second_mean_delta": None,
            "margin": None,
        }
    best = scores[0]
    second = float("inf")
    margin = float("inf")
    position_ok = max(abs(best[1]), abs(best[2])) <= REFERENCE_FRAME_MAX_OFFSET
    detected = (
        position_ok
        and best[0] <= REFERENCE_FRAME_MAX_MEAN_DELTA
        and margin >= REFERENCE_FRAME_MIN_MARGIN
    )
    return {
        "available": True,
        "detected": detected,
        "best_offset": [best[1], best[2]],
        "best_mean_delta": round(best[0], 6),
        "second_mean_delta": None,
        "margin": None,
        "position_ok": position_ok,
        "search_radius": REFERENCE_FRAME_SEARCH_RADIUS,
        "strip_width": strip,
    }


def _grid_position_status(
    image: np.ndarray,
    geometry: GridGeometry,
    reference_image: np.ndarray | None,
) -> dict:
    height, width = image.shape[:2]
    x_lines = [round(value) for value in geometry.x_lines]
    y_lines = [round(value) for value in geometry.y_lines]
    board_left, board_right = x_lines[0], x_lines[-1]
    board_top, board_bottom = y_lines[0], y_lines[-1]
    valid_bounds = (
        0 <= board_left < board_right < width
        and 0 <= board_top < board_bottom < height
        and len(x_lines) == geometry.columns + 1
        and len(y_lines) == geometry.rows + 1
    )
    if not valid_bounds:
        return {
            "detected": False,
            "mean_line_delta": None,
            "max_line_delta": None,
            "reference_cached": reference_image is not None,
            "check": "anchored_grid_lines",
        }
    gray = image[:, :, :3].astype(np.float32).mean(axis=2)

    def line_support(position: int, *, vertical: bool) -> tuple[float, int]:
        supports: list[tuple[float, int]] = []
        for offset in range(-ANCHORED_GRID_LINE_SEARCH_RADIUS, ANCHORED_GRID_LINE_SEARCH_RADIUS + 1):
            candidate = position + offset
            if vertical:
                candidate = min(max(0, candidate), width - 1)
                strip = gray[board_top : board_bottom + 1, max(0, candidate - 1) : min(width, candidate + 2)]
                support = float(np.mean(np.any(strip <= ANCHORED_GRID_DARK_THRESHOLD, axis=1)))
            else:
                candidate = min(max(0, candidate), height - 1)
                strip = gray[max(0, candidate - 1) : min(height, candidate + 2), board_left : board_right + 1]
                support = float(np.mean(np.any(strip <= ANCHORED_GRID_DARK_THRESHOLD, axis=0)))
            supports.append((support, offset))
        return max(supports, key=lambda item: item[0])

    x_supports = [line_support(x, vertical=True) for x in x_lines]
    y_supports = [line_support(y, vertical=False) for y in y_lines]
    support_values = [support for support, _ in [*x_supports, *y_supports]]
    minimum_support = min(support_values) if support_values else 0.0
    mean_support = float(np.mean(support_values)) if support_values else 0.0
    low_support_values = [
        support for support in support_values if support < ANCHORED_GRID_MIN_LINE_SUPPORT
    ]
    low_support_count = len(low_support_values)
    line_count = len(support_values)
    strong_line_ratio = (
        (line_count - low_support_count) / line_count if line_count else 0.0
    )
    x_low_support_count = sum(
        support < ANCHORED_GRID_MIN_LINE_SUPPORT for support, _ in x_supports
    )
    y_low_support_count = sum(
        support < ANCHORED_GRID_MIN_LINE_SUPPORT for support, _ in y_supports
    )
    strict_detected = (
        minimum_support >= ANCHORED_GRID_MIN_LINE_SUPPORT
        and mean_support >= ANCHORED_GRID_MIN_MEAN_SUPPORT
    )
    limited_coverage_detected = (
        minimum_support >= ANCHORED_GRID_EXCEPTION_MIN_SUPPORT
        and mean_support >= ANCHORED_GRID_EXCEPTION_MIN_MEAN_SUPPORT
        and low_support_count <= ANCHORED_GRID_EXCEPTION_MAX_LOW_SUPPORT_LINES
        and low_support_count / line_count <= ANCHORED_GRID_EXCEPTION_MAX_LOW_SUPPORT_RATIO
        and strong_line_ratio >= ANCHORED_GRID_EXCEPTION_MIN_STRONG_LINE_RATIO
        and x_low_support_count <= ANCHORED_GRID_EXCEPTION_MAX_LOW_SUPPORT_LINES_PER_AXIS
        and y_low_support_count <= ANCHORED_GRID_EXCEPTION_MAX_LOW_SUPPORT_LINES_PER_AXIS
    )
    reference_frame = _reference_frame_status(image, geometry, reference_image)
    reference_detected = bool(
        reference_frame["detected"]
        and mean_support >= 0.70
        and strong_line_ratio >= 0.50
    )
    return {
        "detected": strict_detected or limited_coverage_detected or reference_detected,
        "mean_line_delta": None,
        "max_line_delta": None,
        "reference_cached": reference_image is not None,
        "check": "anchored_grid_lines",
        "line_search_radius": ANCHORED_GRID_LINE_SEARCH_RADIUS,
        "minimum_line_support": round(minimum_support, 6),
        "mean_line_support": round(mean_support, 6),
        "low_support_line_count": low_support_count,
        "low_support_line_ratio": round(low_support_count / line_count, 6),
        "strong_line_ratio": round(strong_line_ratio, 6),
        "x_low_support_line_count": x_low_support_count,
        "y_low_support_line_count": y_low_support_count,
        "limited_coverage_exception": limited_coverage_detected,
        "reference_frame": reference_frame,
        "reference_frame_exception": reference_detected and not strict_detected and not limited_coverage_detected,
        "x_line_support": [round(support, 6) for support, _ in x_supports],
        "y_line_support": [round(support, 6) for support, _ in y_supports],
        "x_line_offsets": [offset for _, offset in x_supports],
        "y_line_offsets": [offset for _, offset in y_supports],
    }


def _palette_ring(image: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    x, y = center
    top = max(0, y - 40)
    bottom = min(image.shape[0], y + 40)
    left = max(0, x - 150)
    right = min(image.shape[1], x + 151)
    patch = image[top:bottom, left:right, :3]
    if patch.shape[0] != 80 or patch.shape[1] != 301:
        return np.empty((0, 3), dtype=np.uint8)
    mask = np.ones((80, 301), dtype=bool)
    mask[6:-6, 6:-6] = False
    return patch[mask]


def _palette_surface_status(image: np.ndarray, palette: list[PaletteColor]) -> dict:
    buttons = []
    for index, color in enumerate(palette):
        x, y = color.center
        top = max(0, y - 6)
        bottom = min(image.shape[0], y + 7)
        left = max(0, x - 6)
        right = min(image.shape[1], x + 7)
        patch = image[top:bottom, left:right, :3]
        if not patch.size or patch.shape[0] != 13 or patch.shape[1] != 13:
            actual_rgb = None
            distance = float("inf")
        else:
            actual_rgb = tuple(int(round(value)) for value in np.median(patch.reshape(-1, 3), axis=0))
            distance = _rgb_distance(actual_rgb, tuple(int(round(value)) for value in color.rgb))
        ring = _palette_ring(image, color.center)
        selected_border_score = (
            float(np.mean(np.all(ring >= 220, axis=1))) if ring.size else 0.0
        )
        buttons.append(
            {
                "palette_index": index,
                "center": [int(x), int(y)],
                "expected_rgb": [int(round(value)) for value in color.rgb],
                "actual_rgb": None if actual_rgb is None else list(actual_rgb),
                "rgb_distance": None if not np.isfinite(distance) else round(distance, 6),
                "rgb_valid": bool(distance <= PALETTE_RGB_TOLERANCE),
                "selected_border_score": round(selected_border_score, 6),
            }
        )
    selected_candidates = [
        item["palette_index"]
        for item in buttons
        if item["selected_border_score"] >= PALETTE_SELECTED_BORDER_THRESHOLD
    ]
    return {
        "complete": bool(buttons) and all(item["rgb_valid"] for item in buttons),
        "buttons": buttons,
        "selected_candidates": selected_candidates,
        "selected_border_detected": len(selected_candidates) == 1,
        "selected_palette_index": (
            selected_candidates[0] if len(selected_candidates) == 1 else None
        ),
    }


def _paint_surface_status(
    image: np.ndarray,
    geometry: GridGeometry,
    palette: list[PaletteColor],
    reference_image: np.ndarray | None = None,
) -> dict:
    editor_detected = _palette_editor_detected(image)
    grid_status = _grid_position_status(image, geometry, reference_image)
    palette_status = _palette_surface_status(image, palette)
    return {
        "safe": not editor_detected and grid_status["detected"] and palette_status["complete"],
        "palette_editor_detected": editor_detected,
        "grid_detected": grid_status["detected"],
        "grid": _geometry_report(geometry),
        "grid_detection_method": "anchored" if grid_status["detected"] else "unconfirmed",
        "grid_position": grid_status,
        "palette_complete": palette_status["complete"],
        "palette_buttons": palette_status["buttons"],
        "selected_candidates": palette_status["selected_candidates"],
        "selected_border_detected": palette_status["selected_border_detected"],
        "selected_palette_index": palette_status["selected_palette_index"],
    }


def _require_paint_surface(
    context: Context,
    expected_resolution: tuple[int, int],
    geometry: GridGeometry,
    palette: list[PaletteColor],
    *,
    expected_selected: int | None = None,
    segment: PaintSegment | None = None,
    image: np.ndarray | None = None,
    reference_image: np.ndarray | None = None,
) -> dict:
    image = _capture(context, expected_resolution) if image is None else image
    status = _paint_surface_status(image, geometry, palette, reference_image)
    if not status["grid_detected"]:
        fallback_geometry = detect_color_grid(image)
        status["grid_full_fallback"] = {
            "attempted": True,
            "detected": _geometry_report(fallback_geometry),
            "matches_anchor": _grid_matches(geometry, fallback_geometry),
        }
        if status["grid_full_fallback"]["matches_anchor"]:
            status["grid_detected"] = True
            status["grid"] = _geometry_report(fallback_geometry)
            status["grid_detection_method"] = "full_detect_fallback"
    else:
        status["grid_full_fallback"] = {"attempted": False}
    status["safe"] = (
        not status["palette_editor_detected"]
        and status["grid_detected"]
        and status["palette_complete"]
    )
    status["selected_palette_before_action"] = expected_selected
    if segment is not None:
        try:
            for row, column in segment.cells:
                _validate_cell_center(geometry, row, column)
            status["target_centers_valid"] = True
        except Exception:
            status["target_centers_valid"] = False
    else:
        status["target_centers_valid"] = True
    reasons = []
    if status["palette_editor_detected"]:
        reasons.append("palette editor detected")
    if not status["grid_detected"]:
        reasons.append("15x15 grid was not confirmed")
    if not status["palette_complete"]:
        reasons.append("palette buttons were not confirmed")
    if not status["target_centers_valid"]:
        reasons.append("target center is outside the confirmed board")
    selected_candidates = status["selected_candidates"]
    if expected_selected is not None and selected_candidates and expected_selected not in selected_candidates:
        reasons.append(
            f"selected palette mismatch: expected={expected_selected} actual={selected_candidates}"
        )
    if reasons:
        raise PaintSurfaceError("unsafe paint surface: " + "; ".join(reasons), status)
    return status


def _select_eraser(
    context: Context,
    expected_resolution: tuple[int, int],
    *,
    pause: float = 0.12,
) -> np.ndarray:
    image = _capture(context, expected_resolution)
    detail = context.run_recognition_direct(
        JRecognitionType.TemplateMatch,
        JTemplateMatch(
            ["抹掉颜色.png"],
            roi=(0, 0, image.shape[1], image.shape[0]),
            threshold=[0.75],
        ),
        image,
    )
    box = _recognition_box(detail)
    if box is None:
        raise RuntimeError("eraser tool was not recognized")
    x, y, width, height = box
    result = context.tasker.controller.post_click(x + width // 2, y + height // 2).wait()
    if not result.succeeded:
        raise RuntimeError("eraser tool selection failed")
    time.sleep(pause)
    selected = _capture(context, expected_resolution)
    if not _template_hit(context, selected, "抹掉颜色.png"):
        raise RuntimeError("eraser tool state was not confirmed after selection")
    return selected


def _select_palette(
    context: Context,
    palette: list[PaletteColor],
    color: int,
    *,
    expected_resolution: tuple[int, int],
    geometry: GridGeometry,
    selected_palette_index: int | None,
    surface_image: np.ndarray | None = None,
    reference_image: np.ndarray | None = None,
    verify_paint_tool: bool = False,
    pause: float = 0.12,
    fast_path: bool = False,
) -> tuple[int, dict, np.ndarray]:
    if not 0 <= color < len(palette):
        raise ValueError(f"palette index is outside palette: {color}")
    before_image = _capture(context, expected_resolution) if surface_image is None else surface_image
    before_surface = _require_paint_surface(
        context,
        expected_resolution,
        geometry,
        palette,
        expected_selected=selected_palette_index,
        image=before_image,
        reference_image=reference_image,
    )
    target_rgb = [int(round(value)) for value in palette[color].rgb]
    if selected_palette_index == color:
        if verify_paint_tool and not _template_hit(context, before_image, "涂色.png"):
            raise RuntimeError("paint tool state was not confirmed for already selected palette")
        return color, {
            "palette_index": color,
            "center": [int(palette[color].center[0]), int(palette[color].center[1])],
            "rgb": target_rgb,
            "selected_palette_before_action": selected_palette_index,
            "selected_palette_after_action": color,
            "clicked": False,
            "selected": True,
            "selection_confirmed": True,
            "palette_editor_detected": before_surface["palette_editor_detected"],
            "grid_detected": before_surface["grid_detected"],
            "palette_complete": before_surface["palette_complete"],
            "selected_border_detected": before_surface["selected_border_detected"],
            "selected_border_index": before_surface["selected_palette_index"],
            "palette_surface": before_surface["palette_buttons"],
            "surface_status": before_surface,
        }, before_image
    result = context.tasker.controller.post_click(*palette[color].center).wait()
    if not result.succeeded:
        raise RuntimeError(f"palette selection failed: color={color}")
    time.sleep(max(0.08 if fast_path else 0.15, min(0.3, pause)))
    selected_image = _capture(context, expected_resolution)
    if verify_paint_tool and not _template_hit(context, selected_image, "涂色.png"):
        raise RuntimeError("paint tool state was not confirmed after palette selection")
    after_surface = _require_paint_surface(
        context,
        expected_resolution,
        geometry,
        palette,
        expected_selected=color,
        image=selected_image,
        reference_image=reference_image,
    )
    selected_border_index = after_surface["selected_palette_index"]
    return color, {
        "palette_index": color,
        "center": [int(palette[color].center[0]), int(palette[color].center[1])],
        "rgb": target_rgb,
        "selected_palette_before_action": selected_palette_index,
        "selected_palette_after_action": color,
        "clicked": True,
        "selected": True,
        "selection_confirmed": (
            not after_surface["selected_candidates"]
            or selected_border_index == color
        ),
        "palette_editor_detected": after_surface["palette_editor_detected"],
        "grid_detected": after_surface["grid_detected"],
        "palette_complete": after_surface["palette_complete"],
        "selected_border_detected": after_surface["selected_border_detected"],
        "selected_border_index": selected_border_index,
        "palette_surface": after_surface["palette_buttons"],
        "surface_status": after_surface,
    }, selected_image


def _wait_until_cell_blank(
    context: Context,
    expected_resolution: tuple[int, int],
    geometry: GridGeometry,
    palette: list[PaletteColor],
    background_rgb: tuple[float, float, float],
    row: int,
    column: int,
    *,
    initial_delay: float,
    poll_interval: float,
    timeout: float,
) -> tuple[np.ndarray, CellObservation, list[dict], bool]:
    started = time.monotonic()
    time.sleep(initial_delay)
    previous_image = None
    previous_valid = False
    observations = []
    while True:
        current = _capture(context, expected_resolution)
        board = _observe_cells(
            current,
            geometry,
            palette,
            background_rgb,
            ((row, column),),
        )
        cell = board[(row, column)]
        valid = cell.state in BASELINE_ALLOWED_STATES
        delta = (
            None
            if previous_image is None
            else _segment_rgb_delta(
                previous_image,
                current,
                geometry,
                PaintSegment(0, row, column, column),
            )
        )
        stable = valid and previous_valid and delta is not None and delta <= TARGET_STABILITY_MEAN_DELTA
        observations.append(
            {
                **cell.to_dict(),
                "delay_ms": round((time.monotonic() - started) * 1000),
                "stable_delta": None if delta is None else round(delta, 6),
                "stable": stable,
            }
        )
        if stable:
            return current, cell, observations, True
        if time.monotonic() - started >= timeout:
            return current, cell, observations, False
        previous_image = current
        previous_valid = valid
        time.sleep(poll_interval)


def _erase_cells(
    context: Context,
    expected_resolution: tuple[int, int],
    geometry: GridGeometry,
    palette: list[PaletteColor],
    background_rgb: tuple[float, float, float],
    cells: list[tuple[int, int]],
    restore_color: int,
    *,
    pause: float = 0.12,
    reference_image: np.ndarray | None = None,
    initial_delay: float = DEFAULT_PAINT_SETTLE_INITIAL,
    poll_interval: float = DEFAULT_PAINT_SETTLE_POLL,
    timeout: float = DEFAULT_PAINT_SETTLE_TIMEOUT,
) -> tuple[np.ndarray, int, dict]:
    if not cells:
        raise ValueError("erase_cells must contain at least one cell")
    _select_eraser(context, expected_resolution, pause=pause)
    for row, column in cells:
        _require_paint_surface(
            context,
            expected_resolution,
            geometry,
            palette,
            segment=PaintSegment(0, row, column, column),
            reference_image=reference_image,
        )
        _click_cell(context, geometry, row, column)
        image, cell, _, stable = _wait_until_cell_blank(
            context,
            expected_resolution,
            geometry,
            palette,
            background_rgb,
            row,
            column,
            initial_delay=initial_delay,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        if not stable or cell.state not in BASELINE_ALLOWED_STATES:
            raise RuntimeError(
                f"erased cell did not return to blank state: row={row} column={column} "
                f"state={cell.state}"
            )
    restored_color, selection, _ = _select_palette(
        context,
        palette,
        restore_color,
        expected_resolution=expected_resolution,
        geometry=geometry,
        selected_palette_index=None,
        reference_image=reference_image,
        verify_paint_tool=True,
        pause=pause,
    )
    return image, restored_color, selection


def _completion_bar_detected(image: np.ndarray) -> bool:
    height, width = image.shape[:2]
    region = image[max(0, int(height * 0.78)) :, :, :3]
    cyan = (region[..., 1] >= 115) & (region[..., 2] >= 115) & (region[..., 0] <= 165)
    dark = np.mean(region, axis=2) <= 55
    row_cyan = np.mean(cyan, axis=1)
    row_dark = np.mean(dark, axis=1)
    cyan_rows = np.flatnonzero(row_cyan >= 0.45)
    if len(cyan_rows) < 2:
        return False
    for top in cyan_rows:
        for bottom in cyan_rows[cyan_rows > top + 20]:
            interior = row_dark[top + 8 : bottom - 7]
            if len(interior) and float(np.mean(interior >= 0.65)) >= 0.6:
                return True
    return False


def _completion_status(context: Context, image: np.ndarray) -> dict:
    template_hit = False
    template_error = None
    try:
        detail = context.run_recognition_direct(
            JRecognitionType.TemplateMatch,
            JTemplateMatch(
                ["恭喜过关了.png"],
                roi=(0, 0, image.shape[1], image.shape[0]),
                threshold=[0.8],
            ),
            image,
        )
        template_hit = bool(detail is not None and detail.hit)
    except Exception as exc:
        template_error = str(exc)
    continue_hit = False
    continue_template_error = None
    try:
        detail = context.run_recognition_direct(
            JRecognitionType.TemplateMatch,
            JTemplateMatch(
                ["继续.png"],
                roi=(0, 0, image.shape[1], image.shape[0]),
                threshold=[0.75],
            ),
            image,
        )
        continue_hit = bool(detail is not None and detail.hit)
    except Exception as exc:
        continue_template_error = str(exc)
    ocr_texts = []
    ocr_error = None
    try:
        detail = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(expected=[], roi=(0, 0, image.shape[1], image.shape[0]), threshold=0.2),
            image,
        )
        ocr_texts = [str(getattr(result, "text", "")) for result in _ocr_results(detail)]
    except Exception as exc:
        ocr_error = str(exc)
    ocr_hit = any(
        "恭喜" in text and ("过关" in text or "完成" in text)
        for text in ocr_texts
    )
    bar_hit = _completion_bar_detected(image)
    return {
        "detected": template_hit or continue_hit or ocr_hit or bar_hit,
        "completion_bar_hit": bar_hit,
        "template_hit": template_hit,
        "continue_hit": continue_hit,
        "ocr_hit": ocr_hit,
        "ocr_texts": ocr_texts,
        "template_error": template_error,
        "continue_template_error": continue_template_error,
        "ocr_error": ocr_error,
    }


def _initial_report(param: dict) -> dict:
    return {
        "status": "failure",
        "mode": "paint_test",
        "phase": "gate",
        "allow_paint": param.get("allow_paint"),
        "reason": None,
        "resolution": None,
        "grid": None,
        "palette": [],
        "background_rgb": None,
        "clues": None,
        "baseline_cells": None,
        "baseline_row_x_counts": None,
        "baseline_column_x_counts": None,
        "empty_rows": [],
        "empty_columns": [],
        "solution": None,
        "solution_hash": None,
        "expected_cells": 0,
        "expected_segments": 0,
        "segments": None,
        "paint_settle": None,
        "erase_cells": [],
        "erase_screenshot": None,
        "erase_selection": None,
        "paint_operations": [],
        "painted_cells": 0,
        "verified_cells": 0,
        "trial": None,
        "mismatches": [],
        "before_screenshot": None,
        "before_report": None,
        "trial_screenshot": None,
        "after_screenshot": None,
        "failure_screenshot": None,
        "completion_screenshot": None,
        "completion": None,
        "stability_mean_delta": None,
        "after_stability_mean_delta": None,
        "answer_fallback": None,
    }


def _read_answer_view(
    context: Context,
    image: np.ndarray,
    expected_resolution: tuple[int, int],
    *,
    answer_delay: float,
    pause: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    detail = context.run_recognition_direct(
        JRecognitionType.TemplateMatch,
        JTemplateMatch(
            ["查看答案.png"],
            roi=(0, 0, image.shape[1], image.shape[0]),
            threshold=[0.82],
        ),
        image,
    )
    box = _recognition_box(detail)
    if box is None:
        raise AnswerBoardError("查看答案按钮未识别")
    x, y, width, height = box
    result = context.tasker.controller.post_click(x + width // 2, y + height // 2).wait()
    if not result.succeeded:
        raise AnswerBoardError("查看答案按钮点击失败")
    time.sleep(answer_delay)
    answer_image = _capture(context, expected_resolution)
    result = context.tasker.controller.post_click_key(4).wait()
    if not result.succeeded:
        raise AnswerBoardError("答案页返回动作失败")
    time.sleep(pause)
    return answer_image, _capture(context, expected_resolution), box


@AgentServer.custom_action("ColorNonogram1080pPaintTest")
class ColorNonogram1080pPaintTest(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        param = _load_param(argv.custom_action_param)
        success_node = str(param.get("success_node", DEFAULT_SUCCESS_NODE))
        failure_node = str(param.get("failure_node", DEFAULT_FAILURE_NODE))
        report = _initial_report(param)
        phase = "gate"
        image = None
        reference_surface_image = None
        baseline = None
        geometry = None
        palette: list[PaletteColor] = []
        solution = None
        background_rgb: tuple[float, float, float] | None = None
        empty_rows: set[int] = set()
        empty_columns: set[int] = set()
        painted: set[tuple[int, int]] = set()
        selected_palette_index: int | None = None
        run_image_paths: list[str] = []

        def save_run_image(image_value: np.ndarray, prefix: str, timestamp: str | None = None) -> str:
            path = _save_image(image_value, prefix, timestamp)
            run_image_paths.append(path)
            return path

        try:
            if param.get("allow_paint") is not True:
                raise PermissionError("paint test is disabled; explicit allow_paint=true is required")
            if str(param.get("mode", "paint_test")) != "paint_test":
                raise ValueError("unsupported paint test mode")
            expected_resolution = tuple(
                int(value) for value in param.get("expected_resolution", (1920, 1080))
            )
            pause = max(0.08, min(2.0, float(param.get("pause", 0.12))))
            answer_delay = max(0.5, min(3.0, float(param.get("answer_page_delay", 1.0))))
            stability_delay = max(0.05, min(1.0, float(param.get("stability_delay", 0.15))))
            settle_initial = max(
                0.1,
                min(0.5, float(param.get("paint_settle_initial", DEFAULT_PAINT_SETTLE_INITIAL))),
            )
            settle_poll = max(
                0.05,
                min(0.4, float(param.get("paint_settle_poll", DEFAULT_PAINT_SETTLE_POLL))),
            )
            settle_timeout = max(
                0.5,
                min(3.0, float(param.get("paint_settle_timeout", DEFAULT_PAINT_SETTLE_TIMEOUT))),
            )
            fast_paint = param.get("fast_paint") is True
            fast_settle_initial = max(
                0.05,
                min(settle_initial, float(param.get("fast_paint_settle_initial", DEFAULT_FAST_PAINT_SETTLE_INITIAL))),
            )
            fast_settle_poll = max(
                0.03,
                min(settle_poll, float(param.get("fast_paint_settle_poll", DEFAULT_FAST_PAINT_SETTLE_POLL))),
            )
            fast_settle_timeout = max(
                fast_settle_poll * 2,
                min(settle_timeout, float(param.get("fast_paint_settle_timeout", DEFAULT_FAST_PAINT_SETTLE_TIMEOUT))),
            )
            report["paint_settle"] = {
                "initial_ms": round(settle_initial * 1000),
                "poll_ms": round(settle_poll * 1000),
                "timeout_ms": round(settle_timeout * 1000),
                "target_stability_mean_delta": TARGET_STABILITY_MEAN_DELTA,
                "fast_path": fast_paint,
                "fast_initial_ms": round(fast_settle_initial * 1000),
                "fast_poll_ms": round(fast_settle_poll * 1000),
                "fast_timeout_ms": round(fast_settle_timeout * 1000),
            }
            phase = "before_capture"
            image, stable_delta = _capture_stable(context, expected_resolution, stability_delay)
            reference_surface_image = image
            report["stability_mean_delta"] = round(stable_delta, 6)
            report["resolution"] = [int(image.shape[1]), int(image.shape[0])]
            before_stamp = _stamp()
            report["before_screenshot"] = save_run_image(image, "paint_before", before_stamp)
            phase = "before_grid"
            geometry = detect_color_grid(image)
            report["grid"] = _geometry_report(geometry)
            if geometry is None or geometry.rows != COLOR_GRID_SIZE or geometry.columns != COLOR_GRID_SIZE:
                raise ValueError("unable to locate a stable 15x15 colored grid before paint")
            phase = "before_palette"
            palette = _palette_colors(image)
            report["palette"] = _palette_report(palette)
            if not 2 <= len(palette) <= 12:
                raise ValueError(f"expected 2-12 palette colors, found {len(palette)}")
            if not _verify_clue_color_match(context, image, geometry, palette):
                raise ValueError("Maa ColorMatch did not verify the paint baseline")
            phase = "before_baseline"
            background_rgb = _estimate_background_rgb(image, geometry)
            report["background_rgb"] = [int(round(value)) for value in background_rgb]
            baseline = _observe_board(image, geometry, palette, background_rgb)
            report["baseline_cells"] = _board_report(baseline)
            empty_rows, empty_columns, row_x_counts, column_x_counts = _board_x_constraints(baseline)
            report["baseline_row_x_counts"] = row_x_counts
            report["baseline_column_x_counts"] = column_x_counts
            report["empty_rows"] = sorted(empty_rows)
            report["empty_columns"] = sorted(empty_columns)
            erase_cells = _parse_cells(param.get("erase_cells"), geometry)
            report["erase_cells"] = [
                {"row": row, "column": column} for row, column in erase_cells
            ]
            if erase_cells:
                erase_set = set(erase_cells)
                invalid = [
                    cell
                    for line in baseline
                    for cell in line
                    if cell.state not in BASELINE_ALLOWED_STATES
                    and (cell.row, cell.column) not in erase_set
                ]
                if invalid:
                    report["mismatches"] = [cell.to_dict() for cell in invalid]
                    raise ValueError("board has non-blank cells outside explicit erase_cells")
                unknown_erase_cells = [
                    baseline[row][column].to_dict()
                    for row, column in erase_cells
                    if baseline[row][column].state == "unknown"
                ]
                if unknown_erase_cells:
                    report["mismatches"] = unknown_erase_cells
                    raise ValueError("cannot erase cells with unknown pre-erase state")
                phase = "erase"
                restore_color = int(param.get("restore_color", 0))
                if not 0 <= restore_color < len(palette):
                    raise ValueError(f"restore_color is outside palette: {restore_color}")
                image, selected_palette_index, report["erase_selection"] = _erase_cells(
                    context,
                    expected_resolution,
                    geometry,
                    palette,
                    background_rgb,
                    erase_cells,
                    restore_color,
                    pause=pause,
                    reference_image=reference_surface_image,
                    initial_delay=settle_initial,
                    poll_interval=settle_poll,
                    timeout=settle_timeout,
                )
                image = _capture(context, expected_resolution)
                reference_surface_image = image
                report["erase_screenshot"] = save_run_image(image, "paint_erase")
                baseline = _observe_board(image, geometry, palette, background_rgb)
                report["baseline_cells"] = _board_report(baseline)
                empty_rows, empty_columns, row_x_counts, column_x_counts = _board_x_constraints(baseline)
                report["baseline_row_x_counts"] = row_x_counts
                report["baseline_column_x_counts"] = column_x_counts
                report["empty_rows"] = sorted(empty_rows)
                report["empty_columns"] = sorted(empty_columns)
            baseline_mismatches = _baseline_mismatches(baseline)
            if baseline_mismatches:
                report["mismatches"] = baseline_mismatches
                raise ValueError("board is not blank before paint")
            phase = "before_solve"
            report["before_report"] = _write_report(report, "paint_before", before_stamp)
            rows = None
            columns = None
            try:
                extracted = _colored_clues(context, image, geometry, palette)
                if extracted is None:
                    raise ValueError("colored clues are incomplete before paint")
                rows, columns, _, _, row_cache, column_cache, _, _ = extracted
                rows, columns, solution = _solve_with_candidate_recovery(
                    context,
                    image,
                    geometry,
                    palette,
                    rows,
                    columns,
                    row_cache,
                    column_cache,
                    empty_rows=empty_rows,
                    empty_columns=empty_columns,
                )
            except Exception as solve_error:
                phase = "answer_fallback"
                answer_image, image, answer_button = _read_answer_view(
                    context,
                    image,
                    expected_resolution,
                    answer_delay=answer_delay,
                    pause=pause,
                )
                answer_screenshot = save_run_image(answer_image, "paint_answer")
                report["answer_fallback"] = {
                    "status": "captured",
                    "reason": str(solve_error),
                    "button": list(answer_button),
                    "screenshot": answer_screenshot,
                }
                answer_board = extract_answer_board(answer_image)
                if answer_board.color_count != len(palette):
                    raise AnswerBoardError(
                        f"答案图颜色数量与当前调色板不一致: answer={answer_board.color_count} palette={len(palette)}"
                    )
                mapping = map_answer_colors(
                    answer_board.colors,
                    palette,
                    maximum_distance=75.0,
                    minimum_margin=10.0,
                )
                answer_grid = remap_answer_grid(answer_board.grid, mapping)
                solution = [
                    [0 if value == EMPTY_CELL else int(value) + 1 for value in line]
                    for line in answer_grid
                ]
                report["answer_fallback"] = {
                    "status": "ready_to_paint",
                    "reason": str(solve_error),
                    "button": list(answer_button),
                    "screenshot": answer_screenshot,
                    "grid_roi": list(answer_board.grid_roi),
                    "answer_color_count": answer_board.color_count,
                    "palette_mapping": list(mapping),
                }
            if rows is not None and columns is not None:
                report["clues"] = {
                    "rows": _clues_report(rows),
                    "columns": _clues_report(columns),
                }
            report["solution"] = [[int(value) for value in line] for line in solution]
            report["solution_hash"] = _solution_hash(report["solution"])
            targets = _target_cells(report["solution"], len(palette))
            segments = _target_segments(report["solution"], len(palette))
            report["expected_cells"] = len(targets)
            report["expected_segments"] = len(segments)
            report["segments"] = [segment.to_dict() for segment in segments]
            for _, row, column in targets:
                _validate_cell_center(geometry, row, column)
            report["before_report"] = _write_report(report, "paint_before", before_stamp)
            trial_segment = _trial_segment(segments, palette, background_rgb)
            report["trial"] = {
                **trial_segment.to_dict(),
                "expected_color": trial_segment.color,
                "column": trial_segment.start_column if trial_segment.length == 1 else None,
                "status": "pending",
                "stable": False,
                "timeout_ms": round(settle_timeout * 1000),
                "observations": [],
                "selected_palette_before_action": selected_palette_index,
                "palette_rgb": [int(round(value)) for value in palette[trial_segment.color].rgb],
                "palette_editor_detected": None,
                "grid_detected": None,
                "palette_selection": None,
                "pre_action": None,
                "action_attempt": None,
                "retry_attempt": None,
                "attempt_observations": [],
                "observed_states": [],
            }
            phase = "trial_select"
            try:
                selected_palette_index, palette_selection, selection_image = _select_palette(
                    context,
                    palette,
                    trial_segment.color,
                    expected_resolution=expected_resolution,
                    geometry=geometry,
                    selected_palette_index=selected_palette_index,
                    surface_image=image,
                    reference_image=reference_surface_image,
                    pause=pause,
                    fast_path=fast_paint,
                )
                report["trial"]["palette_selection"] = palette_selection
                report["trial"]["palette_editor_detected"] = palette_selection[
                    "palette_editor_detected"
                ]
                report["trial"]["grid_detected"] = palette_selection["grid_detected"]
            except PaintSurfaceError as exc:
                report["trial"]["palette_selection"] = {"surface": exc.status}
                report["trial"]["palette_editor_detected"] = exc.status[
                    "palette_editor_detected"
                ]
                report["trial"]["grid_detected"] = exc.status["grid_detected"]
                raise
            phase = "trial_action"
            trial_action_report = _paint_action_report(trial_segment, 1)
            try:
                pre_action = _paint_segment(
                    context,
                    geometry,
                    trial_segment,
                    expected_resolution=expected_resolution,
                    palette=palette,
                    selected_palette_index=selected_palette_index,
                    surface_image=selection_image,
                    reference_image=reference_surface_image,
                    action_report=trial_action_report,
                    verified_surface_status=palette_selection["surface_status"],
                )
                report["trial"]["pre_action"] = pre_action
                report["trial"]["palette_editor_detected"] = pre_action[
                    "palette_editor_detected"
                ]
                report["trial"]["grid_detected"] = pre_action["grid_detected"]
            except PaintSurfaceError as exc:
                report["trial"]["pre_action"] = exc.status
                report["trial"]["palette_editor_detected"] = exc.status[
                    "palette_editor_detected"
                ]
                report["trial"]["grid_detected"] = exc.status["grid_detected"]
                raise
            phase = "trial_wait"
            (
                trial_image,
                trial_board,
                trial_observations,
                trial_stable,
                trial_action_attempt,
                trial_retry_attempt,
                trial_attempt_observations,
            ) = _wait_until_segment_painted_with_retry(
                context,
                expected_resolution,
                geometry,
                palette,
                background_rgb,
                trial_segment,
                initial_delay=settle_initial,
                poll_interval=settle_poll,
                timeout=settle_timeout,
                reference_image=reference_surface_image,
                action_report=trial_action_report,
                fast_initial_delay=fast_settle_initial if fast_paint else None,
                fast_poll_interval=fast_settle_poll if fast_paint else None,
                fast_timeout=fast_settle_timeout if fast_paint else None,
            )
            image = trial_image
            report["trial"]["action_attempt"] = trial_action_attempt
            report["trial"]["retry_attempt"] = trial_retry_attempt
            report["trial"]["attempt_observations"] = trial_attempt_observations
            report["trial"]["observed_states"] = [
                item
                for attempt in trial_attempt_observations
                for item in attempt["observed_states"]
            ]
            report["trial"]["observations"] = trial_observations
            report["trial"]["stable"] = trial_stable
            report["trial_screenshot"] = save_run_image(trial_image, "paint_trial")
            trial_mismatches = _verify_segment_targets(trial_board, trial_segment)
            if not trial_stable:
                report["mismatches"] = _verify_segment_targets(trial_board, trial_segment)
                raise ValueError("paint trial did not reach a stable expected color before timeout")
            if trial_mismatches:
                report["mismatches"] = trial_mismatches
                raise ValueError("paint trial verification failed")
            report["trial"]["status"] = "success"
            painted.update(trial_segment.cells)
            report["painted_cells"] = len(painted)
            phase = "paint_cells"
            for segment in _segments_without_trial(segments, trial_segment):
                operation = segment.to_dict()
                operation.update(
                    {
                        "status": "pending",
                        "stable": False,
                        "timeout_ms": round(settle_timeout * 1000),
                        "observations": [],
                        "selected_palette_before_action": selected_palette_index,
                        "palette_rgb": [int(round(value)) for value in palette[segment.color].rgb],
                        "palette_editor_detected": None,
                        "grid_detected": None,
                        "palette_selection": None,
                        "pre_action": None,
                        "action_attempt": None,
                        "retry_attempt": None,
                        "attempt_observations": [],
                        "observed_states": [],
                    }
                )
                report["paint_operations"].append(operation)
                phase = "paint_select"
                try:
                    selected_palette_index, palette_selection, selection_image = _select_palette(
                        context,
                        palette,
                        segment.color,
                        expected_resolution=expected_resolution,
                        geometry=geometry,
                        selected_palette_index=selected_palette_index,
                        surface_image=image,
                        reference_image=reference_surface_image,
                        pause=pause,
                        fast_path=fast_paint,
                    )
                    operation["palette_selection"] = palette_selection
                    operation["palette_editor_detected"] = palette_selection[
                        "palette_editor_detected"
                    ]
                    operation["grid_detected"] = palette_selection["grid_detected"]
                except PaintSurfaceError as exc:
                    operation["palette_selection"] = {"surface": exc.status}
                    operation["palette_editor_detected"] = exc.status[
                        "palette_editor_detected"
                    ]
                    operation["grid_detected"] = exc.status["grid_detected"]
                    raise
                phase = "paint_action"
                action_report = _paint_action_report(segment, 1)
                try:
                    pre_action = _paint_segment(
                        context,
                        geometry,
                        segment,
                        expected_resolution=expected_resolution,
                        palette=palette,
                        selected_palette_index=selected_palette_index,
                        surface_image=selection_image,
                        reference_image=reference_surface_image,
                        action_report=action_report,
                        verified_surface_status=palette_selection["surface_status"],
                    )
                    operation["pre_action"] = pre_action
                    operation["palette_editor_detected"] = pre_action[
                        "palette_editor_detected"
                    ]
                    operation["grid_detected"] = pre_action["grid_detected"]
                except PaintSurfaceError as exc:
                    operation["pre_action"] = exc.status
                    operation["palette_editor_detected"] = exc.status[
                        "palette_editor_detected"
                    ]
                    operation["grid_detected"] = exc.status["grid_detected"]
                    raise
                phase = "paint_wait"
                (
                    current,
                    current_board,
                    observations,
                    stable,
                    action_attempt,
                    retry_attempt,
                    attempt_observations,
                ) = _wait_until_segment_painted_with_retry(
                    context,
                    expected_resolution,
                    geometry,
                    palette,
                    background_rgb,
                    segment,
                    initial_delay=settle_initial,
                    poll_interval=settle_poll,
                    timeout=settle_timeout,
                    reference_image=reference_surface_image,
                    action_report=action_report,
                    fast_initial_delay=fast_settle_initial if fast_paint else None,
                    fast_poll_interval=fast_settle_poll if fast_paint else None,
                    fast_timeout=fast_settle_timeout if fast_paint else None,
                )
                image = current
                operation["action_attempt"] = action_attempt
                operation["retry_attempt"] = retry_attempt
                operation["attempt_observations"] = attempt_observations
                operation["observed_states"] = [
                    item
                    for attempt in attempt_observations
                    for item in attempt["observed_states"]
                ]
                operation["observations"] = observations
                operation["stable"] = stable
                mismatches = _verify_segment_targets(current_board, segment)
                if not stable:
                    recovery = _repair_unchanged_segment_cells(
                        context,
                        expected_resolution,
                        geometry,
                        palette,
                        background_rgb,
                        segment,
                        current_board,
                        selected_palette_index=selected_palette_index,
                        before_image=selection_image,
                        after_image=current,
                        reference_image=reference_surface_image,
                    )
                    operation["recovery"] = recovery
                    if recovery["status"] == "clicked_single_unchanged_cell":
                        (
                            current,
                            current_board,
                            recovery_observations,
                            stable,
                        ) = _wait_until_segment_painted(
                            context,
                            expected_resolution,
                            geometry,
                            palette,
                            background_rgb,
                            segment,
                            initial_delay=settle_initial,
                            poll_interval=settle_poll,
                            timeout=settle_timeout,
                        )
                        observations.extend(recovery_observations)
                        operation["recovery_observations"] = recovery_observations
                        operation["observed_states"] = _observed_state_summary(observations)
                        mismatches = _verify_segment_targets(current_board, segment)
                    if not stable:
                        report["mismatches"] = _verify_segment_targets(current_board, segment)
                        raise ValueError(
                            f"paint segment did not reach a stable expected color before timeout: {segment}"
                        )
                if mismatches:
                    report["mismatches"] = mismatches
                    raise ValueError(f"paint segment verification failed: {segment}")
                operation["status"] = "success"
                painted.update(segment.cells)
                report["painted_cells"] = len(painted)
            phase = "after_verify"
            after_image, after_delta = _capture_stable(context, expected_resolution, stability_delay)
            image = after_image
            report["after_stability_mean_delta"] = round(after_delta, 6)
            after_stamp = _stamp()
            report["after_screenshot"] = save_run_image(after_image, "paint_after", after_stamp)
            after_board = _observe_board(after_image, geometry, palette, background_rgb)
            mismatches = _verify_board(after_board, baseline, report["solution"])
            report["mismatches"] = mismatches
            report["verified_cells"] = COLOR_GRID_SIZE * COLOR_GRID_SIZE - len(mismatches)
            if mismatches:
                raise ValueError(f"full-board verification failed: mismatches={len(mismatches)}")
            phase = "completion"
            completion_wait = max(0.2, min(3.0, float(param.get("completion_wait", 0.5))))
            time.sleep(completion_wait)
            completion_image = _capture(context, expected_resolution)
            image = completion_image
            report["completion_screenshot"] = save_run_image(completion_image, "paint_completion")
            completion = _completion_status(context, completion_image)
            report["completion"] = completion
            if not completion["detected"]:
                raise ValueError("board verified but completion state was not detected")
            report["status"] = "success"
            report["phase"] = "complete"
            report["reason"] = "painted and verified"
            report["cleaned_success_images"] = _cleanup_success_images(run_image_paths)
            report["after_report"] = _write_report(report, "paint_after", after_stamp)
            context.override_next(argv.node_name, [success_node])
            return True
        except Exception as exc:
            report["status"] = "failure"
            report["phase"] = phase
            report["reason"] = str(exc)
            report["painted_cells"] = len(painted)
            if image is not None and report["failure_screenshot"] is None:
                try:
                    report["failure_screenshot"] = _save_image(image, "paint_failure")
                except Exception:
                    pass
            try:
                report_path = _write_report(report, "paint_failure")
            except Exception as report_error:
                report_path = f"unavailable: {report_error}"
            print(
                f"color nonogram paint test failed status=failure phase={phase} "
                f"reason={exc} report={report_path}",
                flush=True,
            )
            context.override_next(argv.node_name, [failure_node])
            return True
