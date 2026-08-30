from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from color_nonogram_model import ColorNonogramPuzzle, Grid, Line
from color_nonogram_vision import GridCalibration


Rgb = tuple[int, int, int]


@dataclass(frozen=True)
class ClueTruth:
    axis: str
    line: int
    slot: int
    color: int
    value: int
    background_rgb: Rgb


@dataclass(frozen=True)
class BoardTruth:
    source: str
    grid: Grid
    palette_rgb: tuple[Rgb, ...]
    empty_rgb: Rgb
    rows: tuple[Line, ...]
    columns: tuple[Line, ...]
    completed: bool
    reason: str
    filled_cells: int
    high_variance_cells: int

    def puzzle(self) -> ColorNonogramPuzzle:
        if not self.completed:
            raise ValueError(f"board truth is incomplete: {self.reason}")
        puzzle = ColorNonogramPuzzle(self.rows, self.columns)
        puzzle.validate()
        puzzle.validate_color_totals()
        return puzzle

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "completed": self.completed,
            "reason": self.reason,
            "filled_cells": self.filled_cells,
            "high_variance_cells": self.high_variance_cells,
            "empty_rgb": list(self.empty_rgb),
            "palette_rgb": [list(color) for color in self.palette_rgb],
            "grid": [list(row) for row in self.grid],
            "rows": [[list(clue) for clue in line] for line in self.rows],
            "columns": [[list(clue) for clue in line] for line in self.columns],
        }


def extract_board_truth(
    image: np.ndarray,
    calibration: GridCalibration,
    cluster_distance: float = 32.0,
    empty_distance: float = 45.0,
    maximum_patch_std: float = 10.0,
) -> BoardTruth:
    samples = _sample_cells(image, calibration)
    uniform_samples = [sample for sample in samples if sample[1] <= maximum_patch_std]
    clusters = _cluster_colors([sample[0] for sample in uniform_samples], cluster_distance)
    if not clusters:
        return _empty_truth(calibration, "no uniform board cells", len(samples))

    minimum_empty_count = max(3, calibration.size * calibration.size // 20)
    empty_candidates = [cluster for cluster in clusters if cluster["count"] >= minimum_empty_count]
    if not empty_candidates:
        return _empty_truth(calibration, "no dominant empty-cell color", len(samples))
    empty_cluster = max(empty_candidates, key=lambda cluster: _empty_color_score(cluster["mean"]))
    empty_rgb = _rgb(empty_cluster["mean"])

    palette_clusters = [
        cluster
        for cluster in clusters
        if cluster is not empty_cluster
        and _color_distance(cluster["mean"], empty_cluster["mean"]) >= empty_distance
    ]
    palette_clusters.sort(key=lambda cluster: tuple(cluster["mean"]))
    palette_rgb = tuple(_rgb(cluster["mean"]) for cluster in palette_clusters)

    grid_values = []
    for color, patch_std in samples:
        if patch_std > maximum_patch_std:
            grid_values.append(0)
            continue
        nearest = min(clusters, key=lambda cluster: _color_distance(color, cluster["mean"]))
        if nearest is empty_cluster or not any(nearest is cluster for cluster in palette_clusters):
            grid_values.append(0)
        else:
            grid_values.append(next(index for index, cluster in enumerate(palette_clusters) if nearest is cluster) + 1)
    grid = tuple(
        tuple(grid_values[row * calibration.size : (row + 1) * calibration.size])
        for row in range(calibration.size)
    )
    rows, columns = clues_from_grid(grid)
    filled_cells = sum(value > 0 for value in grid_values)
    high_variance_cells = len(samples) - len(uniform_samples)

    reasons = []
    maximum_noisy_cells = max(4, calibration.size * calibration.size // 25)
    if high_variance_cells > maximum_noisy_cells:
        reasons.append("board contains strokes or non-uniform cell markings")
    if filled_cells < calibration.size:
        reasons.append("board has too few solid filled cells")
    if not palette_rgb:
        reasons.append("board has no filled-color palette")
    if len(palette_rgb) > 6:
        reasons.append("board has too many color clusters")
    completed = not reasons
    return BoardTruth(
        source=calibration.source,
        grid=grid,
        palette_rgb=palette_rgb,
        empty_rgb=empty_rgb,
        rows=rows if completed else (),
        columns=columns if completed else (),
        completed=completed,
        reason="completed board" if completed else "; ".join(reasons),
        filled_cells=filled_cells,
        high_variance_cells=high_variance_cells,
    )


def clues_from_grid(grid: Grid) -> tuple[tuple[Line, ...], tuple[Line, ...]]:
    rows = tuple(_line_clues(row) for row in grid)
    width = len(grid[0]) if grid else 0
    columns = tuple(_line_clues(tuple(row[column] for row in grid)) for column in range(width))
    return rows, columns


def iter_clue_truth(truth: BoardTruth) -> tuple[ClueTruth, ...]:
    if not truth.completed:
        return ()
    results = []
    for axis, lines in (("row", truth.rows), ("column", truth.columns)):
        for line_index, clues in enumerate(lines):
            for slot, (color, value) in enumerate(reversed(clues)):
                results.append(
                    ClueTruth(
                        axis=axis,
                        line=line_index,
                        slot=slot,
                        color=color,
                        value=value,
                        background_rgb=truth.palette_rgb[color],
                    )
                )
    return tuple(results)


def _sample_cells(
    image: np.ndarray,
    calibration: GridCalibration,
) -> list[tuple[np.ndarray, float]]:
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("board image must be an RGB image")
    x0, y0, x1, y1 = calibration.grid
    samples = []
    for row in range(calibration.size):
        for column in range(calibration.size):
            left = round(x0 + (column + 0.30) * (x1 - x0) / calibration.size)
            right = round(x0 + (column + 0.70) * (x1 - x0) / calibration.size)
            top = round(y0 + (row + 0.30) * (y1 - y0) / calibration.size)
            bottom = round(y0 + (row + 0.70) * (y1 - y0) / calibration.size)
            patch = image[top:bottom, left:right, :3].astype(np.float32)
            if not patch.size:
                raise ValueError(f"empty board cell patch at row {row}, column {column}")
            samples.append(
                (
                    np.median(patch, axis=(0, 1)),
                    float(np.mean(np.std(patch, axis=(0, 1)))),
                )
            )
    return samples


def _cluster_colors(colors: list[np.ndarray], maximum_distance: float) -> list[dict]:
    clusters = []
    for color in colors:
        nearest = None
        nearest_distance = float("inf")
        for cluster in clusters:
            distance = _color_distance(color, cluster["mean"])
            if distance < nearest_distance:
                nearest = cluster
                nearest_distance = distance
        if nearest is not None and nearest_distance <= maximum_distance:
            nearest["colors"].append(color)
            nearest["mean"] = np.mean(nearest["colors"], axis=0)
            nearest["count"] = len(nearest["colors"])
        else:
            clusters.append({"mean": color.copy(), "colors": [color], "count": 1})
    return clusters


def _line_clues(line: tuple[int, ...]) -> Line:
    clues = []
    index = 0
    while index < len(line):
        color = line[index]
        if color == 0:
            index += 1
            continue
        end = index + 1
        while end < len(line) and line[end] == color:
            end += 1
        clues.append((color - 1, end - index))
        index = end
    return tuple(clues)


def _empty_color_score(color: np.ndarray) -> float:
    return float(np.mean(color) - (np.max(color) - np.min(color)))


def _color_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left.astype(np.float32) - right.astype(np.float32)))


def _rgb(color: np.ndarray) -> Rgb:
    return tuple(int(round(value)) for value in color)


def _empty_truth(calibration: GridCalibration, reason: str, sample_count: int) -> BoardTruth:
    grid = tuple(tuple(0 for _ in range(calibration.size)) for _ in range(calibration.size))
    return BoardTruth(
        source=calibration.source,
        grid=grid,
        palette_rgb=(),
        empty_rgb=(0, 0, 0),
        rows=(),
        columns=(),
        completed=False,
        reason=reason,
        filled_cells=0,
        high_variance_cells=sample_count,
    )
