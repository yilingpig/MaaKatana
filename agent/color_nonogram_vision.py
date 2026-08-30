from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GridCalibration:
    source: str
    grid: tuple[float, float, float, float]
    clue_left: float
    clue_top: float
    size: int = 15

    @property
    def cell_width(self) -> float:
        return (self.grid[2] - self.grid[0]) / self.size

    @property
    def cell_height(self) -> float:
        return (self.grid[3] - self.grid[1]) / self.size


@dataclass(frozen=True)
class ClueCell:
    axis: str
    line: int
    slot: int
    roi: tuple[int, int, int, int]
    background_rgb: tuple[int, int, int]
    foreground_pixels: int
    glyph: np.ndarray


def clue_cell_rois(
    calibration: GridCalibration,
) -> list[tuple[str, int, int, tuple[int, int, int, int]]]:
    x0, y0, _, _ = calibration.grid
    cell_width = calibration.cell_width
    cell_height = calibration.cell_height
    rois = []
    for row in range(calibration.size):
        for slot in range(calibration.size - 1, -1, -1):
            roi = (
                round(x0 - (slot + 1) * cell_width + cell_width * 0.08),
                round(y0 + row * cell_height + cell_height * 0.08),
                max(1, round(cell_width * 0.84)),
                max(1, round(cell_height * 0.84)),
            )
            if roi[0] + roi[2] >= calibration.clue_left:
                rois.append(("row", row, slot, roi))
    for column in range(calibration.size):
        for slot in range(calibration.size - 1, -1, -1):
            roi = (
                round(x0 + column * cell_width + cell_width * 0.08),
                round(y0 - (slot + 1) * cell_height + cell_height * 0.08),
                max(1, round(cell_width * 0.84)),
                max(1, round(cell_height * 0.84)),
            )
            if roi[1] + roi[3] >= calibration.clue_top:
                rois.append(("column", column, slot, roi))
    return rois


def crop_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, roi_width, roi_height = roi
    left = max(0, min(width, x))
    top = max(0, min(height, y))
    right = max(left, min(width, x + roi_width))
    bottom = max(top, min(height, y + roi_height))
    return image[top:bottom, left:right]


def normalize_clue_glyph(
    patch: np.ndarray,
    output_size: int = 24,
) -> tuple[np.ndarray, tuple[int, int, int], int]:
    if patch.ndim != 3 or patch.shape[2] < 3 or not patch.size:
        return np.zeros((output_size, output_size), dtype=np.uint8), (0, 0, 0), 0
    pixels = patch[:, :, :3].astype(np.float32)
    inset_y = max(1, round(pixels.shape[0] * 0.08))
    inset_x = max(1, round(pixels.shape[1] * 0.08))
    inner = pixels[inset_y : pixels.shape[0] - inset_y, inset_x : pixels.shape[1] - inset_x]
    if not inner.size:
        inner = pixels
    border = np.concatenate(
        (
            inner[0],
            inner[-1],
            inner[:, 0],
            inner[:, -1],
        ),
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(inner - background, axis=2)
    threshold = max(18.0, float(np.percentile(distance, 82)) * 0.62)
    mask = distance >= threshold
    mask = _keep_glyph_components(mask)
    foreground_pixels = int(mask.sum())
    if foreground_pixels < 4:
        return np.zeros((output_size, output_size), dtype=np.uint8), tuple(int(round(value)) for value in background), 0
    y_positions, x_positions = np.nonzero(mask)
    glyph = mask[
        y_positions.min() : y_positions.max() + 1,
        x_positions.min() : x_positions.max() + 1,
    ]
    aspect_ratio = glyph.shape[1] / glyph.shape[0]
    if aspect_ratio < 0.25 or aspect_ratio > 4.0:
        return np.zeros((output_size, output_size), dtype=np.uint8), tuple(int(round(value)) for value in background), 0
    available = output_size - 4
    scale = min(available / glyph.shape[1], available / glyph.shape[0])
    resized_width = max(1, round(glyph.shape[1] * scale))
    resized_height = max(1, round(glyph.shape[0] * scale))
    resized = np.asarray(
        Image.fromarray((glyph * 255).astype(np.uint8)).resize(
            (resized_width, resized_height),
            Image.Resampling.NEAREST,
        )
    )
    output = np.zeros((output_size, output_size), dtype=np.uint8)
    left = (output_size - resized_width) // 2
    top = (output_size - resized_height) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output, tuple(int(round(value)) for value in background), foreground_pixels


def extract_clue_cells(
    image: np.ndarray,
    calibration: GridCalibration,
) -> list[ClueCell]:
    cells = []
    for axis, line, slot, roi in clue_cell_rois(calibration):
        patch = crop_roi(image, roi)
        glyph, background, foreground_pixels = normalize_clue_glyph(patch)
        if foreground_pixels == 0:
            continue
        cells.append(
            ClueCell(
                axis=axis,
                line=line,
                slot=slot,
                roi=roi,
                background_rgb=background,
                foreground_pixels=foreground_pixels,
                glyph=glyph,
            )
        )
    return cells


def _keep_glyph_components(mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if not 0 <= next_y < height or not 0 <= next_x < width:
                    continue
                if visited[next_y, next_x] or not mask[next_y, next_x]:
                    continue
                visited[next_y, next_x] = True
                stack.append((next_y, next_x))
        if len(component) >= 2:
            for y, x in component:
                output[y, x] = True
    return output
