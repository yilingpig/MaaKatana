import json
import pathlib
import re
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JColorMatch, JOCR, JRecognitionType

from nonogram_solver import (
    GridGeometry,
    _clip_roi,
    _find_clue_boundary,
    _grid_boundary_transition_score,
    _normalize_clue_text,
    _ocr_results,
    _peak_positions,
    _regular_sequences,
    _to_gray,
)

from color_nonogram_core import (
    line_minimum as _line_minimum,
    runs as _runs,
    solve_puzzle,
    SolutionSearchTimeout,
)
from color_nonogram_model import ColorNonogramPuzzle
from color_nonogram_digits import DigitCandidate, TemplateDigitClassifier, load_digit_templates
from color_nonogram_disambiguation import ClueCandidate, ClueObservation, disambiguate_puzzle
from color_nonogram_vision import normalize_clue_glyph

COLOR_GRID_SIZE = 15


def _project_root(module_path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(module_path).resolve()
    for candidate in (path.parent.parent, path.parent.parent.parent):
        if (candidate / "AGENTS.md").is_file():
            return candidate
    return path.parent.parent


PROJECT_ROOT = _project_root(__file__)
_LIVE_DIGIT_CLASSIFIER: TemplateDigitClassifier | None = None

_COLOR_GRID_MIN_NORMALIZED_SCORE = -2.0
_COLOR_GRID_MIN_STEP_RATIO = 0.025
_COLOR_GRID_MAX_STEP_RATIO = 0.055
_COLOR_GRID_MIN_CLUE_GAP_CELLS = 3.0
_COLOR_GRID_TARGET_PALETTE_GAP_CELLS = 0.85
_COLOR_GRID_MAX_PALETTE_OVERLAP_CELLS = 1.25
_COLOR_GRID_MIN_NO_PALETTE_SCORE = 0.0
_PALETTE_DEDUP_DISTANCE = 12.0


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


@dataclass(frozen=True)
class PaletteColor:
    center: tuple[int, int]
    rgb: tuple[float, float, float]


def _patch_median(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = _clip_roi(roi, image)
    patch = image[y : y + height, x : x + width, :3].astype(np.float32)
    if not patch.size:
        return np.zeros(3, dtype=np.float32)
    return np.median(patch.reshape(-1, 3), axis=0)


def _saturation(pixels: np.ndarray) -> np.ndarray:
    return pixels.max(axis=-1) - pixels.min(axis=-1)


def _palette_band(image: np.ndarray) -> tuple[int, int] | None:
    height, width = image.shape[:2]
    pixels = image[:, :, :3].astype(np.float32)
    dark = pixels.mean(axis=-1) < 65
    wide = dark.sum(axis=1) >= round(width * 0.7)
    borders = []
    start = None
    for row, value in enumerate(np.r_[wide, False]):
        if value and start is None:
            start = row
        elif not value and start is not None:
            borders.append((start, row - 1))
            start = None
    candidates = []
    for index, (_, top) in enumerate(borders[:-1]):
        for bottom, _ in borders[index + 1:]:
            band_height = bottom - top
            if not round(height * 0.035) <= band_height <= round(height * 0.16):
                continue
            band = pixels[top:bottom, :]
            darkness = band.mean(axis=-1) < 65
            active = darkness.sum(axis=0) >= round(band_height * 0.48)
            separator_count = 0
            start_column = None
            for column, value in enumerate(np.r_[active, False]):
                if value and start_column is None:
                    start_column = column
                elif not value and start_column is not None:
                    if column - start_column <= 12:
                        separator_count += 1
                    start_column = None
            colorful = np.mean((band.mean(axis=-1) > 25) & (_saturation(band) > 12))
            if separator_count >= 3 and colorful >= 0.15:
                candidates.append((separator_count, colorful, bottom, top, bottom))
            break
    if not candidates:
        return None
    _, _, _, top, bottom = max(candidates)
    return top, bottom


def _palette_colors(image: np.ndarray) -> list[PaletteColor]:
    height, width = image.shape[:2]
    palette_band = _palette_band(image)
    if palette_band is None:
        top = round(height * 0.90)
        bottom = height
    else:
        top, bottom = palette_band
    band = image[top:bottom, :].astype(np.float32)
    profile = np.median(
        band[round(band.shape[0] * 0.18) : round(band.shape[0] * 0.82)],
        axis=0,
    )
    changes = np.linalg.norm(np.diff(profile, axis=0), axis=1)
    threshold = max(20.0, float(np.percentile(changes, 95)) * 0.25)
    minimum_width = max(20, round(width * 0.025))
    boundary_candidates = [
        index + 1
        for index in range(1, len(changes) - 1)
        if changes[index] >= threshold
        and changes[index] >= changes[index - 1]
        and changes[index] >= changes[index + 1]
    ]
    boundaries = []
    for value in sorted(boundary_candidates, key=lambda item: changes[item - 1], reverse=True):
        if all(abs(value - existing) >= minimum_width for existing in boundaries):
            boundaries.append(value)
    boundaries.sort()
    if boundaries:
        start = boundaries[0] if boundaries[0] <= minimum_width else 0
        end = boundaries[-1] if boundaries[-1] >= width - minimum_width else width - 1
        internal = boundaries[:]
        if start == boundaries[0]:
            internal = internal[1:]
        if end == boundaries[-1]:
            internal = internal[:-1]
        borders = [start, *[value for value in internal if start < value < end], end]
    else:
        borders = [0, width - 1]
    colors = []
    for left_border, right_border in zip(borders, borders[1:]):
        if right_border - left_border < max(26, round(width * 0.045)):
            continue
        left = left_border + max(3, round((right_border - left_border) * 0.16))
        right = right_border - max(3, round((right_border - left_border) * 0.16))
        patch = band[round(band.shape[0] * 0.18) : round(band.shape[0] * 0.82), left:right]
        pixels = patch.reshape(-1, 3)
        if not pixels.size:
            continue
        rgb = tuple(float(value) for value in np.median(pixels, axis=0))
        colors.append(PaletteColor((round((left_border + right_border) / 2), round(top + band.shape[0] / 2)), rgb))
    unique_colors: list[PaletteColor] = []
    for color in colors:
        if any(
            np.linalg.norm(np.asarray(color.rgb) - np.asarray(existing.rgb))
            <= _PALETTE_DEDUP_DISTANCE
            for existing in unique_colors
        ):
            continue
        unique_colors.append(color)
    return unique_colors


def _clue_palette_evidence(
    image: np.ndarray,
    geometry: GridGeometry,
    palette: list[PaletteColor],
) -> bool:
    left = max(0, round(geometry.clue_left or geometry.x_lines[0] - geometry.cell_width * 8))
    top = max(0, round(geometry.clue_top or geometry.y_lines[0] - geometry.cell_height * 8))
    board_left = round(geometry.x_lines[0])
    board_top = round(geometry.y_lines[0])
    board_right = round(geometry.x_lines[-1])
    board_bottom = round(geometry.y_lines[-1])
    regions = [
        image[top:board_bottom, left:board_left],
        image[top:board_top, left:board_right],
    ]
    targets = np.asarray([item.rgb for item in palette], dtype=np.float32)
    for region in regions:
        if not region.size:
            continue
        pixels = region[..., :3].reshape(-1, 3).astype(np.float32)
        distances = np.linalg.norm(pixels[:, None, :] - targets[None, :, :], axis=2)
        if int(np.count_nonzero(np.min(distances, axis=1) <= 45.0)) >= 24:
            return True
    return False


def _verify_clue_color_match(
    context: Context,
    image: np.ndarray,
    geometry: GridGeometry,
    palette: list[PaletteColor],
) -> bool:
    left = round(geometry.clue_left or geometry.x_lines[0] - geometry.cell_width * 8)
    top = round(geometry.clue_top or geometry.y_lines[0] - geometry.cell_height * 8)
    roi = _clip_roi((left, top, round(geometry.x_lines[-1] - left), round(geometry.y_lines[-1] - top)), image)
    # Keep the channel order returned by the screenshot for both local matching
    # and Maa ColorMatch. This avoids an RGB/BGR mismatch between the two paths.
    targets = np.asarray([item.rgb for item in palette], dtype=np.int32)
    tolerance = np.asarray((20, 20, 20), dtype=np.int32)
    detail = context.run_recognition_direct(
        JRecognitionType.ColorMatch,
        JColorMatch(
            lower=np.maximum(0, targets - tolerance).tolist(),
            upper=np.minimum(255, targets + tolerance).tolist(),
            roi=roi,
            method=4,
            count=max(100, round(geometry.cell_width * geometry.cell_height * 0.3)),
            connected=True,
        ),
        image,
    )
    return bool(detail is not None and detail.hit) or _clue_palette_evidence(image, geometry, palette)


def _clue_roi_within_bounds(
    roi: tuple[int, int, int, int],
    image: np.ndarray,
    *,
    clue_left: float | None = None,
    clue_top: float | None = None,
) -> bool:
    x, y, width, height = roi
    image_height, image_width = image.shape[:2]
    if width <= 0 or height <= 0:
        return False
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        return False
    if clue_left is not None and x < round(clue_left):
        return False
    if clue_top is not None and y < round(clue_top):
        return False
    return True


def _color_from_match(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    palette: list[PaletteColor],
    lenient: bool = False,
    background_rgb: tuple[float, float, float] | None = None,
) -> int | None:
    x, y, width, height = _clip_roi(roi, image)
    if not palette or width <= 0 or height <= 0:
        return None
    margin_x = max(2, round(width * 0.14))
    margin_y = max(2, round(height * 0.14))
    inner_roi = _clip_roi(
        (x + margin_x, y + margin_y, width - margin_x * 2, height - margin_y * 2),
        image,
    )
    inner_x, inner_y, inner_width, inner_height = inner_roi
    if inner_width <= 0 or inner_height <= 0:
        return None
    pixels = image[inner_y : inner_y + inner_height, inner_x : inner_x + inner_width, :3].reshape(-1, 3).astype(np.int16)
    targets = np.asarray([item.rgb for item in palette], dtype=np.int16)

    if lenient:
        # 自适应tolerance：考虑调色板与背景的距离，应对低对比度场景
        if background_rgb is not None:
            bg = np.asarray(background_rgb, dtype=np.int16)
            # 计算每个调色板颜色与背景的最小距离
            min_dist_to_bg = np.min(np.sqrt(np.sum((targets - bg)**2, axis=1)))
            # 抗锯齿会产生50%混合色，距离约为原距离的一半
            # tolerance需要覆盖抗锯齿过渡色，设为min_dist的70%
            adaptive_tol = max(50, min(90, int(min_dist_to_bg * 0.7)))
            tolerance = np.asarray((adaptive_tol, adaptive_tol, adaptive_tol), dtype=np.int16)
        else:
            tolerance = np.asarray((60, 60, 60), dtype=np.int16)
        match_ratio = 0.10
    else:
        tolerance = np.asarray((20, 20, 20), dtype=np.int16)
        match_ratio = 0.22

    matched = np.all(np.abs(pixels[:, np.newaxis, :] - targets[np.newaxis, :, :]) <= tolerance, axis=2)
    counts = matched.sum(axis=0)
    minimum = max(18, round(pixels.shape[0] * match_ratio))
    candidates = np.flatnonzero(counts >= minimum)

    # 如果没有候选，使用fallback策略：找最接近的颜色
    if not candidates.size and lenient:
        # 计算每个像素到每个目标颜色的距离
        distances = np.sqrt(np.sum((pixels[:, np.newaxis, :] - targets[np.newaxis, :, :])**2, axis=2))
        # 对每个目标颜色，找到最近的像素距离的中位数
        median_distances = np.median(distances, axis=0)
        best_idx = int(np.argmin(median_distances))
        second_best_dist = np.partition(median_distances, 1)[1] if len(median_distances) > 1 else np.inf
        # 只有当最佳颜色显著优于第二最佳时才接受（差异>20%）
        if median_distances[best_idx] < second_best_dist * 0.8:
            return best_idx
        return None

    if not candidates.size:
        return None
    return int(candidates[np.argmax(counts[candidates])])


def _color_grid_consistency(image: np.ndarray, x_lines: tuple[float, ...], y_lines: tuple[float, ...]) -> float:
    medians = []
    for row in range(COLOR_GRID_SIZE):
        for column in range(COLOR_GRID_SIZE):
            left = round(x_lines[column] + (x_lines[column + 1] - x_lines[column]) * 0.35)
            right = round(x_lines[column] + (x_lines[column + 1] - x_lines[column]) * 0.65)
            top = round(y_lines[row] + (y_lines[row + 1] - y_lines[row]) * 0.35)
            bottom = round(y_lines[row] + (y_lines[row + 1] - y_lines[row]) * 0.65)
            patch = image[top:bottom, left:right, :3]
            if patch.size:
                medians.append(np.median(patch.reshape(-1, 3), axis=0))
    if not medians:
        return 1e6
    values = np.asarray(medians)
    return float(np.mean(np.linalg.norm(values - np.median(values, axis=0), axis=1)))


def _select_color_grid_candidate(
    candidates: list[tuple[float, float, float, tuple[float, ...], tuple[float, ...]]],
    palette_band: tuple[int, int] | None,
) -> tuple[float, float, float, tuple[float, ...], tuple[float, ...]] | None:
    palette_safe_candidates = [
        candidate
        for candidate in candidates
        if candidate[2] <= _COLOR_GRID_MAX_PALETTE_OVERLAP_CELLS
    ]
    if not palette_safe_candidates:
        return None
    if palette_band is not None:
        return min(palette_safe_candidates, key=lambda item: (item[1], -item[0]))
    return max(palette_safe_candidates, key=lambda item: item[0])


def detect_color_grid(image: np.ndarray) -> GridGeometry | None:
    gray = _to_gray(image)
    dark = gray < 185
    image_height, image_width = dark.shape
    palette_band = _palette_band(image)
    palette_top = palette_band[0] if palette_band is not None else round(image_height * 0.90)
    x_scores = dark.sum(axis=0)
    y_scores = dark.sum(axis=1)
    x_candidates = []
    y_candidates = []
    for score, lines in _regular_sequences(x_scores, COLOR_GRID_SIZE + 1):
        step = (lines[-1] - lines[0]) / COLOR_GRID_SIZE
        x_candidates.append((score, lines, step))
    for score, lines in _regular_sequences(y_scores, COLOR_GRID_SIZE + 1):
        step = (lines[-1] - lines[0]) / COLOR_GRID_SIZE
        y_candidates.append((score, lines, step))
    candidates = []
    for x_score, x_lines, x_step in x_candidates:
        for y_score, y_lines, y_step in y_candidates:
            if abs(x_step - y_step) / max(x_step, y_step) > 0.07:
                continue
            consistency = _color_grid_consistency(image, x_lines, y_lines)
            transition = _grid_boundary_transition_score(
                image, x_lines, y_lines, COLOR_GRID_SIZE, COLOR_GRID_SIZE
            )
            score = x_score + y_score + transition * 0.2 - consistency * 0.8
            palette_gap = palette_top - y_lines[-1]
            palette_overlap = max(0.0, -palette_gap) / max(1.0, y_step)
            palette_alignment = abs(
                palette_gap - y_step * _COLOR_GRID_TARGET_PALETTE_GAP_CELLS
            ) / max(1.0, y_step)
            candidates.append((score, palette_alignment, palette_overlap, x_lines, y_lines))
    if not candidates:
        return None
    best = _select_color_grid_candidate(candidates, palette_band)
    if best is None:
        return None
    if palette_band is None and best[0] < _COLOR_GRID_MIN_NO_PALETTE_SCORE:
        return None
    _, _, _, x_lines, y_lines = best
    y_start = max(0, round(y_lines[0]))
    y_end = min(image_height, round(y_lines[-1]) + 1)
    x_start = max(0, round(x_lines[0]))
    x_end = min(image_width, round(x_lines[-1]) + 1)
    clue_x_scores = dark[y_start:y_end, :].sum(axis=0)
    clue_y_scores = dark[:, x_start:x_end].sum(axis=1)
    # Try to find clue boundaries with original tolerance
    clue_left = _find_clue_boundary(
        clue_x_scores,
        x_lines[0],
        (x_lines[-1] - x_lines[0]) / COLOR_GRID_SIZE,
        COLOR_GRID_SIZE,
        tolerance_ratio=0.18,
    )
    clue_top = _find_clue_boundary(
        clue_y_scores,
        y_lines[0],
        (y_lines[-1] - y_lines[0]) / COLOR_GRID_SIZE,
        COLOR_GRID_SIZE,
        tolerance_ratio=0.18,
    )

    # If clue boundaries not found (e.g., due to background patterns), try relaxed tolerance
    cell_width = (x_lines[-1] - x_lines[0]) / COLOR_GRID_SIZE
    cell_height = (y_lines[-1] - y_lines[0]) / COLOR_GRID_SIZE

    if clue_left is None:
        clue_left = _find_clue_boundary(
            clue_x_scores,
            x_lines[0],
            cell_width,
            COLOR_GRID_SIZE,
            tolerance_ratio=0.35,
        )

    if clue_top is None:
        clue_top = _find_clue_boundary(
            clue_y_scores,
            y_lines[0],
            cell_height,
            COLOR_GRID_SIZE,
            tolerance_ratio=0.35,
        )

    # Last resort: estimate based on typical clue region size (3-5 cells)
    if clue_left is None:
        clue_left = max(0, x_lines[0] - cell_width * 4.5)

    if clue_top is None:
        clue_top = max(0, y_lines[0] - cell_height * 4.5)

    geometry = GridGeometry(COLOR_GRID_SIZE, COLOR_GRID_SIZE, x_lines, y_lines, best[0], clue_left, clue_top)
    cell_step = max(geometry.cell_width, geometry.cell_height)
    step_ratio = cell_step / max(1, image_height)
    normalized_score = geometry.score / max(1.0, cell_step)
    clue_left_gap = (
        geometry.x_lines[0] - geometry.clue_left
        if geometry.clue_left is not None
        else 0.0
    )
    clue_top_gap = (
        geometry.y_lines[0] - geometry.clue_top
        if geometry.clue_top is not None
        else 0.0
    )
    if (
        geometry.clue_left is None
        or geometry.clue_top is None
        or not _COLOR_GRID_MIN_STEP_RATIO <= step_ratio <= _COLOR_GRID_MAX_STEP_RATIO
        or normalized_score < _COLOR_GRID_MIN_NORMALIZED_SCORE
        or clue_left_gap < geometry.cell_width * _COLOR_GRID_MIN_CLUE_GAP_CELLS
        or clue_top_gap < geometry.cell_height * _COLOR_GRID_MIN_CLUE_GAP_CELLS
        or geometry.x_lines[0] < 0
        or geometry.y_lines[0] < 0
        or geometry.x_lines[-1] >= image_width
        or geometry.y_lines[-1] >= image_height
    ):
        return None
    return geometry


def _load_live_digit_classifier() -> TemplateDigitClassifier | None:
    global _LIVE_DIGIT_CLASSIFIER
    if _LIVE_DIGIT_CLASSIFIER is not None:
        return _LIVE_DIGIT_CLASSIFIER
    manifests = (
        PROJECT_ROOT / "color_digit_templates.json",
        PROJECT_ROOT / "assets" / "color_digit_templates.json",
        PROJECT_ROOT / "install" / "color_digit_templates.json",
        PROJECT_ROOT / "tests" / "fixtures" / "color_digit_templates.json",
    )
    for manifest in manifests:
        if not manifest.exists():
            continue
        try:
            _LIVE_DIGIT_CLASSIFIER = TemplateDigitClassifier(load_digit_templates(manifest))
            return _LIVE_DIGIT_CLASSIFIER
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def _prefer_template_digit(
    ocr_value: int,
    candidates: list[DigitCandidate],
    minimum_score: float = 0.82,
    minimum_margin: float = 0.08,
) -> int | None:
    """
    检查是否应该优先使用模板匹配结果而不是 OCR 结果。
    主要用于处理 1 和 7 的混淆情况。

    注意：此函数当前未在生产代码中使用，但保留用于测试和未来可能的优化。
    """
    if not 1 <= ocr_value <= 15 or not candidates:
        return None
    top = candidates[0]
    if top.value not in (1, 7) or top.score < minimum_score:
        return None
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    if top.score - second_score < minimum_margin:
        return None
    if top.value == ocr_value:
        return None
    return top.value


def _template_digit_from_patch(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
) -> int | None:
    classifier = _load_live_digit_classifier()
    if classifier is None:
        return None
    x, y, width, height = _clip_roi(roi, image)
    patch = image[y : y + height, x : x + width]
    glyph, _, foreground_pixels = normalize_clue_glyph(patch)
    if foreground_pixels == 0:
        return None
    candidates = classifier.classify(glyph, top_k=3)
    if not candidates or candidates[0].score < 0.82:
        return None
    margin = candidates[0].score - candidates[1].score if len(candidates) > 1 else 1.0
    if margin < 0.08:
        return None
    return candidates[0].value


def _read_cell(context: Context, image: np.ndarray, roi: tuple[int, int, int, int]) -> list[str]:
    detail = context.run_recognition_direct(
        JRecognitionType.OCR,
        JOCR(expected=[], roi=_clip_roi(roi, image), threshold=0.2),
        image,
    )
    results = _ocr_results(detail)
    results.sort(key=lambda result: (getattr(getattr(result, "box", None), "x", 0), getattr(getattr(result, "box", None), "y", 0)))
    texts = [str(getattr(result, "text", "")) for result in results]
    ocr_value = _number(texts)
    template_value = _template_digit_from_patch(image, roi)
    if template_value is not None and template_value != ocr_value:
        print(f"color nonogram template retry roi={roi} ocr={ocr_value} template={template_value}")
        return [str(template_value)]
    if template_value is not None and ocr_value is None:
        print(f"color nonogram template fallback roi={roi} ocr=None template={template_value}")
        return [str(template_value)]

    # If OCR returned no text, try with lower threshold
    if not texts or ocr_value is None:
        detail_relaxed = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(expected=[], roi=_clip_roi(roi, image), threshold=0.1),
            image,
        )
        results_relaxed = _ocr_results(detail_relaxed)
        if results_relaxed:
            results_relaxed.sort(key=lambda result: (getattr(getattr(result, "box", None), "x", 0), getattr(getattr(result, "box", None), "y", 0)))
            texts_relaxed = [str(getattr(result, "text", "")) for result in results_relaxed]
            relaxed_value = _number(texts_relaxed)
            if relaxed_value is not None:
                print(f"color nonogram OCR relaxed threshold succeeded roi={roi} value={relaxed_value}")
                return texts_relaxed

    return texts


def _read_cell_enlarged(context: Context, image: np.ndarray, roi: tuple[int, int, int, int]) -> list[str]:
    x, y, width, height = _clip_roi(roi, image)
    crop = image[y : y + height, x : x + width]
    if not crop.size:
        return []
    padding = max(2, round(min(crop.shape[:2]) * 0.12))
    padded = np.pad(crop, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
    enlarged = np.asarray(
        Image.fromarray(padded).resize(
            (padded.shape[1] * 2, padded.shape[0] * 2),
            Image.Resampling.LANCZOS,
        )
    )
    return _read_cell(context, enlarged, (0, 0, enlarged.shape[1], enlarged.shape[0]))


def _number(texts: list[str]) -> int | None:
    digits = "".join(re.findall(r"\d+", "".join(_normalize_clue_text(text) for text in texts)))
    if not digits:
        return None
    value = int(digits)
    return value if 1 <= value <= COLOR_GRID_SIZE else None


def _clue_cell_has_mark(image: np.ndarray, roi: tuple[int, int, int, int]) -> bool:
    x, y, width, height = _clip_roi(roi, image)
    patch = image[y : y + height, x : x + width, :3]
    if not patch.size:
        return False
    _, _, foreground_pixels = normalize_clue_glyph(patch)
    return foreground_pixels >= max(4, round(width * height * 0.015))


def _use_small_clue_guard(cell_width: float, cell_height: float) -> bool:
    return min(cell_width, cell_height) <= 42


def _skip_unmarked_small_clue(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    use_small_clue_guard: bool,
) -> bool:
    if not use_small_clue_guard or _clue_cell_has_mark(image, roi):
        return False
    x, y, width, height = _clip_roi(roi, image)
    patch = image[y : y + height, x : x + width, :3]
    if not patch.size:
        return False
    core = patch[2:-2, 2:-2] if min(patch.shape[:2]) > 4 else patch
    gray = core.astype(np.float32).mean(axis=2)
    return float(gray.std()) <= 6.0 and float(gray.max() - gray.min()) <= 24.0


def _validate_clue_slots(axis: str, line: int, slots: set[int]) -> bool:
    if not slots:
        return False
    expected = set(range(max(slots) + 1))
    if slots != expected:
        missing = sorted(expected - slots)
        raise ValueError(f"missing {axis} clue slots at line={line}: {missing}")
    return True


def _colored_clues(
    context: Context,
    image: np.ndarray,
    geometry: GridGeometry,
    palette: list[PaletteColor],
    background_rgb: tuple[float, float, float] | None = None,
) -> tuple[
    list[list[tuple[int, int]]],
    list[list[tuple[int, int]]],
    list[list[tuple[int, int, int, int]]],
    list[list[tuple[int, int, int, int]]],
    list[dict[int, tuple[int, int]]],
    list[dict[int, tuple[int, int]]],
    list[dict[str, set[int]]],
    list[dict[str, set[int]]],
] | None:
    cell_width = geometry.cell_width
    cell_height = geometry.cell_height
    # 判断是否是小单元格,需要使用放大策略提高OCR准确率
    use_enlarged_ocr = min(cell_width, cell_height) < 40
    use_small_clue_guard = _use_small_clue_guard(cell_width, cell_height)
    row_slots = COLOR_GRID_SIZE
    column_slots = COLOR_GRID_SIZE
    rows = []
    columns = []
    row_rois = []
    column_rois = []
    row_caches = []
    column_caches = []
    row_slot_diagnostics = []
    column_slot_diagnostics = []
    for row in range(COLOR_GRID_SIZE):
        clues = []
        rois = []
        cache: dict[int, tuple[int, int]] = {}
        observed_slots: set[int] = set()
        checked_slot_set: set[int] = set()
        checked_slots = 0
        for slot in range(row_slots - 1, -1, -1):
            roi = (
                round(geometry.x_lines[0] - (slot + 1) * cell_width + cell_width * 0.1),
                round(geometry.y_lines[row] + cell_height * 0.1),
                round(cell_width * 0.8),
                round(cell_height * 0.8),
            )
            if not _clue_roi_within_bounds(roi, image, clue_left=geometry.clue_left):
                continue
            checked_slots += 1
            checked_slot_set.add(slot)
            color = _color_from_match(image, roi, palette, lenient=True, background_rgb=background_rgb)
            if color is None:
                if _clue_cell_has_mark(image, roi):
                    raise ValueError(f"unable to identify row clue color at line={row} slot={slot}")
                continue
            if _skip_unmarked_small_clue(image, roi, use_small_clue_guard=use_small_clue_guard):
                continue
            # 对小单元格使用放大策略提高OCR识别率
            if use_enlarged_ocr:
                texts = _read_cell_enlarged(context, image, roi)
            else:
                texts = _read_cell(context, image, roi)
            number = _number(texts)
            if number is None:
                raise ValueError(f"unable to read row clue at line={row} slot={slot}")
            observed_slots.add(slot)
            clues.append((color, number))
            rois.append(roi)
            cache[slot] = (color, number)
        if not checked_slots:
            raise ValueError(f"no visible row clue slots at line={row}")
        _validate_clue_slots("row", row, observed_slots)
        if not observed_slots:
            print(f"color nonogram row line={row} explicitly blank checked_slots={checked_slots}")
        rows.append(clues)
        row_rois.append(rois)
        row_caches.append(cache)
        row_slot_diagnostics.append(
            {"checked": checked_slot_set, "recognized": set(observed_slots)}
        )
    for column in range(COLOR_GRID_SIZE):
        clues = []
        rois = []
        cache = {}
        observed_slots = set()
        checked_slot_set: set[int] = set()
        checked_slots = 0
        for slot in range(column_slots - 1, -1, -1):
            roi = (
                round(geometry.x_lines[column] + cell_width * 0.1),
                round(geometry.y_lines[0] - (slot + 1) * cell_height + cell_height * 0.1),
                round(cell_width * 0.8),
                round(cell_height * 0.8),
            )
            if not _clue_roi_within_bounds(roi, image, clue_top=geometry.clue_top):
                continue
            checked_slots += 1
            checked_slot_set.add(slot)
            color = _color_from_match(image, roi, palette, lenient=True, background_rgb=background_rgb)
            if color is None:
                if _clue_cell_has_mark(image, roi):
                    raise ValueError(f"unable to identify column clue color at line={column} slot={slot}")
                continue
            if _skip_unmarked_small_clue(image, roi, use_small_clue_guard=use_small_clue_guard):
                continue
            # 对小单元格使用放大策略提高OCR识别率
            if use_enlarged_ocr:
                texts = _read_cell_enlarged(context, image, roi)
            else:
                texts = _read_cell(context, image, roi)
            number = _number(texts)
            if number is None:
                raise ValueError(f"unable to read column clue at line={column} slot={slot}")
            observed_slots.add(slot)
            clues.append((color, number))
            rois.append(roi)
            cache[slot] = (color, number)
        if not checked_slots:
            raise ValueError(f"no visible column clue slots at line={column}")
        _validate_clue_slots("column", column, observed_slots)
        if not observed_slots:
            print(f"color nonogram column line={column} explicitly blank checked_slots={checked_slots}")
        columns.append(clues)
        column_rois.append(rois)
        column_caches.append(cache)
        column_slot_diagnostics.append(
            {"checked": checked_slot_set, "recognized": set(observed_slots)}
        )
    print(f"color nonogram clues rows={rows!r} columns={columns!r}")
    return (
        rows,
        columns,
        row_rois,
        column_rois,
        row_caches,
        column_caches,
        row_slot_diagnostics,
        column_slot_diagnostics,
    )



def _rank_digit_candidates(
    scores: dict[int, float],
    template_scores: dict[int, float],
) -> tuple[DigitCandidate, ...]:
    ranked = sorted(
        (DigitCandidate(value, score) for value, score in scores.items()),
        key=lambda item: (item.score, item.value),
        reverse=True,
    )
    if len(ranked) <= 1:
        return tuple(ranked)
    top = ranked[0]
    selected = [top]
    for candidate in ranked[1:]:
        if candidate.score >= 0.78 and top.score - candidate.score <= 0.12:
            selected.append(candidate)
    if top.value in (1, 7):
        opposite = 7 if top.value == 1 else 1
        if template_scores.get(opposite, 0.0) >= 0.62:
            opposite_candidate = next(
                (candidate for candidate in ranked if candidate.value == opposite),
                None,
            )
            if opposite_candidate is not None and opposite_candidate not in selected:
                selected.append(opposite_candidate)
    return tuple(selected[:3])

def _digit_candidates(
    context: Context,
    image: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[DigitCandidate, ...]:
    candidates: dict[int, float] = {}

    # OCR 识别
    value = _number(_read_cell(context, image, roi))
    if value is not None:
        candidates[value] = 1.0

    enlarged_value = _number(_read_cell_enlarged(context, image, roi))
    if enlarged_value is not None:
        candidates[enlarged_value] = max(candidates.get(enlarged_value, 0.0), 0.96)

    template_scores: dict[int, float] = {}
    classifier = _load_live_digit_classifier()
    if classifier is not None:
        x, y, width, height = _clip_roi(roi, image)
        patch = image[y : y + height, x : x + width]
        glyph, _, foreground_pixels = normalize_clue_glyph(patch)
        if foreground_pixels:
            for candidate in classifier.classify(glyph, top_k=3):
                if candidate.score >= 0.62:
                    template_scores[candidate.value] = candidate.score
                    candidates[candidate.value] = max(candidates.get(candidate.value, 0.0), candidate.score)

    return _rank_digit_candidates(candidates, template_scores)


def _colored_clue_observations(
    context: Context,
    image: np.ndarray,
    geometry: GridGeometry,
    palette: list[PaletteColor],
    *,
    target_colors: set[int] | None = None,
    existing_rows: list[dict[int, tuple[int, int]]] | None = None,
    existing_columns: list[dict[int, tuple[int, int]]] | None = None,
) -> tuple[
    list[list[ClueObservation]],
    list[list[ClueObservation]],
    set[int],
    set[int],
]:
    rows: list[list[ClueObservation]] = []
    columns: list[list[ClueObservation]] = []
    empty_rows: set[int] = set()
    empty_columns: set[int] = set()
    for row in range(COLOR_GRID_SIZE):
        line: list[ClueObservation] = []
        observed_slots: set[int] = set()
        checked_slots = 0
        for slot in range(COLOR_GRID_SIZE - 1, -1, -1):
            roi = (
                round(geometry.x_lines[0] - (slot + 1) * geometry.cell_width + geometry.cell_width * 0.1),
                round(geometry.y_lines[row] + geometry.cell_height * 0.1),
                round(geometry.cell_width * 0.8),
                round(geometry.cell_height * 0.8),
            )
            if not _clue_roi_within_bounds(roi, image, clue_left=geometry.clue_left):
                continue
            checked_slots += 1
            color = _color_from_match(image, roi, palette)
            if color is None:
                if _clue_cell_has_mark(image, roi):
                    raise ValueError(f"unable to identify row clue color at line={row} slot={slot}")
                continue
            observed_slots.add(slot)
            if existing_rows is not None:
                cached = existing_rows[row].get(slot)
                if cached is None:
                    raise ValueError(f"row clue cache is incomplete at line={row} slot={slot}")
                cached_value = cached[1]
            else:
                cached_value = None
            if target_colors is None or color in target_colors:
                candidates = _digit_candidates(context, image, roi)
            elif cached_value is not None:
                candidates = (DigitCandidate(cached_value, 1.0),)
            else:
                candidates = ()
            if not candidates:
                raise ValueError(f"unable to read row clue at line={row} slot={slot}")
            line.append(
                ClueObservation(
                    "row",
                    row,
                    slot,
                    tuple(ClueCandidate(item.value, color, item.score) for item in candidates),
                )
            )
        if not checked_slots:
            raise ValueError(f"no visible row clue slots at line={row}")
        if not observed_slots:
            empty_rows.add(row)
            print(f"color nonogram row line={row} explicitly blank checked_slots={checked_slots}")
        else:
            _validate_clue_slots("row", row, observed_slots)
        rows.append(line)
    for column in range(COLOR_GRID_SIZE):
        line: list[ClueObservation] = []
        observed_slots: set[int] = set()
        checked_slots = 0
        for slot in range(COLOR_GRID_SIZE - 1, -1, -1):
            roi = (
                round(geometry.x_lines[column] + geometry.cell_width * 0.1),
                round(geometry.y_lines[0] - (slot + 1) * geometry.cell_height + geometry.cell_height * 0.1),
                round(geometry.cell_width * 0.8),
                round(geometry.cell_height * 0.8),
            )
            if not _clue_roi_within_bounds(roi, image, clue_top=geometry.clue_top):
                continue
            checked_slots += 1
            color = _color_from_match(image, roi, palette)
            if color is None:
                if _clue_cell_has_mark(image, roi):
                    raise ValueError(f"unable to identify column clue color at line={column} slot={slot}")
                continue
            observed_slots.add(slot)
            if existing_columns is not None:
                cached = existing_columns[column].get(slot)
                if cached is None:
                    raise ValueError(f"column clue cache is incomplete at line={column} slot={slot}")
                cached_value = cached[1]
            else:
                cached_value = None
            if target_colors is None or color in target_colors:
                candidates = _digit_candidates(context, image, roi)
            elif cached_value is not None:
                candidates = (DigitCandidate(cached_value, 1.0),)
            else:
                candidates = ()
            if not candidates:
                raise ValueError(f"unable to read column clue at line={column} slot={slot}")
            line.append(
                ClueObservation(
                    "column",
                    column,
                    slot,
                    tuple(ClueCandidate(item.value, color, item.score) for item in candidates),
                )
            )
        if not checked_slots:
            raise ValueError(f"no visible column clue slots at line={column}")
        if not observed_slots:
            empty_columns.add(column)
            print(f"color nonogram column line={column} explicitly blank checked_slots={checked_slots}")
        else:
            _validate_clue_slots("column", column, observed_slots)
        columns.append(line)
    return rows, columns, empty_rows, empty_columns


def _solve_with_candidate_recovery(
    context: Context,
    image: np.ndarray,
    geometry: GridGeometry,
    palette: list[PaletteColor],
    rows: list[list[tuple[int, int]]],
    columns: list[list[tuple[int, int]]],
    row_cache: list[dict[int, tuple[int, int]]] | None = None,
    column_cache: list[dict[int, tuple[int, int]]] | None = None,
    *,
    empty_rows: set[int] | None = None,
    empty_columns: set[int] | None = None,
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]], list[list[int]]]:
    forced_empty_rows = set(empty_rows or ())
    forced_empty_columns = set(empty_columns or ())
    if any(not 0 <= row < len(rows) for row in forced_empty_rows):
        raise ValueError("empty row index is outside the puzzle")
    if any(not 0 <= column < len(columns) for column in forced_empty_columns):
        raise ValueError("empty column index is outside the puzzle")
    rows = [list(line) for line in rows]
    columns = [list(line) for line in columns]
    for row in forced_empty_rows:
        rows[row] = []
    for column in forced_empty_columns:
        columns[column] = []
    try:
        solution = _solve(rows, columns, maximum_seconds=3.0)
        _validate_empty_solution(solution, forced_empty_rows, forced_empty_columns)
        return rows, columns, solution
    except SolutionSearchTimeout as direct_error:
        raise SolutionSearchTimeout(
            f"direct color nonogram solve timed out before OCR candidate recovery: {direct_error}"
        ) from direct_error
    except ValueError as direct_error:
        print(f"color nonogram direct clue solve rejected: {direct_error}", flush=True)
    row_totals = [sum(number for line in rows for color, number in line if color == index) for index in range(len(palette))]
    column_totals = [sum(number for line in columns for color, number in line if color == index) for index in range(len(palette))]
    target_colors = {index for index, (row_total, column_total) in enumerate(zip(row_totals, column_totals)) if row_total != column_total}
    if not target_colors:
        target_colors = set(range(len(palette)))
    print(f"color nonogram targeted reread colors={sorted(target_colors)}")
    observed_rows, observed_columns, empty_rows, empty_columns = _colored_clue_observations(
        context,
        image,
        geometry,
        palette,
        target_colors=target_colors,
        existing_rows=row_cache,
        existing_columns=column_cache,
    )
    empty_rows = set(empty_rows) | forced_empty_rows
    empty_columns = set(empty_columns) | forced_empty_columns
    for row in forced_empty_rows:
        observed_rows[row] = []
    for column in forced_empty_columns:
        observed_columns[column] = []
    result = disambiguate_puzzle(
        observed_rows,
        observed_columns,
        width=COLOR_GRID_SIZE,
        height=COLOR_GRID_SIZE,
        color_count=len(palette),
        top_k=3,
        maximum_line_options=96,
        maximum_attempts=3000,
        require_unique=True,
        maximum_seconds=12.0,
        empty_rows=empty_rows,
        empty_columns=empty_columns,
    )
    if result.status != "unique" or result.puzzle is None or result.solution is None:
        raise ValueError(
            f"colored clue candidate search {result.status}: {result.reason}; "
            f"attempts={result.attempts}; candidate_timeouts={result.candidate_timeouts}; "
            f"row_option_counts={result.row_option_counts}; "
            f"column_option_counts={result.column_option_counts}"
        )
    recovered_rows = [list(line) for line in result.puzzle.rows]
    recovered_columns = [list(line) for line in result.puzzle.columns]
    _validate_empty_solution(
        [list(line) for line in result.solution.grid],
        forced_empty_rows,
        forced_empty_columns,
    )
    print(f"color nonogram candidate recovery status={result.status} attempts={result.attempts}")
    print(f"color nonogram recovered clues rows={recovered_rows!r} columns={recovered_columns!r}")
    return recovered_rows, recovered_columns, [list(row) for row in result.solution.grid]


def _solve(
    rows: list[list[tuple[int, int]]],
    columns: list[list[tuple[int, int]]],
    *,
    maximum_seconds: float | None = None,
) -> list[list[int]]:
    puzzle = ColorNonogramPuzzle.from_clues(rows, columns)
    puzzle.validate_color_totals()
    return [list(row) for row in solve_puzzle(puzzle, require_unique=True, maximum_seconds=maximum_seconds).grid]


def _validate_empty_solution(
    solution: list[list[int]],
    empty_rows: set[int],
    empty_columns: set[int],
) -> None:
    for row in empty_rows:
        if any(solution[row]):
            raise ValueError(f"solution has colored cells in empty row={row}")
    for column in empty_columns:
        if any(solution[row][column] for row in range(len(solution))):
            raise ValueError(f"solution has colored cells in empty column={column}")


@AgentServer.custom_action("识别并完成彩色数织")
class SolveColorNonogramAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        param = _load_param(argv.custom_action_param)
        try:
            image = context.tasker.controller.post_screencap().wait().get()
            if image is None:
                raise RuntimeError("screenshot returned no image")
            image = np.asarray(image)
            geometry = detect_color_grid(image)
            if geometry is None or geometry.rows != COLOR_GRID_SIZE or geometry.columns != COLOR_GRID_SIZE:
                raise ValueError("unable to locate a 15x15 colored grid")

            # 增强页面验证: 检查网格检测置信度
            if geometry.score < 50.0:
                print(f"Warning: low grid detection confidence (score={geometry.score:.1f}), may not be a valid puzzle page")
                raise ValueError(f"grid detection confidence too low (score={geometry.score:.1f}), possibly not a puzzle page")

            palette = _palette_colors(image)
            if not 2 <= len(palette) <= 12:
                # 如果只检测到1种颜色,很可能是在错误的页面上(如用户作品列表、菜单等)
                if len(palette) == 1:
                    print(f"Warning: only 1 palette color detected, likely not on a puzzle page")
                raise ValueError(f"expected 2-12 palette colors, found {len(palette)}")
            if not _verify_clue_color_match(context, image, geometry, palette):
                raise ValueError("Maa ColorMatch did not find any palette color in the clue area")
            print(f"color nonogram grid={geometry} palette={palette!r}")

            # 计算背景色：采样网格中心区域的中位数RGB
            center_x = int((geometry.x_lines[0] + geometry.x_lines[-1]) / 2)
            center_y = int((geometry.y_lines[0] + geometry.y_lines[-1]) / 2)
            sample_size = 20
            bg_patch = image[
                max(0, center_y - sample_size):min(image.shape[0], center_y + sample_size),
                max(0, center_x - sample_size):min(image.shape[1], center_x + sample_size),
                :3
            ]
            background_rgb = tuple(float(x) for x in np.median(bg_patch.reshape(-1, 3), axis=0)) if bg_patch.size > 0 else None

            extracted = _colored_clues(context, image, geometry, palette, background_rgb)
            if extracted is None:
                raise ValueError("colored clues are incomplete")
            (
                rows,
                columns,
                row_rois,
                column_rois,
                row_cache,
                column_cache,
                _,
                _,
            ) = extracted
            row_totals = [sum(number for line in rows for color, number in line if color == palette_index) for palette_index in range(len(palette))]
            column_totals = [sum(number for line in columns for color, number in line if color == palette_index) for palette_index in range(len(palette))]
            print(f"color nonogram clues rows={rows!r} columns={columns!r}")
            print(f"color nonogram color totals rows={row_totals!r} columns={column_totals!r}")
            rows, columns, solution = _solve_with_candidate_recovery(
                context,
                image,
                geometry,
                palette,
                rows,
                columns,
                row_cache,
                column_cache,
            )
            for color, row, start, end in _runs(solution):
                if color >= len(palette):
                    raise ValueError("solution color is not present in palette")
                palette_x, palette_y = palette[color].center
                selection = context.tasker.controller.post_click(palette_x, palette_y).wait()
                if not selection.succeeded:
                    raise RuntimeError("color palette selection failed")
                start_x, y = geometry.center(row, start)
                end_x, _ = geometry.center(row, end)
                if start == end:
                    result = context.tasker.controller.post_click(start_x, y).wait()
                else:
                    duration = max(250, 140 + (end - start + 1) * 65)
                    result = context.tasker.controller.post_swipe(start_x, y, end_x, y, duration).wait()
                if not result.succeeded:
                    raise RuntimeError("color paint action failed")
                time.sleep(max(0.03, float(param.get("pause", 0.08))))
            context.override_next(argv.node_name, [str(param.get("success_node", "彩色数织测试完成"))])
            return True
        except Exception as exc:
            print(f"color nonogram solve failed: {exc}")
            context.override_next(argv.node_name, [str(param.get("failure_node", "彩色数织测试失败"))])
            return True
