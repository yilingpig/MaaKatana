from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DigitTemplate:
    label: int
    glyph: np.ndarray
    source: str = ""


@dataclass(frozen=True)
class DigitCandidate:
    value: int
    score: float


@dataclass(frozen=True)
class DigitDecision:
    candidates: tuple[DigitCandidate, ...]
    accepted: bool
    reason: str


class TemplateDigitClassifier:
    def __init__(self, templates: Iterable[DigitTemplate]):
        grouped: dict[int, list[DigitTemplate]] = {}
        for template in templates:
            if not 1 <= template.label <= 15:
                raise ValueError("colored clue labels must be between 1 and 15")
            grouped.setdefault(template.label, []).append(template)
        if not grouped:
            raise ValueError("at least one digit template is required")
        self._templates = grouped

    def classify(self, glyph: np.ndarray, top_k: int = 3) -> list[DigitCandidate]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = []
        for label, templates in self._templates.items():
            score = max(shifted_dice_similarity(glyph, template.glyph) for template in templates)
            scores.append(DigitCandidate(label, score))
        scores.sort(key=lambda candidate: candidate.score, reverse=True)
        return scores[:top_k]

    def decide(
        self,
        glyph: np.ndarray,
        top_k: int = 3,
        minimum_score: float = 0.82,
        minimum_margin: float = 0.06,
    ) -> DigitDecision:
        candidates = tuple(self.classify(glyph, top_k=max(2, top_k)))
        if not candidates:
            return DigitDecision((), False, "no templates")
        if candidates[0].score < minimum_score:
            return DigitDecision(candidates[:top_k], False, "score below threshold")
        margin = candidates[0].score - candidates[1].score if len(candidates) > 1 else 1.0
        if margin < minimum_margin:
            return DigitDecision(candidates[:top_k], False, "top candidates are ambiguous")
        return DigitDecision(candidates[:top_k], True, "accepted")

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(sorted(self._templates))


def load_digit_templates(manifest_path: pathlib.Path) -> list[DigitTemplate]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    templates = []
    for item in data:
        glyph_path = manifest_path.parent / item["path"]
        templates.append(
            DigitTemplate(
                label=int(item["label"]),
                glyph=np.asarray(Image.open(glyph_path).convert("L")),
                source=str(item.get("source", glyph_path.name)),
            )
        )
    return templates


def shifted_dice_similarity(left: np.ndarray, right: np.ndarray, max_shift: int = 2) -> float:
    if left.shape != right.shape:
        raise ValueError("glyph shapes must match")
    left_mask = left > 0
    right_mask = right > 0
    if not left_mask.any() and not right_mask.any():
        return 1.0
    if not left_mask.any() or not right_mask.any():
        return 0.0
    best = 0.0
    for shift_y in range(-max_shift, max_shift + 1):
        for shift_x in range(-max_shift, max_shift + 1):
            shifted = _shift_mask(right_mask, shift_x, shift_y)
            overlap = int(np.logical_and(left_mask, shifted).sum())
            denominator = int(left_mask.sum() + shifted.sum())
            if denominator:
                best = max(best, 2.0 * overlap / denominator)
    return best


def greedy_cluster_glyphs(
    glyphs: list[np.ndarray],
    threshold: float = 0.78,
) -> list[list[int]]:
    clusters: list[list[int]] = []
    for index, glyph in enumerate(glyphs):
        best_cluster = None
        best_score = threshold
        for cluster_index, members in enumerate(clusters):
            score = max(shifted_dice_similarity(glyph, glyphs[member]) for member in members[:4])
            if score > best_score:
                best_cluster = cluster_index
                best_score = score
        if best_cluster is None:
            clusters.append([index])
        else:
            clusters[best_cluster].append(index)
    clusters.sort(key=len, reverse=True)
    return clusters


def _shift_mask(mask: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    output = np.zeros_like(mask)
    source_x0 = max(0, -shift_x)
    source_x1 = min(mask.shape[1], mask.shape[1] - shift_x)
    source_y0 = max(0, -shift_y)
    source_y1 = min(mask.shape[0], mask.shape[0] - shift_y)
    target_x0 = source_x0 + shift_x
    target_x1 = source_x1 + shift_x
    target_y0 = source_y0 + shift_y
    target_y1 = source_y1 + shift_y
    if source_x1 > source_x0 and source_y1 > source_y0:
        output[target_y0:target_y1, target_x0:target_x1] = mask[source_y0:source_y1, source_x0:source_x1]
    return output
