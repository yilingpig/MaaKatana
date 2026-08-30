import argparse
import json
import pathlib
import shutil

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
AGENT_ROOT = PROJECT_ROOT / "agent"

import sys

sys.path.insert(0, str(AGENT_ROOT))

from color_nonogram_core import solve_puzzle
from color_nonogram_digits import TemplateDigitClassifier, greedy_cluster_glyphs, load_digit_templates
from color_nonogram_truth import extract_board_truth, iter_clue_truth
from color_nonogram_vision import GridCalibration, crop_roi, extract_clue_cells


def _load_calibrations(path: pathlib.Path) -> list[GridCalibration]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        GridCalibration(
            source=item["source"],
            grid=tuple(item["grid"]),
            clue_left=float(item["clue_left"]),
            clue_top=float(item["clue_top"]),
            size=int(item.get("size", 15)),
        )
        for item in data
    ]


def _reset_output(output: pathlib.Path) -> None:
    resolved = output.resolve()
    debug_root = (PROJECT_ROOT / "debug").resolve()
    if debug_root not in resolved.parents:
        raise ValueError("dataset output must stay under the project debug directory")
    if resolved.exists():
        shutil.rmtree(resolved)
    (resolved / "glyphs").mkdir(parents=True)
    (resolved / "crops").mkdir(parents=True)


def _contact_sheet(
    output: pathlib.Path,
    glyphs: list[np.ndarray],
    clusters: list[list[int]],
) -> None:
    scale = 3
    glyph_size = glyphs[0].shape[0] if glyphs else 24
    tile_width = glyph_size * scale + 12
    tile_height = glyph_size * scale + 28
    members_per_cluster = 8
    columns = 4
    cluster_width = tile_width * members_per_cluster
    cluster_height = tile_height
    rows = (len(clusters) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cluster_width, max(1, rows) * cluster_height), "white")
    draw = ImageDraw.Draw(sheet)
    for cluster_index, members in enumerate(clusters):
        origin_x = (cluster_index % columns) * cluster_width
        origin_y = (cluster_index // columns) * cluster_height
        draw.text((origin_x + 2, origin_y + 2), f"C{cluster_index:03d} n={len(members)}", fill="black")
        for position, member in enumerate(members[:members_per_cluster]):
            glyph = Image.fromarray(glyphs[member]).resize(
                (glyph_size * scale, glyph_size * scale),
                Image.Resampling.NEAREST,
            )
            sheet.paste(glyph.convert("RGB"), (origin_x + position * tile_width, origin_y + 20))
    sheet.save(output / "clusters.png")


def _source_screenshot(source: str) -> str:
    marker = ".png"
    end = source.find(marker)
    return source[: end + len(marker)] if end >= 0 else source


def _template_holdout_report(templates: list) -> dict:
    results = []
    tested = 0
    correct = 0
    accepted_correct = 0
    for template in templates:
        source = _source_screenshot(template.source)
        training = [
            candidate
            for candidate in templates
            if _source_screenshot(candidate.source) != source
        ]
        if not any(candidate.label == template.label for candidate in training):
            results.append(
                {
                    "label": template.label,
                    "source": template.source,
                    "tested": False,
                    "reason": "no same-label template from another screenshot",
                }
            )
            continue
        decision = TemplateDigitClassifier(training).decide(template.glyph)
        predicted = decision.candidates[0].value if decision.candidates else None
        is_correct = predicted == template.label
        tested += 1
        correct += int(is_correct)
        accepted_correct += int(is_correct and decision.accepted)
        results.append(
            {
                "label": template.label,
                "source": template.source,
                "tested": True,
                "predicted": predicted,
                "correct": is_correct,
                "accepted": decision.accepted,
                "reason": decision.reason,
                "candidates": [
                    {"value": candidate.value, "score": round(candidate.score, 6)}
                    for candidate in decision.candidates
                ],
            }
        )
    return {
        "tested": tested,
        "correct": correct,
        "accepted_correct": accepted_correct,
        "accuracy": round(correct / tested, 6) if tested else None,
        "accepted_accuracy": round(accepted_correct / tested, 6) if tested else None,
        "results": results,
    }


def _ground_truth_report(
    manifest: list[dict],
    glyphs: list[np.ndarray],
    board_truths: list,
    templates: list,
    color_distance_threshold: float = 55.0,
) -> dict:
    samples = {
        (item["source"], item["axis"], item["line"], item["slot"]): (item, glyph)
        for item, glyph in zip(manifest, glyphs)
    }
    totals = {
        "truth_samples": 0,
        "matched_samples": 0,
        "top1_correct": 0,
        "top3_correct": 0,
        "accepted_samples": 0,
        "accepted_correct": 0,
        "accepted_incorrect": 0,
        "color_matches": 0,
    }
    per_label = {}
    per_screenshot = []
    confusions = {}
    results = []
    for truth in board_truths:
        screenshot = {
            "source": truth.source,
            "completed": truth.completed,
            "reason": truth.reason,
            "filled_cells": truth.filled_cells,
            "high_variance_cells": truth.high_variance_cells,
            "palette_rgb": [list(color) for color in truth.palette_rgb],
            "truth_samples": 0,
            "matched_samples": 0,
            "top1_correct": 0,
            "top3_correct": 0,
            "accepted_samples": 0,
            "accepted_correct": 0,
            "accepted_incorrect": 0,
            "color_matches": 0,
            "unique_solution": False,
            "solution_matches_board": False,
        }
        if not truth.completed:
            per_screenshot.append(screenshot)
            continue
        try:
            solution = solve_puzzle(truth.puzzle(), require_unique=True)
            screenshot["unique_solution"] = True
            screenshot["solution_matches_board"] = solution.grid == truth.grid
        except ValueError:
            pass
        training = [
            template
            for template in templates
            if _source_screenshot(template.source) != truth.source
        ]
        classifier = TemplateDigitClassifier(training)
        for expected in iter_clue_truth(truth):
            totals["truth_samples"] += 1
            screenshot["truth_samples"] += 1
            label = per_label.setdefault(
                str(expected.value),
                {
                    "truth_samples": 0,
                    "top1_correct": 0,
                    "top3_correct": 0,
                    "accepted_correct": 0,
                },
            )
            label["truth_samples"] += 1
            key = (truth.source, expected.axis, expected.line, expected.slot)
            sample = samples.get(key)
            if sample is None:
                results.append(
                    {
                        "source": truth.source,
                        "axis": expected.axis,
                        "line": expected.line,
                        "slot": expected.slot,
                        "truth_value": expected.value,
                        "truth_color": expected.color,
                        "matched": False,
                    }
                )
                continue
            item, glyph = sample
            decision = classifier.decide(glyph)
            candidates = [
                {"value": candidate.value, "score": round(candidate.score, 6)}
                for candidate in decision.candidates
            ]
            values = [candidate["value"] for candidate in candidates]
            predicted = values[0] if values else None
            top1_correct = predicted == expected.value
            top3_correct = expected.value in values
            accepted_correct = decision.accepted and top1_correct
            accepted_incorrect = decision.accepted and not top1_correct
            color_distance = float(
                np.linalg.norm(
                    np.asarray(item["background_rgb"], dtype=np.float32)
                    - np.asarray(expected.background_rgb, dtype=np.float32)
                )
            )
            color_matches = color_distance <= color_distance_threshold
            totals["matched_samples"] += 1
            totals["top1_correct"] += int(top1_correct)
            totals["top3_correct"] += int(top3_correct)
            totals["accepted_samples"] += int(decision.accepted)
            totals["accepted_correct"] += int(accepted_correct)
            totals["accepted_incorrect"] += int(accepted_incorrect)
            totals["color_matches"] += int(color_matches)
            screenshot["matched_samples"] += 1
            screenshot["top1_correct"] += int(top1_correct)
            screenshot["top3_correct"] += int(top3_correct)
            screenshot["accepted_samples"] += int(decision.accepted)
            screenshot["accepted_correct"] += int(accepted_correct)
            screenshot["accepted_incorrect"] += int(accepted_incorrect)
            screenshot["color_matches"] += int(color_matches)
            label["top1_correct"] += int(top1_correct)
            label["top3_correct"] += int(top3_correct)
            label["accepted_correct"] += int(accepted_correct)
            if not top1_correct:
                confusion_key = (expected.value, predicted)
                confusions[confusion_key] = confusions.get(confusion_key, 0) + 1
            truth_data = {
                "value": expected.value,
                "color": expected.color,
                "background_rgb": list(expected.background_rgb),
                "color_distance": round(color_distance, 6),
            }
            item["truth"] = truth_data
            results.append(
                {
                    "sample_id": item["id"],
                    "source": truth.source,
                    "axis": expected.axis,
                    "line": expected.line,
                    "slot": expected.slot,
                    "truth_value": expected.value,
                    "truth_color": expected.color,
                    "truth_background_rgb": list(expected.background_rgb),
                    "observed_background_rgb": item["background_rgb"],
                    "color_distance": round(color_distance, 6),
                    "color_matches": color_matches,
                    "matched": True,
                    "predicted": predicted,
                    "top1_correct": top1_correct,
                    "top3_correct": top3_correct,
                    "accepted": decision.accepted,
                    "accepted_correct": accepted_correct,
                    "reason": decision.reason,
                    "candidates": candidates,
                }
            )
        per_screenshot.append(screenshot)
    for metrics in per_label.values():
        count = metrics["truth_samples"]
        metrics["top1_accuracy"] = round(metrics["top1_correct"] / count, 6)
        metrics["top3_accuracy"] = round(metrics["top3_correct"] / count, 6)
        metrics["accepted_accuracy"] = round(metrics["accepted_correct"] / count, 6)
    matched = totals["matched_samples"]
    totals["top1_accuracy"] = round(totals["top1_correct"] / matched, 6) if matched else None
    totals["top3_accuracy"] = round(totals["top3_correct"] / matched, 6) if matched else None
    totals["accepted_accuracy"] = (
        round(totals["accepted_correct"] / matched, 6) if matched else None
    )
    totals["color_accuracy"] = round(totals["color_matches"] / matched, 6) if matched else None
    return {
        "mode": "leave-one-screenshot-out",
        "completed_screenshots": sum(truth.completed for truth in board_truths),
        "unique_solution_matches": sum(
            screenshot["solution_matches_board"] for screenshot in per_screenshot
        ),
        **totals,
        "per_screenshot": per_screenshot,
        "per_label": per_label,
        "confusions": [
            {"truth": truth, "predicted": predicted, "count": count}
            for (truth, predicted), count in sorted(
                confusions.items(), key=lambda item: item[1], reverse=True
            )
        ],
        "results": results,
    }


def build_dataset(
    calibrations_path: pathlib.Path,
    output: pathlib.Path,
    cluster_threshold: float = 0.88,
    template_manifest: pathlib.Path | None = None,
) -> dict:
    calibrations = _load_calibrations(calibrations_path)
    _reset_output(output)
    image_root = PROJECT_ROOT / "assets" / "resource" / "image"
    manifest = []
    glyphs = []
    board_truths = []
    for calibration in calibrations:
        image_path = image_root / calibration.source
        image = np.asarray(Image.open(image_path).convert("RGB"))
        cells = extract_clue_cells(image, calibration)
        board_truths.append(extract_board_truth(image, calibration))
        for cell in cells:
            sample_id = len(manifest)
            glyph_name = f"{sample_id:05d}.png"
            crop_name = f"{sample_id:05d}.png"
            Image.fromarray(cell.glyph).save(output / "glyphs" / glyph_name)
            Image.fromarray(crop_roi(image, cell.roi)).save(output / "crops" / crop_name)
            glyphs.append(cell.glyph)
            manifest.append(
                {
                    "id": sample_id,
                    "source": calibration.source,
                    "axis": cell.axis,
                    "line": cell.line,
                    "slot": cell.slot,
                    "roi": list(cell.roi),
                    "background_rgb": list(cell.background_rgb),
                    "foreground_pixels": cell.foreground_pixels,
                    "glyph": f"glyphs/{glyph_name}",
                    "crop": f"crops/{crop_name}",
                    "label": None,
                }
            )
    clusters = greedy_cluster_glyphs(glyphs, threshold=cluster_threshold)
    for cluster_index, members in enumerate(clusters):
        for member in members:
            manifest[member]["cluster"] = cluster_index
    accepted = 0
    rejected = 0
    template_labels = []
    templates = []
    template_holdout = None
    ground_truth = None
    if template_manifest is not None:
        templates = load_digit_templates(template_manifest)
        classifier = TemplateDigitClassifier(templates)
        template_labels = list(classifier.labels)
        template_holdout = _template_holdout_report(templates)
        for item, glyph in zip(manifest, glyphs):
            decision = classifier.decide(glyph)
            item["digit_candidates"] = [
                {"value": candidate.value, "score": round(candidate.score, 6)}
                for candidate in decision.candidates
            ]
            item["digit_accepted"] = decision.accepted
            item["digit_reason"] = decision.reason
            if decision.accepted:
                accepted += 1
            else:
                rejected += 1
        ground_truth = _ground_truth_report(manifest, glyphs, board_truths, templates)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cluster_labels = [
        {
            "cluster": cluster_index,
            "count": len(members),
            "label": None,
            "sample_ids": members[:16],
        }
        for cluster_index, members in enumerate(clusters)
    ]
    (output / "cluster_labels.json").write_text(
        json.dumps(cluster_labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "board_truth.json").write_text(
        json.dumps([truth.to_dict() for truth in board_truths], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if template_holdout is not None:
        (output / "template_holdout.json").write_text(
            json.dumps(template_holdout, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if ground_truth is not None:
        (output / "ground_truth_evaluation.json").write_text(
            json.dumps(ground_truth, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _contact_sheet(output, glyphs, clusters)
    summary = {
        "screenshots": len(calibrations),
        "samples": len(manifest),
        "clusters": len(clusters),
        "cluster_threshold": cluster_threshold,
        "template_labels": template_labels,
        "missing_template_labels": [label for label in range(1, 16) if label not in template_labels],
        "accepted_samples": accepted,
        "rejected_samples": rejected,
        "template_holdout_tested": template_holdout["tested"] if template_holdout else 0,
        "template_holdout_correct": template_holdout["correct"] if template_holdout else 0,
        "template_holdout_accepted_correct": (
            template_holdout["accepted_correct"] if template_holdout else 0
        ),
        "ground_truth_completed_screenshots": (
            ground_truth["completed_screenshots"] if ground_truth else 0
        ),
        "ground_truth_unique_solution_matches": (
            ground_truth["unique_solution_matches"] if ground_truth else 0
        ),
        "ground_truth_samples": ground_truth["truth_samples"] if ground_truth else 0,
        "ground_truth_top1_correct": ground_truth["top1_correct"] if ground_truth else 0,
        "ground_truth_top3_correct": ground_truth["top3_correct"] if ground_truth else 0,
        "ground_truth_accepted_correct": ground_truth["accepted_correct"] if ground_truth else 0,
        "ground_truth_accepted_incorrect": (
            ground_truth["accepted_incorrect"] if ground_truth else 0
        ),
        "output": str(output.resolve()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibrations",
        type=pathlib.Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "color_nonogram_screenshots.json",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=PROJECT_ROOT / "debug" / "color_nonogram_glyph_dataset",
    )
    parser.add_argument("--cluster-threshold", type=float, default=0.88)
    parser.add_argument(
        "--template-manifest",
        type=pathlib.Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "color_digit_templates.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(
                args.calibrations,
                args.output,
                args.cluster_threshold,
                args.template_manifest,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
