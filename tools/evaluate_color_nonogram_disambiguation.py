from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
AGENT_ROOT = PROJECT_ROOT / "agent"
sys.path.insert(0, str(AGENT_ROOT))

from color_nonogram_disambiguation import (
    ClueCandidate,
    ClueObservation,
    disambiguate_puzzle,
)


DATASET_ROOT = PROJECT_ROOT / "debug" / "color_nonogram_glyph_dataset"


def _nearest_color(rgb: list[int], palette: list[list[int]]) -> int:
    distances = [np.linalg.norm(np.asarray(rgb) - np.asarray(color)) for color in palette]
    return int(np.argmin(distances))


def evaluate() -> dict:
    evaluation = json.loads((DATASET_ROOT / "ground_truth_evaluation.json").read_text(encoding="utf-8"))
    truths = {
        item["source"]: item
        for item in json.loads((DATASET_ROOT / "board_truth.json").read_text(encoding="utf-8"))
        if item["completed"]
    }
    reports = []
    for source, truth in truths.items():
        rows = [[] for _ in range(15)]
        columns = [[] for _ in range(15)]
        for item in evaluation["results"]:
            if item.get("source") != source or not item.get("matched"):
                continue
            color = _nearest_color(item["observed_background_rgb"], truth["palette_rgb"])
            observation = ClueObservation(
                axis=item["axis"],
                line=item["line"],
                slot=item["slot"],
                candidates=tuple(
                    ClueCandidate(candidate["value"], color, candidate["score"])
                    for candidate in item["candidates"]
                ),
            )
            target = rows if item["axis"] == "row" else columns
            target[item["line"]].append(observation)
        result = disambiguate_puzzle(
            rows,
            columns,
            width=15,
            height=15,
            color_count=len(truth["palette_rgb"]),
            maximum_attempts=2000,
            empty_rows={index for index, line in enumerate(truth["rows"]) if not line},
            empty_columns={index for index, line in enumerate(truth["columns"]) if not line},
        )
        reports.append(
            {
                "source": source,
                "status": result.status,
                "attempts": result.attempts,
                "reason": result.reason,
                "row_option_counts": list(result.row_option_counts),
                "column_option_counts": list(result.column_option_counts),
                "returned_executable_solution": result.solution is not None,
            }
        )
    return {
        "mode": "verified-color-top3-candidate-search",
        "screenshots": reports,
        "safe_unique_screenshots": sum(item["status"] == "unique" for item in reports),
        "ambiguous_screenshots": sum(item["status"] == "multiple" for item in reports),
        "inconclusive_screenshots": sum(item["status"] == "inconclusive" for item in reports),
    }


def main() -> None:
    report = evaluate()
    output = DATASET_ROOT / "disambiguation_evaluation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
