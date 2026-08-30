from __future__ import annotations

import datetime
import json
import pathlib

import numpy as np
from PIL import Image

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from color_nonogram_solver import (
    _colored_clues,
    _load_param,
    _palette_colors,
    _project_root,
    _solve_with_candidate_recovery,
    _verify_clue_color_match,
    detect_color_grid,
)


PROJECT_ROOT = _project_root(__file__)
DEFAULT_SUCCESS_NODE = "彩色数织1080p测试完成"
DEFAULT_FAILURE_NODE = "彩色数织1080p测试失败"


def _geometry_report(geometry) -> dict | None:
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


def _palette_report(palette) -> list[dict]:
    return [
        {
            "center": [int(color.center[0]), int(color.center[1])],
            "rgb": [int(round(value)) for value in color.rgb],
        }
        for color in palette
    ]


def _clues_report(lines) -> list[list[list[int]]]:
    return [
        [[int(color), int(value)] for color, value in line]
        for line in lines
    ]


def _slot_diagnostics_report(diagnostics) -> list[dict]:
    return [
        {
            "checked_count": len(item["checked"]),
            "checked_slots": sorted(int(slot) for slot in item["checked"]),
            "recognized_count": len(item["recognized"]),
            "recognized_slots": sorted(int(slot) for slot in item["recognized"]),
        }
        for item in diagnostics
    ]


def _color_totals_report(rows, columns, color_count: int) -> dict:
    row_totals = [
        sum(number for line in rows for color, number in line if color == index)
        for index in range(color_count)
    ]
    column_totals = [
        sum(number for line in columns for color, number in line if color == index)
        for index in range(color_count)
    ]
    return {
        "rows": row_totals,
        "columns": column_totals,
        "difference_rows_minus_columns": [
            row - column for row, column in zip(row_totals, column_totals)
        ],
    }


@AgentServer.custom_action("测试彩色数织1080p")
class TestColorNonogram1080pAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        param = _load_param(argv.custom_action_param)
        success_node = str(param.get("success_node", DEFAULT_SUCCESS_NODE))
        failure_node = str(param.get("failure_node", DEFAULT_FAILURE_NODE))
        mode = str(param.get("mode", "capture")).lower()
        phase = "screenshot"
        image = None
        output_path = "disabled"
        actual = None
        geometry = None
        palette = []
        rows = None
        columns = None
        solution = None
        row_slot_diagnostics = None
        column_slot_diagnostics = None
        report = {
            "status": "failure",
            "mode": mode,
            "phase": phase,
            "reason": None,
            "screenshot": output_path,
            "resolution": None,
            "grid": None,
            "palette": [],
            "rows": None,
            "columns": None,
            "solution": None,
            "clue_slots": {
                "rows": None,
                "columns": None,
            },
            "color_totals": None,
        }
        try:
            image = context.tasker.controller.post_screencap().wait().get()
            if image is None:
                raise RuntimeError("screenshot returned no image")
            image = np.asarray(image)
            expected = tuple(int(value) for value in param.get("expected_resolution", (1920, 1080)))
            actual = (int(image.shape[1]), int(image.shape[0]))
            report["resolution"] = list(actual)
            if actual != expected:
                raise ValueError(f"unexpected screenshot resolution: expected={expected}, actual={actual}")
            output_path = self._save_screenshot(image, param)
            report["screenshot"] = output_path
            phase = "grid"
            report["phase"] = phase
            geometry = detect_color_grid(image)
            report["grid"] = _geometry_report(geometry)
            if geometry is None:
                raise ValueError("unable to locate a 15x15 colored grid at 1920x1080")
            if geometry.rows != 15 or geometry.columns != 15:
                raise ValueError(f"expected a 15x15 colored grid, found {geometry}")
            phase = "palette"
            report["phase"] = phase
            palette = _palette_colors(image)
            report["palette"] = _palette_report(palette)
            if not 2 <= len(palette) <= int(param.get("max_palette_colors", 12)):
                raise ValueError(f"unexpected palette color count: {len(palette)}")
            print(
                f"color nonogram 1080p probe resolution={actual} "
                f"grid={geometry} palette={palette!r} screenshot={output_path}"
            )
            if mode == "capture":
                report["status"] = "success"
                report["phase"] = "complete"
                report["reason"] = "capture checks passed"
                report_path = self._write_diagnostic_report(report)
                print(f"color nonogram 1080p probe status=success report={report_path}", flush=True)
                context.override_next(argv.node_name, [success_node])
                return True
            if mode != "dry_run":
                raise ValueError(f"unsupported safe test mode: {mode}")
            phase = "color_match"
            report["phase"] = phase
            if not _verify_clue_color_match(context, image, geometry, palette):
                raise ValueError("Maa ColorMatch did not find any palette color in the clue area")
            phase = "clues"
            report["phase"] = phase
            extracted = _colored_clues(context, image, geometry, palette)
            if extracted is None:
                raise ValueError("colored clues are incomplete")
            (
                rows,
                columns,
                _,
                _,
                row_cache,
                column_cache,
                row_slot_diagnostics,
                column_slot_diagnostics,
            ) = extracted
            report["rows"] = _clues_report(rows)
            report["columns"] = _clues_report(columns)
            report["clue_slots"] = {
                "rows": _slot_diagnostics_report(row_slot_diagnostics),
                "columns": _slot_diagnostics_report(column_slot_diagnostics),
            }
            report["color_totals"] = _color_totals_report(rows, columns, len(palette))
            phase = "solve"
            report["phase"] = phase
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
            report["rows"] = _clues_report(rows)
            report["columns"] = _clues_report(columns)
            report["color_totals"] = _color_totals_report(rows, columns, len(palette))
            report["solution"] = [[int(value) for value in line] for line in solution]
            print(f"color nonogram 1080p dry-run rows={rows!r} columns={columns!r}")
            print(f"color nonogram 1080p dry-run solution={solution!r}")
            if bool(param.get("allow_paint", False)):
                raise ValueError("painting is disabled in the 1080p test entry")
            report["status"] = "success"
            report["phase"] = "complete"
            report["reason"] = "dry-run checks passed"
            report_path = self._write_diagnostic_report(report)
            print(f"color nonogram 1080p probe status=success report={report_path}", flush=True)
            context.override_next(argv.node_name, [success_node])
            return True
        except Exception as exc:
            report["status"] = "failure"
            report["phase"] = phase
            report["reason"] = str(exc)
            report["grid"] = _geometry_report(geometry)
            report["palette"] = _palette_report(palette)
            if actual is not None:
                report["resolution"] = list(actual)
            if rows is not None:
                report["rows"] = _clues_report(rows)
            if columns is not None:
                report["columns"] = _clues_report(columns)
            if solution is not None:
                report["solution"] = [[int(value) for value in line] for line in solution]
            if row_slot_diagnostics is not None and column_slot_diagnostics is not None:
                report["clue_slots"] = {
                    "rows": _slot_diagnostics_report(row_slot_diagnostics),
                    "columns": _slot_diagnostics_report(column_slot_diagnostics),
                }
            if rows is not None and columns is not None:
                report["color_totals"] = _color_totals_report(rows, columns, len(palette))
            try:
                report_path = self._write_diagnostic_report(report)
            except Exception as report_error:
                report_path = f"unavailable: {report_error}"
            print(
                f"color nonogram 1080p test failed status=failure phase={phase} "
                f"reason={exc} report={report_path}",
                flush=True,
            )
            context.override_next(argv.node_name, [failure_node])
            return True

    @staticmethod
    def _save_screenshot(image: np.ndarray, param: dict) -> str:
        if not bool(param.get("save_screenshot", True)):
            return "disabled"
        output_dir = PROJECT_ROOT / "debug" / "color_nonogram_live"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_dir / f"screencap_{timestamp}.png"
        Image.fromarray(image[:, :, :3].astype(np.uint8)).save(output_path)
        return str(output_path)

    @staticmethod
    def _write_diagnostic_report(report: dict) -> str:
        output_dir = PROJECT_ROOT / "debug" / "color_nonogram_live"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_path = output_dir / f"report_{timestamp}.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(report_path)
