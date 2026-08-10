import json
import re

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.pipeline import JOCR, JRecognitionType


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


def _normalize(text: str) -> str:
    return re.sub(r"[\s\W_]", "", text)


def _to_box(value, fallback) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(item) for item in value)
        if len(values) == 4:
            return values
    except (TypeError, ValueError):
        pass
    return tuple(int(item) for item in fallback)


def _match_target(text: str, target: str, max_mismatches: int):
    normalized = _normalize(text)
    if re.fullmatch(re.escape(target), normalized):
        return "exact", 0, normalized
    if len(normalized) != len(target):
        return None
    mismatches = sum(left != right for left, right in zip(normalized, target))
    if mismatches <= max_mismatches:
        return "near", mismatches, normalized
    return None


def _collect_candidates(detail):
    candidates = []
    seen = set()
    results = [
        getattr(detail, "best_result", None),
        *getattr(detail, "all_results", []),
        *getattr(detail, "filtered_results", []),
    ]
    for result in results:
        if result is None:
            continue
        text = getattr(result, "text", "")
        if text and text not in seen:
            seen.add(text)
            candidates.append((text, getattr(result, "box", None)))
    merged = "".join(text for text, _ in candidates)
    if merged and merged not in seen:
        candidates.append((merged, None))
    return candidates


@AgentServer.custom_recognition("奖励已发放确认")
class RewardIssuedRecognition(CustomRecognition):
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult | None:
        param = _load_param(argv.custom_recognition_param)
        target = str(param.get("target", "奖励已发放"))
        roi = tuple(param.get("roi", (1075, 2, 205, 102)))
        max_mismatches = max(0, int(param.get("max_mismatches", 1)))
        detail = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(expected=[], roi=roi),
            argv.image,
        )
        candidates = _collect_candidates(detail) if detail else []
        for text, box in candidates:
            matched = _match_target(text, target, max_mismatches)
            if matched is None:
                continue
            mode, mismatches, normalized = matched
            print(
                f"{argv.node_name}: reward text matched mode={mode} "
                f"mismatches={mismatches} text={text!r}"
            )
            return CustomRecognition.AnalyzeResult(
                box=_to_box(box, roi),
                detail={
                    "target": target,
                    "text": text,
                    "normalized": normalized,
                    "mode": mode,
                    "mismatches": mismatches,
                },
            )
        print(f"{argv.node_name}: reward text not matched candidates={[text for text, _ in candidates]!r}")
        return None