import json
import re

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.pipeline import JOCR, JRecognitionType, JTemplateMatch


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
    if normalized in (f"{target}X", f"{target}x"):
        return "close", 0, normalized
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


def _best_countdown_result(detail, threshold: float):
    candidates = [] if detail is None else [
        getattr(detail, "best_result", None),
        *getattr(detail, "all_results", []),
        *getattr(detail, "filtered_results", []),
    ]
    best = None
    best_score = None
    for result in candidates:
        if result is None:
            continue
        try:
            score = float(getattr(result, "score", 0.0))
        except (TypeError, ValueError):
            continue
        if best_score is None or score > best_score:
            best = result
            best_score = score
    if best is None or best_score is None or best_score < threshold:
        return None, best_score
    return best, best_score


def _ocr_variants(image, roi):
    variants = [("original", image)]
    try:
        import cv2

        x, y, width, height = (int(value) for value in roi)
        if image is None or len(getattr(image, "shape", ())) < 2:
            return variants
        source = image
        if len(source.shape) == 2:
            gray = source
            source_bgr = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        else:
            source_bgr = source
            gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        crop = gray[y:y + height, x:x + width]
        if crop.size == 0:
            return variants

        gray_image = source_bgr.copy()
        gray_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        gray_image[y:y + height, x:x + width] = gray_bgr
        variants.append(("gray", gray_image))

        adaptive = cv2.adaptiveThreshold(
            crop,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            3,
        )
        adaptive_image = source_bgr.copy()
        adaptive_image[y:y + height, x:x + width] = cv2.cvtColor(
            adaptive,
            cv2.COLOR_GRAY2BGR,
        )
        variants.append(("adaptive", adaptive_image))
    except Exception:
        pass
    return variants



@AgentServer.custom_recognition("奖励倒计时有效确认")
class RewardCountdownRecognition(CustomRecognition):
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult | None:
        param = _load_param(argv.custom_recognition_param)
        templates = param.get("template", ["x秒后可得奖励.png"])
        if isinstance(templates, str):
            templates = [templates]
        roi = tuple(param.get("roi", (1075, 2, 205, 102)))
        threshold = float(param.get("threshold", 0.7))
        detail = context.run_recognition_direct(
            JRecognitionType.TemplateMatch,
            JTemplateMatch(template=templates, roi=roi, threshold=[0.0]),
            argv.image,
        )
        best, best_score = _best_countdown_result(detail, threshold)
        if best is not None:
            print(
                f"{argv.node_name}: countdown present "
                f"score={best_score:.6f} threshold={threshold:.6f}"
            )
            return CustomRecognition.AnalyzeResult(
                box=_to_box(getattr(best, "box", None), roi),
                detail={
                    "score": best_score,
                    "threshold": threshold,
                    "template": templates,
                },
            )
        print(
            f"{argv.node_name}: countdown absent "
            f"best_score={best_score!r} threshold={threshold:.6f}"
        )
        return None


@AgentServer.custom_recognition("奖励倒计时不存在确认")
class RewardCountdownAbsentRecognition(CustomRecognition):
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult | None:
        param = _load_param(argv.custom_recognition_param)
        templates = param.get("template", ["x秒后可得奖励.png"])
        if isinstance(templates, str):
            templates = [templates]
        roi = tuple(param.get("roi", (1075, 2, 205, 102)))
        threshold = float(param.get("threshold", 0.7))
        detail = context.run_recognition_direct(
            JRecognitionType.TemplateMatch,
            JTemplateMatch(template=templates, roi=roi, threshold=[0.0]),
            argv.image,
        )
        best, best_score = _best_countdown_result(detail, threshold)
        if best is not None:
            print(
                f"{argv.node_name}: countdown present; exclusion applies "
                f"score={best_score:.6f} threshold={threshold:.6f}"
            )
            return None
        print(
            f"{argv.node_name}: countdown absent; exclusion passes "
            f"best_score={best_score!r} threshold={threshold:.6f}"
        )
        return CustomRecognition.AnalyzeResult(
            box=_to_box(None, roi),
            detail={
                "countdown_present": False,
                "best_score": best_score,
                "threshold": threshold,
                "template": templates,
            },
        )

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
        variant_names = []
        for variant_name, image in _ocr_variants(argv.image, roi):
            variant_names.append(variant_name)
            detail = context.run_recognition_direct(
                JRecognitionType.OCR,
                JOCR(expected=[], roi=roi),
                image,
            )
            candidates = _collect_candidates(detail) if detail else []
            for text, box in candidates:
                matched = _match_target(text, target, max_mismatches)
                if matched is None:
                    continue
                mode, mismatches, normalized = matched
                print(
                    f"{argv.node_name}: reward text matched variant={variant_name} "
                    f"mode={mode} mismatches={mismatches} text={text!r}"
                )
                return CustomRecognition.AnalyzeResult(
                    box=_to_box(box, roi),
                    detail={
                        "target": target,
                        "text": text,
                        "normalized": normalized,
                        "mode": mode,
                        "mismatches": mismatches,
                        "variant": variant_name,
                    },
                )
        print(
            f"{argv.node_name}: reward text not matched "
            f"variants={variant_names!r}"
        )
        return None
