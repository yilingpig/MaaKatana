import json
import re
import time

from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JOCR, JRecognitionType


QUANTITY_PATTERN = re.compile(r"(?:[×xX*]\s*(\d+))?.*?(\d+)\s*/\s*(\d+)", re.S)
DEFAULT_ROI = (384, 109, 515, 515)


def _load_param(raw: str) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _extract_texts(detail) -> list[str]:
    texts = []
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
            texts.append(text)
    return texts


class ManufacturingQuantityRouter(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        param = _load_param(argv.custom_action_param)
        build_node = param.get("build_node", "")
        next_node = param.get("next_node", "")
        roi = tuple(param.get("roi", DEFAULT_ROI))
        product = param.get("product", argv.node_name)

        try:
            image = context.tasker.controller.post_screencap().wait().get()
            detail = context.run_recognition_direct(
                JRecognitionType.OCR,
                JOCR(expected=[], roi=roi),
                image,
            )
            candidates = _extract_texts(detail) if detail and detail.hit else []
            if candidates:
                candidates.append(" ".join(candidates))

            decision = None
            for text in candidates:
                match = QUANTITY_PATTERN.search(text)
                if not match:
                    continue
                extra = int(match.group(1) or 0)
                current = int(match.group(2))
                limit = int(match.group(3))
                decision = {
                    "text": text,
                    "extra": extra,
                    "current": current,
                    "limit": limit,
                    "skip": current >= limit or current + extra > limit,
                }
                break

            if decision is None:
                print(f"{product}: OCR candidates={candidates!r}")
                print(f"{product}: quantity OCR failed; skip safely")
                context.tasker.controller.post_click_key(4).wait()
                time.sleep(0.5)
                context.override_next(argv.node_name, [next_node] if next_node else [])
                return True

            print(f"{product}: {decision}")
            if decision["skip"]:
                context.tasker.controller.post_click_key(4).wait()
                time.sleep(0.5)
                context.override_next(argv.node_name, [next_node] if next_node else [])
            else:
                context.override_next(argv.node_name, [build_node] if build_node else [])
            return True
        except Exception as exc:
            print(f"{product}: quantity routing failed: {exc}")
            context.tasker.controller.post_click_key(4).wait()
            time.sleep(0.5)
            context.override_next(argv.node_name, [next_node] if next_node else [])
            return True

