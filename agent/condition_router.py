import json
import re

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
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


def _to_roi(value) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(item) for item in value)
        if len(values) == 4:
            return values
    except (TypeError, ValueError):
        pass
    return (0, 0, 0, 0)


def _collect_texts(detail) -> list[str]:
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


def _normalize(text: str) -> str:
    normalized = str(text).replace("×", "x").replace("X", "x")
    return re.sub(r"[\s\W_]", "", normalized)


@AgentServer.custom_action("图文条件并跳转")
class ImageTextRouter(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        param = _load_param(argv.custom_action_param)
        templates = param.get("template", [])
        if isinstance(templates, str):
            templates = [templates]
        template_threshold = float(param.get("template_threshold", param.get("threshold", 0.8)))
        template_roi = _to_roi(param.get("template_roi", param.get("roi", (0, 0, 0, 0))))
        target_text = str(param.get("text", ""))
        ocr_roi = _to_roi(param.get("ocr_roi", param.get("roi", (0, 0, 0, 0))))
        hit_node = str(param.get("hit_node", ""))
        miss_node = str(param.get("miss_node", ""))
        require_both = bool(param.get("require_both", False))

        template_hit = False
        ocr_hit = False
        candidates = []
        try:
            image = context.tasker.controller.post_screencap().wait().get()
            if image is not None and templates:
                template_detail = context.run_recognition_direct(
                    JRecognitionType.TemplateMatch,
                    JTemplateMatch(
                        template=templates,
                        roi=template_roi,
                        threshold=[template_threshold],
                    ),
                    image,
                )
                template_hit = bool(template_detail and template_detail.hit)

            if image is not None and target_text:
                ocr_detail = context.run_recognition_direct(
                    JRecognitionType.OCR,
                    JOCR(expected=[], roi=ocr_roi),
                    image,
                )
                candidates = _collect_texts(ocr_detail) if ocr_detail else []
                merged = "".join(candidates)
                normalized_target = _normalize(target_text)
                ocr_hit = any(
                    normalized_target in _normalize(text)
                    for text in [*candidates, merged]
                )

            matched = template_hit and ocr_hit if require_both else template_hit or ocr_hit
            next_node = hit_node if matched else miss_node
            print(
                f"{argv.node_name}: template_hit={template_hit} "
                f"ocr_hit={ocr_hit} candidates={candidates!r}; next={next_node}"
            )
            context.override_next(argv.node_name, [next_node] if next_node else [])
            return True
        except Exception as exc:
            print(f"{argv.node_name}: image/text routing failed: {exc}; next={miss_node}")
            context.override_next(argv.node_name, [miss_node] if miss_node else [])
            return True
