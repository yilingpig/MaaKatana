import json
import re
import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JOCR, JRecognitionType, JTemplateMatch


_PROCESSED: dict[str, set[str]] = {}
_PENDING: dict[str, str] = {}
_SWIPE_COUNTS: dict[str, int] = {}
_STAGE_STATE: dict[str, dict[str, bool | str]] = {}


def _load_param(raw) -> dict:
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
    return re.sub(r"[\s\W_]", "", str(text))


def _as_templates(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value or []]


def _matched(context: Context, image, target: dict) -> tuple[bool, list[str]]:
    templates = _as_templates(target.get("template"))
    if not templates:
        return False, []
    threshold = float(target.get("threshold", target.get("template_threshold", 0.7)))
    template_detail = context.run_recognition_direct(
        JRecognitionType.TemplateMatch,
        JTemplateMatch(
            template=templates,
            roi=_to_roi(target.get("roi")),
            threshold=[threshold],
        ),
        image,
    )
    if not template_detail or not template_detail.hit:
        return False, []

    source_text = _normalize(target.get("source_text", ""))
    target_text = _normalize(target.get("text", target.get("item", "")))
    results = list(getattr(template_detail, "all_results", []) or [])
    if not results:
        best_result = getattr(template_detail, "best_result", None)
        if best_result is not None:
            results = [best_result]
    if not results:
        results = [None]

    fallback_texts = []
    for result in results:
        candidate_box = getattr(result, "box", None) if result is not None else None
        ocr_roi = _to_roi(candidate_box) if candidate_box is not None else _to_roi(
            target.get("ocr_roi", target.get("roi"))
        )
        ocr_detail = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(expected=[], roi=ocr_roi),
            image,
        )
        texts = _collect_texts(ocr_detail) if ocr_detail else []
        fallback_texts.extend(text for text in texts if text not in fallback_texts)
        normalized_texts = [_normalize(text) for text in texts]
        target_hit = bool(target_text) and any(target_text in text for text in normalized_texts)
        source_hit = bool(source_text) and any(source_text in text for text in normalized_texts)
        matched = target_hit and (source_hit if source_text else True)
        if source_text:
            print(
                f"exchange candidate item={target.get('item')} box={candidate_box} "
                f"target_hit={target_hit} source_hit={source_hit} texts={texts!r}"
            )
        if matched:
            return True, texts
    return False, fallback_texts


def _template_hit(context: Context, image, target: dict) -> bool:
    templates = _as_templates(target.get("template"))
    if not templates:
        return False
    threshold = float(target.get("threshold", target.get("template_threshold", 0.7)))
    detail = context.run_recognition_direct(
        JRecognitionType.TemplateMatch,
        JTemplateMatch(
            template=templates,
            roi=_to_roi(target.get("roi")),
            threshold=[threshold],
        ),
        image,
    )
    return bool(detail and detail.hit)


def _capture(context: Context):
    return context.tasker.controller.post_screencap().wait().get()


def _clear_state(node_name: str):
    _PROCESSED.pop(node_name, None)
    _PENDING.pop(node_name, None)
    _SWIPE_COUNTS.pop(node_name, None)
    _STAGE_STATE.pop(node_name, None)


def _gate_hit(context: Context, image, gate: dict, label: str) -> bool:
    if not gate:
        return False
    target = dict(gate)
    target.setdefault("source_text", target.get("text", ""))
    matched, texts = _matched(context, image, target)
    print(f"exchange gate={label} matched={matched} texts={texts!r}")
    return matched


def _legacy_products(products: list[dict], processed: set[str]) -> list[dict]:
    return [
        product for product in products
        if str(product.get("item", "")) and str(product.get("item")) not in processed
    ]


def _gated_products(products: list[dict], processed: set[str], phase: str) -> list[dict]:
    sea_index = next(
        (index for index, product in enumerate(products) if product.get("item") == "海枣"),
        len(products),
    )
    if phase == "before_bedouin":
        candidates = products[:sea_index]
    elif phase == "search_bedouin":
        candidates = []
    elif phase == "after_bedouin":
        candidates = [products[sea_index]] if sea_index < len(products) else []
    else:
        candidates = products[sea_index + 1:]
    return [
        product for product in candidates
        if str(product.get("item", "")) and str(product.get("item")) not in processed
    ]


@AgentServer.custom_action("交易所商品扫描并跳转")
class ExchangeProductRouter(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        param = _load_param(argv.custom_action_param)
        node_name = argv.node_name
        if param.get("reset"):
            _clear_state(node_name)

        products = list(param.get("products", []))
        processed = _PROCESSED.setdefault(node_name, set())
        max_swipes = max(0, int(param.get("max_swipes", 8)))
        gated_max_swipes = max(0, int(param.get("gated_max_swipes", 20)))
        large_max_swipes = max(0, int(param.get("large_max_swipes", 8)))
        small_step = max(1, int(param.get("small_step", 100)))
        gated_step = max(1, int(param.get("gated_step", 150)))
        large_start_y = int(param.get("large_start_y", 600))
        large_end_y = int(param.get("large_end_y", 300))
        large_duration = max(1, int(param.get("large_duration", 600)))
        swipe_interval = max(0, int(param.get("swipe_interval", 900)))
        disappearance_confirmations = max(2, int(param.get("disappearance_confirmations", 2)))
        disappearance_interval = max(0, int(param.get("disappearance_interval", 250)))
        start_x = int(param.get("start_x", 640))
        start_y = int(param.get("start_y", 450))
        duration = max(1, int(param.get("duration", 400)))
        failure_node = str(param.get("failure_node", "交易所交换失败"))
        success_node = str(param.get("success_node", "交易所交换完成"))
        completion_gate = param.get("completion_gate") or {}
        bedouin_gate = param.get("bedouin_gate") or {}
        heretic_gate = param.get("heretic_gate") or {}
        gated = bool(bedouin_gate and heretic_gate)
        state = _STAGE_STATE.setdefault(
            node_name,
            {"phase": "before_bedouin", "bedouin": False, "heretic": False},
        )
        sea_index = next((index for index, product in enumerate(products) if product.get("item") == "海枣"), len(products))
        front_products = products[:sea_index]
        swipe_count = _SWIPE_COUNTS.get(node_name, 0)

        try:
            while True:
                image = _capture(context)
                shape = getattr(image, "shape", None)
                phase = str(state.get("phase", "before_bedouin"))
                print(
                    f"{node_name}: phase={phase} attempt={swipe_count} "
                    f"image_size={shape} swipes={swipe_count} processed={sorted(processed)}"
                )
                matched_product = None
                if image is not None:
                    pending_item = _PENDING.get(node_name)
                    if pending_item and pending_item not in processed:
                        pending_product = next(
                            (product for product in products if str(product.get("item")) == pending_item),
                            None,
                        )
                        if pending_product is not None:
                            absent = True
                            for confirmation in range(disappearance_confirmations):
                                if confirmation:
                                    if disappearance_interval:
                                        time.sleep(disappearance_interval / 1000)
                                    image = _capture(context)
                                if _template_hit(context, image, pending_product):
                                    absent = False
                                    break
                            if absent:
                                processed.add(pending_item)
                                _PENDING.pop(node_name, None)
                                print(
                                    f"{node_name}: confirmed exchanged item disappeared "
                                    f"item={pending_item} confirmations={disappearance_confirmations}"
                                )
                            else:
                                _PENDING.pop(node_name, None)
                    front_complete = gated and bool(front_products) and all(
                        str(product.get("item", "")) in processed for product in front_products
                    )
                    if gated and phase == "search_bedouin" and _gate_hit(
                        context, image, bedouin_gate, "贝多因人"
                    ):
                        state["bedouin"] = True
                        state["phase"] = "after_bedouin"
                        swipe_count = 0
                        _SWIPE_COUNTS[node_name] = 0
                        print(f"{node_name}: entered after_bedouin phase")
                        continue
                    if gated and phase == "after_bedouin" and _gate_hit(
                        context, image, heretic_gate, "异教徒"
                    ):
                        state["heretic"] = True
                        state["phase"] = "after_heretic"
                        swipe_count = 0
                        _SWIPE_COUNTS[node_name] = 0
                        print(f"{node_name}: entered after_heretic phase")
                        continue
                    candidates = (
                        _gated_products(products, processed, phase)
                        if gated
                        else _legacy_products(products, processed)
                    )
                    for product in candidates:
                        matched, texts = _matched(context, image, product)
                        if matched:
                            matched_product = product
                            print(
                                f"{node_name}: matched item={product.get('item')} "
                                f"phase={phase} texts={texts!r} next={product.get('next_node')}"
                            )
                            break

                if matched_product is not None:
                    item = str(matched_product["item"])
                    next_node = str(matched_product.get("next_node", ""))
                    _PENDING[node_name] = item
                    _SWIPE_COUNTS.pop(node_name, None)
                    context.override_next(
                        node_name, [next_node] if next_node else [failure_node]
                    )
                    return True

                if gated and phase == "before_bedouin" and completion_gate and _gate_hit(
                    context, image, completion_gate, "观看广告"
                ):
                    state["phase"] = "search_bedouin"
                    swipe_count = 0
                    _SWIPE_COUNTS[node_name] = 0
                    print(f"{node_name}: entered search_bedouin phase after completion marker")
                    continue

                if all(
                    str(product.get("item", "")) in processed
                    for product in products
                    if product.get("item")
                ):
                    _clear_state(node_name)
                    context.override_next(node_name, [success_node])
                    return True

                searching_bedouin = gated and phase == "search_bedouin"
                swipe_limit = (
                    large_max_swipes if searching_bedouin
                    else gated_max_swipes if gated and phase not in {"before_bedouin", "search_bedouin"} else max_swipes
                )
                if swipe_count >= swipe_limit:
                    _clear_state(node_name)
                    print(f"{node_name}: no match after {swipe_count} swipes; next={failure_node}")
                    context.override_next(node_name, [failure_node])
                    return True

                if searching_bedouin:
                    swipe_start_y = large_start_y
                    swipe_end_y = large_end_y
                    swipe_duration = large_duration
                    swipe_label = f"large {large_start_y}->{large_end_y}"
                else:
                    step = gated_step if gated and phase != "before_bedouin" else small_step
                    swipe_start_y = start_y
                    swipe_end_y = start_y - step
                    swipe_duration = duration
                    swipe_label = f"small {step}px phase={phase}"
                result = context.tasker.controller.post_swipe(
                    start_x, swipe_start_y, start_x, swipe_end_y, swipe_duration
                ).wait()
                if not result or not result.succeeded:
                    _clear_state(node_name)
                    print(f"{node_name}: swipe failed; next={failure_node}")
                    context.override_next(node_name, [failure_node])
                    return True
                swipe_count += 1
                _SWIPE_COUNTS[node_name] = swipe_count
                if swipe_interval:
                    time.sleep(swipe_interval / 1000)
                print(
                    f"{node_name}: swiped {swipe_label}; "
                    f"swipes={swipe_count}"
                )
        except Exception as exc:
            _clear_state(node_name)
            print(f"{node_name}: exchange scan failed: {exc}; next={failure_node}")
            context.override_next(node_name, [failure_node])
            return True