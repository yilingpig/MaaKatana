import json

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JOCR, JRecognitionType, JTemplateMatch


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


def _templates(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value or []]


def _box_from_detail(detail):
    if detail is None or not detail.hit:
        return None
    result = getattr(detail, "best_result", None)
    if result is None:
        results = getattr(detail, "all_results", [])
        result = results[0] if results else None
    box = getattr(result, "box", None) if result is not None else None
    if box is None:
        return None
    try:
        values = tuple(int(value) for value in box)
    except (TypeError, ValueError):
        return None
    return values if len(values) == 4 else None


def _luminance(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3 and array.shape[2] >= 3:
        channels = array[..., :3].astype(np.float32)
        return channels[..., 0] * 0.114 + channels[..., 1] * 0.587 + channels[..., 2] * 0.299
    return array.astype(np.float32).squeeze()


def _longest_run(row: np.ndarray, threshold: float) -> tuple[int, int] | None:
    dark = row < threshold
    padded = np.concatenate(([False], dark, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    if starts.size == 0:
        return None
    lengths = ends - starts
    index = int(np.argmax(lengths))
    return int(starts[index]), int(ends[index])


def _locate_slider(image: np.ndarray, window_box: tuple[int, int, int, int], threshold: float):
    gray = _luminance(image)
    if gray.ndim != 2:
        return None
    screen_height, screen_width = gray.shape
    window_x, window_y, window_width, window_height = window_box
    left = max(0, window_x)
    top = max(0, window_y)
    right = min(screen_width, window_x + window_width)
    bottom = min(screen_height, window_y + window_height)
    if right - left < 200 or bottom - top < 180:
        return None

    band_top = max(top, bottom - 170)
    band_bottom = min(bottom - 65, band_top + 120)
    minimum_width = max(100, int((right - left) * 0.45))
    maximum_width = int((right - left) * 0.9)
    candidates = []
    for y in range(band_top, band_bottom):
        run = _longest_run(gray[y, left:right], threshold)
        if run is None:
            continue
        start, end = run
        width = end - start
        if minimum_width <= width <= maximum_width:
            candidates.append((width, y, left + start, left + end - 1))
    if not candidates:
        return None

    expected_y = bottom - 120
    candidates.sort(key=lambda item: (abs(item[1] - expected_y), -item[0]))
    _, y, start_x, end_x = candidates[0]
    if end_x - start_x < minimum_width:
        return None
    return start_x + 3, y, end_x - 3, y


@AgentServer.custom_action("\u52a8\u6001\u5b9a\u4f4d\u5e76\u6ed1\u52a8\u4ea4\u6613\u6ed1\u6761")
class ExchangeSliderSwipe(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        param = _load_param(argv.custom_action_param)
        template = _templates(param.get("window_template"))
        threshold = float(param.get("window_threshold", 0.7))
        next_node = str(param.get("next_node", ""))
        failure_node = str(param.get("failure_node", "\u626b\u63cf\u4ea4\u6613\u6240\u5546\u54c1"))
        duration = max(200, int(param.get("duration", 400)))
        dark_threshold = float(param.get("dark_threshold", 175))
        if not template:
            print(f"{argv.node_name}: missing window_template; next={failure_node}")
            context.override_next(argv.node_name, [failure_node])
            return True

        try:
            image = context.tasker.controller.post_screencap().wait().get()
            if image is None:
                raise RuntimeError("screenshot unavailable")
            detail = context.run_recognition_direct(
                JRecognitionType.TemplateMatch,
                JTemplateMatch(template=template, threshold=[threshold]),
                image,
            )
            window_box = _box_from_detail(detail)
            if window_box is None:
                raise RuntimeError("exchange window not found")
            gesture = _locate_slider(image, window_box, dark_threshold)
            if gesture is None:
                raise RuntimeError(f"slider not found in window={window_box}")
            start_x, start_y, end_x, end_y = gesture
            result = context.tasker.controller.post_swipe(
                start_x, start_y, end_x, end_y, duration
            ).wait()
            if not result or not result.succeeded:
                raise RuntimeError("slider swipe failed")
            print(
                f"{argv.node_name}: window={window_box} "
                f"slider=({start_x},{start_y})->({end_x},{end_y})"
            )
            if next_node:
                context.override_next(argv.node_name, [next_node])
            return True
        except Exception as exc:
            print(f"{argv.node_name}: {exc}; next={failure_node}")
            context.override_next(argv.node_name, [failure_node])
            return True





def _result_boxes(detail):
    if detail is None:
        return []
    results = []
    for result in getattr(detail, "all_results", []):
        box = getattr(result, "box", None)
        if box is None:
            continue
        try:
            values = tuple(int(value) for value in box)
        except (TypeError, ValueError):
            continue
        if len(values) == 4:
            results.append(values)
    if not results:
        box = _box_from_detail(detail)
        if box is not None:
            results.append(box)
    return results


def _expand_box(box, image_shape, padding=5):
    height, width = image_shape[:2]
    x, y, box_width, box_height = box
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(width, x + box_width + padding)
    bottom = min(height, y + box_height + padding)
    return left, top, max(1, right - left), max(1, bottom - top)


@AgentServer.custom_action("\u52a8\u6001\u5b9a\u4f4d\u5e76\u8bc6\u522b\u8d2d\u4e70\u6309\u94ae")
class ExchangePurchaseButton(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        param = _load_param(argv.custom_action_param)
        template = _templates(param.get("window_template"))
        threshold = float(param.get("window_threshold", 0.7))
        button_threshold = float(param.get("button_threshold", 0.7))
        next_node = str(param.get("next_node", "\u626b\u63cf\u4ea4\u6613\u6240\u5546\u54c1"))
        failure_node = str(param.get("failure_node", "\u626b\u63cf\u4ea4\u6613\u6240\u5546\u54c1"))
        if not template:
            print(f"{argv.node_name}: missing window_template; next={failure_node}")
            context.override_next(argv.node_name, [failure_node])
            return True

        try:
            image = context.tasker.controller.post_screencap().wait().get()
            if image is None:
                raise RuntimeError("screenshot unavailable")
            window_detail = context.run_recognition_direct(
                JRecognitionType.TemplateMatch,
                JTemplateMatch(template=template, threshold=[threshold]),
                image,
            )
            window_box = _box_from_detail(window_detail)
            if window_box is None:
                raise RuntimeError("exchange window not found")
            button_detail = context.run_recognition_direct(
                JRecognitionType.TemplateMatch,
                JTemplateMatch(
                    template=[str(param.get("button_template", "购买按钮.png"))],
                    roi=window_box,
                    threshold=[button_threshold],
                ),
                image,
            )
            boxes = _result_boxes(button_detail)
            if not boxes:
                raise RuntimeError(f"purchase button not found in window={window_box}")
            window_x, window_y, window_width, window_height = window_box
            boxes = [
                box for box in boxes
                if box[0] >= window_x
                and box[1] >= window_y
                and box[0] + box[2] <= window_x + window_width
                and box[1] + box[3] <= window_y + window_height
            ]
            if not boxes:
                raise RuntimeError(f"purchase button outside window={window_box}")
            button_box = max(boxes, key=lambda box: (box[0], box[1]))
            ocr_box = _expand_box(button_box, np.asarray(image).shape)
            ocr_detail = context.run_recognition_direct(
                JRecognitionType.OCR,
                JOCR(expected=["\u8d2d\u4e70"], roi=ocr_box),
                image,
            )
            if ocr_detail is None or not ocr_detail.hit:
                raise RuntimeError(f"purchase OCR failed button={button_box} roi={ocr_box}")
            x, y, width, height = button_box
            click_result = context.tasker.controller.post_click(
                x + width // 2, y + height // 2
            ).wait()
            if not click_result or not click_result.succeeded:
                raise RuntimeError(f"purchase click failed button={button_box}")
            print(
                f"{argv.node_name}: window={window_box} button={button_box} "
                f"ocr_roi={ocr_box} clicked=({x + width // 2},{y + height // 2})"
            )
            context.override_next(argv.node_name, [next_node])
            return True
        except Exception as exc:
            print(f"{argv.node_name}: {exc}; next={failure_node}")
            context.override_next(argv.node_name, [failure_node])
            return True
