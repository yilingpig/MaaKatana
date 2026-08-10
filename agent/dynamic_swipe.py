import json

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction


_SWIPE_STATE: dict[str, int] = {}


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


def _capture(context: Context):
    try:
        image = context.tasker.controller.post_screencap().wait().get()
        return np.asarray(image) if image is not None else None
    except Exception as exc:
        print(f"dynamic swipe screenshot failed: {exc}")
        return None


def _to_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3 and array.shape[2] >= 3:
        channels = array[..., :3].astype(np.float32)
        return channels[..., 0] * 0.114 + channels[..., 1] * 0.587 + channels[..., 2] * 0.299
    return array.astype(np.float32).squeeze()


def _estimate_vertical_shift(before, after, roi, max_shift):
    if before is None or after is None:
        return None
    before_gray = _to_gray(before)
    after_gray = _to_gray(after)
    if before_gray.ndim != 2 or after_gray.ndim != 2:
        return None
    height = min(before_gray.shape[0], after_gray.shape[0])
    width = min(before_gray.shape[1], after_gray.shape[1])
    x, y, roi_width, roi_height = [int(value) for value in roi]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    right = min(width, x + max(1, roi_width))
    bottom = min(height, y + max(1, roi_height))
    if right - x < 32 or bottom - y < 64:
        return None
    before_crop = before_gray[y:bottom, x:right]
    after_crop = after_gray[y:bottom, x:right]
    stride = 4
    before_sample = before_crop[::stride, ::stride]
    after_sample = after_crop[::stride, ::stride]
    max_sample_shift = min(max(1, int(max_shift / stride)), before_sample.shape[0] - 2)
    scores = []
    for sample_shift in range(max_sample_shift + 1):
        if sample_shift == 0:
            left = before_sample
            right_sample = after_sample
        else:
            left = before_sample[sample_shift:, :]
            right_sample = after_sample[:-sample_shift, :]
        if left.shape != right_sample.shape or left.size == 0:
            continue
        scores.append((float(np.mean(np.abs(left - right_sample))), sample_shift))
    if not scores:
        return None
    scores.sort(key=lambda item: item[0])
    best_score, best_shift = scores[0]
    if best_score > 32:
        return None
    if best_shift == max_sample_shift and best_shift > 0:
        return (best_shift + 1) * stride
    return best_shift * stride


@AgentServer.custom_action("动态向上滑动")
class DynamicUpwardSwipe(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        param = _load_param(argv.custom_action_param)
        base_step = max(20, int(param.get("base_step", 80)))
        min_step = max(20, int(param.get("min_step", 45)))
        max_step = max(min_step, int(param.get("max_step", 100)))
        start_x = int(param.get("start_x", 640))
        start_y = int(param.get("start_y", 450))
        duration = max(250, int(param.get("duration", 400)))
        roi = tuple(param.get("roi", (0, 80, 1280, 560)))
        max_shift = max(max_step + 20, int(param.get("max_shift", 220)))
        requested_step = _SWIPE_STATE.get(argv.node_name, base_step)
        requested_step = max(min_step, min(max_step, requested_step))

        before = _capture(context)
        result = context.tasker.controller.post_swipe(
            start_x,
            start_y,
            start_x,
            start_y - requested_step,
            duration,
        ).wait()
        if not result.succeeded:
            print(f"{argv.node_name}: dynamic swipe failed")
            return False

        after = _capture(context)
        actual_shift = _estimate_vertical_shift(before, after, roi, max_shift)
        next_step = requested_step
        if actual_shift is not None:
            if actual_shift > requested_step * 1.35:
                next_step = max(min_step, int(requested_step * 0.65))
            elif actual_shift < requested_step * 0.55:
                next_step = min(max_step, requested_step + 10)
        _SWIPE_STATE[argv.node_name] = next_step
        print(
            f"{argv.node_name}: requested={requested_step}px "
            f"actual={actual_shift if actual_shift is not None else 'unknown'}px "
            f"next={next_step}px duration={duration}ms"
        )
        return True