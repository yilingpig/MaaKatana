import json

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from dynamic_swipe import _capture, _estimate_vertical_shift


_STATE: dict[str, dict] = {}


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


def _swipe(context: Context, direction: str, start_x: int, start_y: int, step: int, duration: int):
    end_y = start_y - step if direction == 'up' else start_y + step
    return context.tasker.controller.post_swipe(
        start_x,
        start_y,
        start_x,
        end_y,
        duration,
    ).wait()


@AgentServer.custom_action("动态往返滑动")
class DynamicBidirectionalSwipe(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        param = _load_param(argv.custom_action_param)
        base_step = max(20, int(param.get("base_step", 80)))
        min_step = max(20, int(param.get("min_step", 45)))
        max_step = max(min_step, int(param.get("max_step", 100)))
        return_step = max(20, int(param.get("return_step", base_step)))
        max_up_steps = max(1, int(param.get("max_up_steps", 6)))
        return_steps = max(1, int(param.get("return_steps", 4)))
        stall_limit = max(1, int(param.get("stall_limit", 2)))
        stall_threshold = max(1, int(param.get("stall_threshold", 25)))
        start_x = int(param.get("start_x", 640))
        start_y = int(param.get("start_y", 450))
        duration = max(250, int(param.get("duration", 400)))
        roi = tuple(param.get("roi", (0, 80, 1280, 560)))
        max_shift = max(max_step + 20, int(param.get("max_shift", 220)))
        state = _STATE.setdefault(
            argv.node_name,
            {
                "direction": "up",
                "step": base_step,
                "up_steps": 0,
                "stall_count": 0,
                "return_steps": 0,
                "cycles": 0,
            },
        )
        direction = state["direction"]
        requested_step = return_step if direction == "down" else state["step"]
        requested_step = max(min_step, min(max_step, requested_step))
        before = _capture(context)
        result = _swipe(context, direction, start_x, start_y, requested_step, duration)
        if not result.succeeded:
            print(f"{argv.node_name}: bidirectional {direction} swipe failed")
            return False
        after = _capture(context)
        actual_shift = _estimate_vertical_shift(before, after, roi, max_shift)

        if direction == "down":
            state["return_steps"] += 1
            if state["return_steps"] >= return_steps:
                state["direction"] = "up"
                state["up_steps"] = 0
                state["stall_count"] = 0
                state["return_steps"] = 0
                state["cycles"] += 1
        else:
            state["up_steps"] += 1
            if actual_shift is not None and actual_shift < stall_threshold:
                state["stall_count"] += 1
            else:
                state["stall_count"] = 0
            if actual_shift is not None and actual_shift > requested_step * 1.35:
                state["step"] = max(min_step, int(requested_step * 0.65))
            elif actual_shift is not None and actual_shift < requested_step * 0.55:
                state["step"] = min(max_step, requested_step + 10)
            if state["up_steps"] >= max_up_steps or state["stall_count"] >= stall_limit:
                state["direction"] = "down"
                state["return_steps"] = 0

        print(
            f"{argv.node_name}: direction={direction} requested={requested_step}px "
            f"actual={actual_shift if actual_shift is not None else 'unknown'}px "
            f"next_direction={state['direction']} cycle={state['cycles']}"
        )
        return True