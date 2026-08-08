import json

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.pipeline import JRecognitionType, JTemplateMatch


def _load_param(raw: str) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


@AgentServer.custom_action("检查建筑并跳转")
class BuildingCheckRouter(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        param = _load_param(argv.custom_action_param)
        templates = param.get("template", [])
        if isinstance(templates, str):
            templates = [templates]
        hit_node = param.get("hit_node", "")
        miss_node = param.get("miss_node", "")
        threshold = float(param.get("threshold", 0.7))
        roi = tuple(param.get("roi", (0, 0, 0, 0)))

        if not templates:
            print(f"{argv.node_name}: no building template configured")
            context.override_next(argv.node_name, [miss_node] if miss_node else [])
            return True

        try:
            image = context.tasker.controller.post_screencap().wait().get()
            detail = context.run_recognition_direct(
                JRecognitionType.TemplateMatch,
                JTemplateMatch(template=templates, roi=roi, threshold=[threshold]),
                image,
            )
            if detail and detail.hit:
                print(f"{argv.node_name}: building found {templates}; next={hit_node}")
                context.override_next(argv.node_name, [hit_node] if hit_node else [])
            else:
                print(f"{argv.node_name}: building not found {templates}; next={miss_node}")
                context.override_next(argv.node_name, [miss_node] if miss_node else [])
            return True
        except Exception as exc:
            print(f"{argv.node_name}: building check failed: {exc}; next={miss_node}")
            context.override_next(argv.node_name, [miss_node] if miss_node else [])
            return True
