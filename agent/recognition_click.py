from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction


@AgentServer.custom_action("点击识别框中心")
class ClickRecognitionCenter(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        x, y, width, height = argv.box
        if width <= 0 or height <= 0:
            print(f"{argv.node_name}: invalid recognition box {argv.box}")
            return False

        target_x = x + width // 2
        target_y = y + height // 2
        result = context.tasker.controller.post_click(target_x, target_y).wait()
        print(
            f"{argv.node_name}: click recognition center "
            f"box={argv.box} point=({target_x}, {target_y}) success={result.succeeded}"
        )
        return result.succeeded
