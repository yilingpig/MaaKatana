import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from maa.context import Context
from maa.controller import AdbController
from maa.custom_action import CustomAction
from maa.pipeline import JOCR, JRecognitionType
from maa.resource import Resource
from maa.tasker import Tasker
from maa.toolkit import Toolkit

from quantity_router import ManufacturingQuantityRouter


QUANTITY_ROI = (384, 109, 515, 515)


class QuantityOcrDump(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        image = context.tasker.controller.post_screencap().wait().get()
        detail = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(expected=[], roi=QUANTITY_ROI),
            image,
        )
        print(f"OCR hit={getattr(detail, 'hit', None)} roi={list(QUANTITY_ROI)}")
        if detail is None:
            return False
        for label in ("best_result", "all_results", "filtered_results"):
            value = getattr(detail, label, [])
            results = value if isinstance(value, list) else [value]
            texts = [getattr(result, "text", "") for result in results if result]
            print(f"{label}: {texts!r}")
        return True


def get_device(adb_path: str | None, address: str | None):
    devices = Toolkit.find_adb_devices(adb_path)
    if address:
        devices = [device for device in devices if device.address == address]
    if not devices:
        raise RuntimeError("未发现可用的 ADB 模拟器，请先确认 adb devices 能看到模拟器")
    if len(devices) > 1 and not address:
        names = ", ".join(f"{d.name} [{d.address}]" for d in devices)
        raise RuntimeError(f"发现多个模拟器，请使用 --address 指定一个：{names}")
    return devices[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 MaaFramework 自定义动作调试入口")
    parser.add_argument("--entry", default="寻找迫击炮", help="要执行的 pipeline 入口节点")
    parser.add_argument("--address", help="ADB 地址，例如 emulator-5554")
    parser.add_argument("--adb-path", help="adb.exe 路径；默认自动发现")
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="只识别数量 ROI 并打印原始文本，不点击、不跳转",
    )
    args = parser.parse_args()

    Toolkit.init_option(ROOT)
    device = get_device(args.adb_path, args.address)
    print(f"使用设备：{device.name} [{device.address}]")

    resource = Resource()
    if not resource.post_bundle(ROOT / "assets" / "resource").wait().succeeded:
        raise RuntimeError("资源加载失败")

    controller = AdbController(
        device.adb_path,
        device.address,
        device.screencap_methods,
        device.input_methods,
        device.config,
    )
    if not controller.post_connection().wait().succeeded:
        raise RuntimeError("模拟器连接失败")

    tasker = Tasker()
    if not tasker.bind(resource, controller):
        raise RuntimeError("Tasker 绑定资源失败")

    if args.ocr_only:
        entry = "__调试数量OCR"
        if not resource.register_custom_action("调试数量OCR", QuantityOcrDump()):
            raise RuntimeError("调试数量 OCR 注册失败")
        override = {
            entry: {
                "recognition": {"type": "DirectHit"},
                "action": {
                    "type": "Custom",
                    "param": {
                        "custom_action": "调试数量OCR",
                        "custom_action_param": "{}",
                    },
                },
            },
        }
        print("开始只读 OCR 调试，不会执行点击或跳转")
        job = tasker.post_task(entry, override)
    else:
        if not resource.register_custom_action(
            "制造数量路由", ManufacturingQuantityRouter()
        ):
            raise RuntimeError("制造数量路由注册失败")
        print(f"开始执行入口：{args.entry}")
        job = tasker.post_task(args.entry)

    job.wait()
    print(f"任务完成：success={job.succeeded}, task_id={job.job_id}")
    return 0 if job.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())

