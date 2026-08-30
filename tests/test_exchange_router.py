import json
import unittest
from types import SimpleNamespace

import exchange_router


class _Result:
    def __init__(self, succeeded=True):
        self.succeeded = succeeded


class _Detail:
    def __init__(self, hit=False, texts=(), box=None, boxes=None):
        self.hit = hit
        results = [SimpleNamespace(text=text, box=(boxes[index] if boxes else box)) for index, text in enumerate(texts)]
        self.best_result = results[0] if results else None
        self.all_results = results
        self.filtered_results = []


class _Controller:
    def __init__(self, images):
        self.images = iter(images)
        self.swipes = []

    def post_screencap(self):
        image = next(self.images)
        return SimpleNamespace(wait=lambda: SimpleNamespace(get=lambda: image))

    def post_swipe(self, *args):
        self.swipes.append(args)
        return SimpleNamespace(wait=lambda: _Result())


class _Context:
    def __init__(self, images, recognitions):
        self.tasker = SimpleNamespace(controller=_Controller(images))
        self.recognitions = iter(recognitions)
        self.routes = []
        self.recognition_calls = 0
        self.recognition_args = []

    def run_recognition_direct(self, *args):
        self.recognition_calls += 1
        self.recognition_args.append(args)
        return next(self.recognitions)

    def override_next(self, node_name, nodes):
        self.routes.append((node_name, nodes))


def _argv(param, node="测试扫描器"):
    return SimpleNamespace(node_name=node, custom_action_param=json.dumps(param))


class ExchangeRouterTests(unittest.TestCase):
    PRODUCTS = [
        {"item": "木梁", "template": "交换木梁.png", "text": "木梁", "next_node": "交换木梁"},
        {"item": "钢铁", "template": "交换钢铁.png", "text": "钢铁", "next_node": "交换钢铁"},
    ]
    GATED_PRODUCTS = [
        {"item": "化学品", "template": "交换化学品.png", "text": "化学品", "next_node": "交换化学品"},
        {"item": "海枣", "template": "交换海枣.png", "text": "海枣", "next_node": "交换海枣"},
        {"item": "木材", "template": "加密币交换木材.png", "text": "木材", "source_text": "加密币", "next_node": "交换木材"},
    ]
    GATES = {
        "completion_gate": {"template": "观看广告字样.png", "text": "观看广告"},
        "bedouin_gate": {"template": "贝多因人.png", "text": "贝多因人"},
        "heretic_gate": {"template": "异教徒.png", "text": "异教徒"},
    }

    def setUp(self):
        exchange_router._PROCESSED.clear()
        exchange_router._PENDING.clear()
        exchange_router._SWIPE_COUNTS.clear()
        exchange_router._STAGE_STATE.clear()

    def test_image_and_ocr_hit_routes_first_product(self):
        context = _Context(
            [object()],
            [_Detail(hit=True), _Detail(texts=("木梁",))],
        )
        result = exchange_router.ExchangeProductRouter().run(
            context,
            _argv({"products": self.PRODUCTS, "max_swipes": 0}),
        )
        self.assertTrue(result)
        self.assertEqual(context.routes, [("测试扫描器", ["交换木梁"])])
        self.assertEqual(context.tasker.controller.swipes, [])

    def test_no_hit_swipes_then_scans_again(self):
        context = _Context(
            [object(), object()],
            [_Detail(), _Detail(), _Detail(), _Detail(hit=True), _Detail(texts=("钢铁",))],
        )
        exchange_router.ExchangeProductRouter().run(
            context,
            _argv({"products": self.PRODUCTS, "max_swipes": 1}),
        )
        self.assertEqual(context.routes, [("测试扫描器", ["交换钢铁"])])
        self.assertEqual(len(context.tasker.controller.swipes), 1)
        self.assertEqual(context.tasker.controller.swipes[0][3], 350)

    def test_reaching_max_swipes_routes_failure(self):
        context = _Context([object(), object()], [_Detail(), _Detail()])
        exchange_router.ExchangeProductRouter().run(
            context,
            _argv({"products": self.PRODUCTS, "max_swipes": 0, "failure_node": "失败"}),
        )
        self.assertEqual(context.routes, [("测试扫描器", ["失败"])])

    def test_processed_product_is_skipped(self):
        first = _Context([object()], [_Detail(hit=True), _Detail(texts=("木梁",))])
        action = exchange_router.ExchangeProductRouter()
        action.run(first, _argv({"products": self.PRODUCTS, "max_swipes": 0}))

        second = _Context(
            [object(), object(), object()],
            [_Detail(), _Detail(), _Detail(hit=True), _Detail(texts=("钢铁",))],
        )
        action.run(second, _argv({"products": self.PRODUCTS, "max_swipes": 0}))
        self.assertEqual(second.routes, [("测试扫描器", ["交换钢铁"])])

    def test_bedouin_gate_must_match_image_and_ocr_before_sea(self):
        exchange_router._PROCESSED["测试扫描器"] = {"化学品"}
        context = _Context([object(), object(), object()], [
            _Detail(hit=True), _Detail(texts=("观看广告",)),
            _Detail(hit=True), _Detail(texts=("贝多因人",)),
            _Detail(hit=True), _Detail(texts=("海枣",)),
        ])
        exchange_router.ExchangeProductRouter().run(
            context,
            _argv({"products": self.GATED_PRODUCTS, **self.GATES, "max_swipes": 0}),
        )
        self.assertEqual(context.routes, [("测试扫描器", ["交易所交换失败"])])
        self.assertEqual(exchange_router._STAGE_STATE, {})

    def test_gate_phase_state_persists_and_uses_150px_swipe(self):
        node = "阶段滑动测试"
        exchange_router._PROCESSED[node] = {"海枣"}
        exchange_router._STAGE_STATE[node] = {
            "phase": "after_bedouin",
            "bedouin": True,
            "heretic": False,
        }
        context = _Context([object(), object()], [_Detail(), _Detail()])
        exchange_router.ExchangeProductRouter().run(
            context,
            _argv({"products": self.GATED_PRODUCTS, **self.GATES, "max_swipes": 1}, node),
        )
        self.assertEqual(context.routes, [(node, ["交易所交换失败"])])
        self.assertEqual(context.tasker.controller.swipes[0][1:4], (450, 640, 300))

    def test_heretic_gate_blocks_encrypted_products(self):
        node = "异教徒门控测试"
        exchange_router._PROCESSED[node] = {"海枣"}
        exchange_router._STAGE_STATE[node] = {
            "phase": "after_bedouin",
            "bedouin": True,
            "heretic": False,
        }
        context = _Context([object(), object()], [_Detail(), _Detail()])
        exchange_router.ExchangeProductRouter().run(
            context,
            _argv({"products": self.GATED_PRODUCTS, **self.GATES, "max_swipes": 0, "gated_max_swipes": 0}, node),
        )
        self.assertEqual(context.routes, [(node, ["交易所交换失败"])])
        self.assertEqual(context.recognition_calls, 1)

    def test_front_products_complete_uses_large_swipe_for_bedouin(self):
        node = "贝多因人大幅滑动测试"
        exchange_router._PROCESSED[node] = {"化学品"}
        context = _Context([object(), object(), object()], [_Detail(hit=True), _Detail(texts=("观看广告",)), _Detail(), _Detail()])
        exchange_router.ExchangeProductRouter().run(
            context,
            _argv({
                "products": self.GATED_PRODUCTS,
                **self.GATES,
                "max_swipes": 1,
                "large_max_swipes": 1,
                "large_start_y": 650,
                "large_end_y": 90,
                "large_duration": 500,
            }, node),
        )
        self.assertEqual(context.routes, [(node, ["交易所交换失败"])])
        self.assertEqual(context.tasker.controller.swipes[0], (640, 650, 640, 90, 500))
    def test_heretic_gate_is_checked_when_sea_date_is_not_found(self):
        node = "海枣未找到仍寻找异教徒"
        exchange_router._STAGE_STATE[node] = {
            "phase": "after_bedouin",
            "bedouin": True,
            "heretic": False,
        }
        context = _Context(
            [object(), object()],
            [
                _Detail(hit=True), _Detail(texts=("异教徒",)),
                _Detail(hit=True), _Detail(texts=("加密币", "木材"), box=(100, 100, 200, 80)),
            ],
        )
        exchange_router.ExchangeProductRouter().run(
            context,
            _argv({"products": self.GATED_PRODUCTS, **self.GATES, "max_swipes": 0}, node),
        )
        self.assertEqual(context.routes, [(node, ["交换木材"])])
        self.assertEqual(exchange_router._STAGE_STATE[node]["phase"], "after_heretic")
    def test_heretic_gate_allows_encrypted_product_after_matching(self):
        node = "异教徒通过测试"
        exchange_router._PROCESSED[node] = {"海枣"}
        exchange_router._STAGE_STATE[node] = {
            "phase": "after_bedouin",
            "bedouin": True,
            "heretic": False,
        }
        context = _Context(
            [object(), object()],
            [
                _Detail(hit=True), _Detail(texts=("异教徒",)),
                _Detail(hit=True), _Detail(texts=("加密币", "木材"), box=(100, 100, 200, 80)),
            ],
        )
        exchange_router.ExchangeProductRouter().run(
            context,
            _argv({"products": self.GATED_PRODUCTS, **self.GATES, "max_swipes": 0}, node),
        )
        self.assertEqual(context.routes, [(node, ["交换木材"])])
        self.assertEqual(exchange_router._STAGE_STATE[node]["phase"], "after_heretic")

    def test_multiple_candidates_ocr_each_box_and_requires_same_box_source(self):
        target = {"item": "木材", "template": "加密币交换木材.png", "text": "木材", "source_text": "加密币"}
        context = _Context([object()], [_Detail(hit=True, texts=("候选一", "候选二"), boxes=[(10, 10, 100, 40), (20, 20, 100, 40)]), _Detail(texts=("木材",)), _Detail(texts=("木材", "加密币"), box=(20, 20, 100, 40))])
        matched, texts = exchange_router._matched(context, object(), target)
        self.assertTrue(matched)
        self.assertEqual(texts, ["木材", "加密币"])
        self.assertEqual(context.recognition_calls, 3)
        self.assertEqual(context.recognition_args[1][1].roi, (10, 10, 100, 40))
        self.assertEqual(context.recognition_args[2][1].roi, (20, 20, 100, 40))

    def test_same_candidate_box_is_required_for_encrypted_product(self):
        target = {"item": "木材", "template": "加密币交换木材.png", "text": "木材", "source_text": "加密币"}
        context = _Context([object()], [_Detail(hit=True, texts=("第一行", "第二行"), box=(10, 10, 100, 40)), _Detail(texts=("木材",)), _Detail(texts=("加密币",))])
        matched, _ = exchange_router._matched(context, object(), target)
        self.assertFalse(matched)
        self.assertEqual(context.recognition_calls, 3)

    def test_pending_product_disappearing_twice_is_marked_processed(self):
        node = "商品消失确认测试"
        exchange_router._PENDING[node] = "木梁"
        context = _Context([object(), object()], [_Detail(), _Detail()])
        exchange_router.ExchangeProductRouter().run(
            context,
            _argv({"products": [self.PRODUCTS[0]], "max_swipes": 0, "disappearance_interval": 0, "success_node": "完成"}, node),
        )
        self.assertEqual(context.routes, [(node, ["完成"])])
        self.assertNotIn(node, exchange_router._PENDING)

    def test_pending_product_still_visible_is_not_marked_processed(self):
        node = "商品仍存在测试"
        exchange_router._PENDING[node] = "木梁"
        context = _Context([object()], [_Detail(hit=True), _Detail(hit=True), _Detail(texts=("木梁",))])
        exchange_router.ExchangeProductRouter().run(context, _argv({"products": self.PRODUCTS, "max_swipes": 0, "disappearance_interval": 0}, node))
        self.assertNotIn("木梁", exchange_router._PROCESSED[node])
        self.assertEqual(context.routes, [(node, ["交换木梁"])])


if __name__ == "__main__":
    unittest.main()