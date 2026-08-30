import unittest

from reward_reco import (
    RewardCountdownAbsentRecognition,
    _best_countdown_result,
    _match_target,
)


class RewardRecognitionTests(unittest.TestCase):
    TARGET = "奖励已发放"

    def test_accepts_target_followed_only_by_close_symbol(self):
        for text in ("奖励已发放X", "奖励已发放 X", "奖励已发放 x", "奖励已发放 ×"):
            with self.subTest(text=text):
                self.assertIsNotNone(_match_target(text, self.TARGET, max_mismatches=1))

    def test_rejects_other_text_after_target(self):
        for text in ("奖励已发放关闭", "奖励已发放 X 关闭", "奖励已发放领取", "奖励已发放XX"):
            with self.subTest(text=text):
                self.assertIsNone(_match_target(text, self.TARGET, max_mismatches=1))

    def test_countdown_candidate_below_threshold_is_absent(self):
        class Result:
            def __init__(self, score):
                self.score = score
                self.box = (1, 2, 3, 4)

        class Detail:
            best_result = Result(0.538913)
            all_results = [best_result]
            filtered_results = []

        result, score = _best_countdown_result(Detail(), 0.7)
        self.assertIsNone(result)
        self.assertAlmostEqual(score, 0.538913)

    def test_countdown_candidate_at_threshold_is_present(self):
        class Result:
            score = 0.7
            box = (1, 2, 3, 4)

        class Detail:
            best_result = Result()
            all_results = [best_result]
            filtered_results = []

        result, score = _best_countdown_result(Detail(), 0.7)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(score, 0.7)

    def test_countdown_absent_recognition_accepts_low_score_candidate(self):
        import json
        from types import SimpleNamespace

        class Result:
            score = 0.538913
            box = (1, 2, 3, 4)

        class Detail:
            best_result = Result()
            all_results = [best_result]
            filtered_results = []

        class Context:
            def run_recognition_direct(self, *args):
                return Detail()

        argv = SimpleNamespace(
            custom_recognition_param=json.dumps({"threshold": 0.7}),
            image=None,
            node_name="测试倒计时不存在",
        )
        result = RewardCountdownAbsentRecognition().analyze(Context(), argv)
        self.assertIsNotNone(result)

    def test_countdown_absent_recognition_rejects_real_countdown(self):
        import json
        from types import SimpleNamespace

        class Result:
            score = 0.91
            box = (1, 2, 3, 4)

        class Detail:
            best_result = Result()
            all_results = [best_result]
            filtered_results = []

        class Context:
            def run_recognition_direct(self, *args):
                return Detail()

        argv = SimpleNamespace(
            custom_recognition_param=json.dumps({"threshold": 0.7}),
            image=None,
            node_name="测试倒计时不存在",
        )
        result = RewardCountdownAbsentRecognition().analyze(Context(), argv)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()