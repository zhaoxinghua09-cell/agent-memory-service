"""limits.check_top_k 的单测（不依赖 fastapi，可直接 python -m unittest 跑）。"""
import unittest

from limits import CONTRACT_MAX_TOP_K, check_top_k


class TopKContract(unittest.TestCase):
    def test_accepts_limit(self):
        self.assertEqual(check_top_k(100), (None, 100))
        self.assertEqual(check_top_k(1), (None, 1))

    def test_rejects_over_limit(self):
        for bad in (101, 200, 500, 10_000):
            err, val = check_top_k(bad)
            self.assertIsNone(val)
            self.assertIn("exceeds contract maximum 100", err)

    def test_rejects_non_positive_and_non_int(self):
        for bad in (0, -3, None, "100", 100.0, True):
            err, val = check_top_k(bad)
            self.assertIsNone(val)
            self.assertEqual(err, "top_k required")

    def test_constant_is_contract_value(self):
        self.assertEqual(CONTRACT_MAX_TOP_K, 100)


if __name__ == "__main__":
    unittest.main()
