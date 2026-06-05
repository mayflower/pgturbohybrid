import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_pylate import compare_token_plan  # noqa: E402


class ComparePylateTokenPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = ROOT / "test" / "fixtures" / "token_plan_red_planet_doc.json"
        with fixture.open(encoding="utf-8") as file:
            self.golden = json.load(file)["plans"][0]

    def _debug_payload(self) -> dict:
        return {
            "vector_count": 8,
            "token_plan": {
                "tokens": [
                    {"id": token_id, "piece": piece, "retained": bool(retained)}
                    for token_id, piece, retained in zip(
                        self.golden["token_ids_after_padding_truncation"],
                        self.golden["token_pieces"],
                        self.golden["retain_mask"],
                    )
                ]
            },
        }

    def test_token_plan_match_passes(self) -> None:
        result = compare_token_plan(
            self._debug_payload(),
            self.golden,
            self.golden["input_text"],
        )

        self.assertTrue(result.token_plan_parity_passed)
        self.assertTrue(result.retain_parity_passed)
        self.assertTrue(result.vector_count_passed)
        self.assertIsNone(result.first_token_mismatch)
        self.assertIsNone(result.first_retain_mismatch)

    def test_first_mismatch_is_reported(self) -> None:
        payload = self._debug_payload()
        payload["token_plan"]["tokens"][7]["retained"] = True

        result = compare_token_plan(payload, self.golden, self.golden["input_text"])

        self.assertTrue(result.token_plan_parity_passed)
        self.assertFalse(result.retain_parity_passed)
        self.assertEqual(result.first_retain_mismatch["index"], 7)
        self.assertEqual(result.first_retain_mismatch["golden"], 0)
        self.assertEqual(result.first_retain_mismatch["pg"], 1)


if __name__ == "__main__":
    unittest.main()
