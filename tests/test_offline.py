"""Offline checks using synthetic inputs; no provider or market-data requests."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("portfolio", ROOT / "src/replicate_tables.py")
portfolio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portfolio)


def notebook_helpers():
    """Load only pure function definitions; never execute notebook data/API cells."""
    nb = json.loads((ROOT / "notebooks/portfolio_experiments.ipynb").read_text(encoding="utf-8"))
    source = "".join(nb["cells"][2]["source"])
    tree = ast.parse(source)
    wanted = {"validate_portfolio_batch", "calculate_diversity_indices", "calculate_average_correlation", "solve_portfolio_v2", "significance_by_p"}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"np": np, "pd": pd, "minimize": minimize, "asset_categories": portfolio.asset_categories}
    exec(compile(tree, "notebook_helpers", "exec"), namespace)
    return namespace


class OfflineResearchChecks(unittest.TestCase):
    def test_diversity_distinguishes_asset_classes(self):
        simpson, shannon = portfolio.calculate_diversity_indices(["AAPL", "AMT", "GLD", "BTC-USD"])
        self.assertAlmostEqual(simpson, 0.75)
        self.assertAlmostEqual(shannon, np.log(4), places=8)
        self.assertEqual(portfolio.calculate_diversity_indices(["AAPL", "MSFT"])[0], 0)

    def test_strict_model_selection_parsing(self):
        pool = ["AAPL", "GLD", "BTC-USD"]
        self.assertEqual(portfolio.parse_tickers('{"tickers":["AAPL","GLD"]}', pool, 2), ["AAPL", "GLD"])
        for text in ['{"tickers":["AAPL"]}', '{"tickers":["AAPL","AAPL"]}', '{"tickers":["AAPL","GLD","BTC-USD"]}', '{"tickers":["AAPL","UNKNOWN"]}']:
            self.assertIsNone(portfolio.parse_tickers(text, pool, 2))

    def test_singleton_is_not_an_independent_sample(self):
        statistic, p_value = portfolio.safe_ttest_ind([0.2], [0.1, 0.3, 0.4])
        self.assertTrue(np.isnan(statistic))
        self.assertTrue(np.isnan(p_value))

    def test_optimizer_constraints_and_objectives(self):
        mean = np.array([0.0003, 0.0005, 0.00015])
        cov = np.array([[0.0001, 0.00002, 0.00001], [0.00002, 0.00025, 0.00001], [0.00001, 0.00001, 0.00005]])
        equal = np.ones(3) / 3
        for helper in [portfolio.get_tangency_portfolio, portfolio.get_min_risk_portfolio, portfolio.get_mean_variance_portfolio]:
            weights = helper(mean, cov)
            self.assertAlmostEqual(weights.sum(), 1, places=6)
            self.assertTrue((weights >= -1e-9).all())
            self.assertTrue((weights <= 1 + 1e-9).all())
        minimum = portfolio.get_min_risk_portfolio(mean, cov)
        self.assertLessEqual(minimum @ cov @ minimum, equal @ cov @ equal + 1e-9)
        tangency = portfolio.get_tangency_portfolio(mean, cov)
        sharpe = lambda weights: (mean @ weights) / np.sqrt(weights @ cov @ weights)
        self.assertGreaterEqual(sharpe(tangency), sharpe(equal) - 1e-7)

    def test_failed_optimization_is_reported(self):
        class FailedResult:
            success = False
            message = "synthetic failure"
        with patch.object(portfolio, "minimize", return_value=FailedResult()):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                portfolio.get_min_risk_portfolio(np.ones(2), np.eye(2))

    def test_invalid_response_never_becomes_a_random_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(portfolio, "openai_chat_completion", return_value='{"tickers":["UNKNOWN"]}'):
                with self.assertRaisesRegex(ValueError, "no random replacement"):
                    portfolio.get_chatgpt_selections(2, 2, ["AAPL", "GLD"], Path(tmp) / "absent.jsonl", "api", "fixture", "unused", "unused", 0)

    def test_cache_mode_requires_enough_valid_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.jsonl"
            cache.write_text('{"k":2,"tickers":["AAPL","GLD"]}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "2 required"):
                portfolio.get_chatgpt_selections(2, 2, ["AAPL", "GLD"], cache, "file", "", "", "", 0)
            cache.write_text("malformed JSON\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid cache record on line 1"):
                portfolio.load_cached_selections(cache)

    def test_notebook_batch_rejects_missing_unknown_and_duplicate_tickers(self):
        helpers = notebook_helpers()
        validate = helpers["validate_portfolio_batch"]
        self.assertEqual(validate([["aapl", "gld"]], 2, ["AAPL", "GLD"], 1), [["AAPL", "GLD"]])
        for data in [[], [["AAPL"]], [["AAPL", "AAPL"]], [["AAPL", "UNKNOWN"]], ["AAPL"]]:
            with self.assertRaises(ValueError):
                validate(data, 2, ["AAPL", "GLD"], 1)
        self.assertEqual(helpers["significance_by_p"](0.049), "**")

    def test_complete_notebook_compiles_without_running_cells(self):
        nb = json.loads((ROOT / "notebooks/portfolio_experiments.ipynb").read_text(encoding="utf-8"))
        code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 3)
        for index, cell in enumerate(code_cells):
            compile("".join(cell["source"]), f"notebook_cell_{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])

    def test_fixed_mode_runs_on_synthetic_prices_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(73)
            dates = pd.bdate_range("2020-12-30", "2023-01-31")
            tickers = ["AAPL", "JPM", "GLD", "GOVT", "BTC-USD", "ETH-USD"]
            changes = rng.normal(0.0003, 0.009, (len(dates), len(tickers)))
            prices = pd.DataFrame(100 * np.exp(np.cumsum(changes, axis=0)), index=dates, columns=tickers)
            path = Path(tmp) / "prices.csv"
            prices.to_csv(path, index_label="Date")
            output = io.StringIO()
            with patch.object(portfolio, "openai_chat_completion", side_effect=AssertionError("Network entry point must not be called")):
                with contextlib.redirect_stdout(output):
                    portfolio.main(["6", "fixed", "--prices", str(path)])
            text = output.getvalue()
            self.assertIn("not an independent LLM sample", text)
            self.assertIn("Fixed selection", text)
            self.assertIn("Table 5", text)
            self.assertNotIn("***", text)


if __name__ == "__main__":
    unittest.main()
