# LLM Portfolio Research

A Python research project exploring whether language-model asset selections improve portfolio diversification, and how simple and complex prompts differ. It combines historical price preparation, diversity metrics, constrained numerical optimization, and out-of-sample evaluation.

The repository contains the complete three-section research notebook and both supporting Python scripts. It was developed independently; the neutral filenames replace the original course and submission labels. This is an implementation inspired by the cited paper, with a prompt-comparison extension. It does not claim to reproduce the paper's published results.

## Research workflow

- A 40-asset universe spans stocks, REITs, cryptocurrencies, commodities, currencies, and bonds.
- Daily log returns from 2021 estimate portfolio parameters. The evaluation period is January 2022 through January 2023.
- Simpson diversity, Shannon diversity, and average pairwise correlation describe selected asset sets.
- Long-only weights lie between zero and one and sum to one. Annualization uses 252 observations for every asset class.
- The notebook compares simple and complex prompts with mean–variance utility and minimum-variance portfolios. The script compares random selections with predefined, cached, or live model selections using maximum-Sharpe tangency and minimum-variance portfolios.

The notebook and script are **different experiments**. Their first optimization objective differs intentionally: the notebook maximizes annualized mean minus one half of annualized variance; the script's tangency portfolio maximizes mean divided by volatility. Sharpe calculations assume a zero risk-free rate. Their table numbers follow the research workflow and do not imply identical tables or results.

## Files

| File | Purpose |
| --- | --- |
| `notebooks/portfolio_experiments.ipynb` | Complete price preparation, simple/complex prompt experiment, and optional connectivity example |
| `src/replicate_tables.py` | Random-versus-selection experiment, optimizers, response parsing, local cache, and printed summary tables |
| `src/extract_paper.py` | Optional text extraction from a separately supplied PDF |
| `tests/test_offline.py` | Synthetic-data checks for numerical helpers, validation, failure handling, and fixed mode |
| `CHANGELOG.md` | Changes made while preparing the original implementation for publication |

## Setup and offline checks

Use Python 3.12 or newer. From the repository root:

```text
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests generate deterministic synthetic prices in a temporary directory. They do not download data, use credentials, call model providers, or require the research paper. They verify implementation behavior rather than reproduce the empirical study.

## Supply price data

No prices, model caches, research PDFs, extracted paper text, or historical result files are bundled. Supply `data/assets_prices.csv`, or pass a different location with `--prices`.

The UTF-8 CSV must have a `Date` index column in ascending order, unique dates, one numeric column per ticker, and finite positive prices without missing values. Include all selected tickers and observations covering both the training and evaluation periods. Fixed mode requires `BTC-USD`, `GLD`, `GOVT`, and `JPM`. The complete asset universe is defined in the notebook and script.

The notebook retains the original `yfinance` preparation route using adjusted closing prices. Set its `DOWNLOAD_MARKET_DATA` flag to `True` only when you choose to download data. That route requests 2020-12-30 through 2023-02-01 (exclusive end), aligns complete observations across all assets, and writes local price and log-return CSVs. When the flag is `False`, it reads an existing CSV or reports that data is missing. Run Jupyter from the repository root, or set the notebook's `DATA_DIR` to `Path("../data")` when its working directory is `notebooks/`.

Check data-provider permissions separately before redistributing data. The repository ignores CSVs and the `data/` directory so the supplied data stays local.

## Run the script

The first positional argument is the number of random baseline draws; the second selects the mode. In `file` and `api` modes, that count also specifies model selections per portfolio size (2, 3, and 4 assets).

```text
python src/replicate_tables.py 200 fixed --prices data/assets_prices.csv
python src/replicate_tables.py 200 file --prices data/assets_prices.csv --cache data/llm_selections.jsonl
```

`fixed` is an offline diagnostic. It evaluates one predefined selection per portfolio size, rather than duplicating it into independent trials. Its t-test columns display `-`; those selections are not evidence of repeated LLM sampling. Random baseline sampling uses seed 42.

`file` requires at least the requested number of valid local selections for every portfolio size. Each cache line is JSON with `k`, `tickers`, and optionally `raw`, for example:

```json
{"k":2,"tickers":["BTC-USD","GLD"],"raw":"optional local response"}
```

Caches are not bundled or committed. A missing or invalid cache fails explicitly. Numeric table output includes annualized return, annualized volatility, Sharpe ratio, and Welch t-tests where independent samples are available. Repeated model responses and overlapping portfolios can limit the independence assumption; table significance should be interpreted as exploratory.

## Optional model API configuration

Live model requests require an OpenAI-compatible chat-completions endpoint. Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` in your shell; `.env.example` lists placeholders, and the code does not automatically load `.env` files. The base URL should end at the API root, such as `/v1`; the code appends `/chat/completions`.

```powershell
$env:OPENAI_API_KEY = "your-local-key"
$env:OPENAI_BASE_URL = "https://your-provider.example/v1"
$env:OPENAI_MODEL = "your-model-name"
python src/replicate_tables.py 20 api --prices data/assets_prices.csv --cache data/llm_selections.jsonl
```

API mode sends model requests and can incur charges. The notebook's `RUN_API_EXPERIMENT` and `RUN_API_SMOKE_CHECK` flags default to `False`; set them explicitly after configuring the provider and choosing the trial count. Its original experiment uses 1,000 selections per prompt and portfolio size, so reduce `N_TRIALS` for an initial manual check. Requests are bounded to five concurrent batches with a timeout. Transport, parsing, and validation failures stop the experiment; random portfolios are never substituted for failed model responses.

Never put a real key in a notebook cell, saved output, cache, or Git history. Clear notebook output before committing.

## Validation and limits

Publication preparation used offline helper tests and deterministic synthetic-price end-to-end checks. A separate local historical-price file was available for an additional fixed-mode check; it is not included. No live model experiment or new market-data download was performed, and no full empirical reproduction is claimed. See `VALIDATION.md` for the recorded checks.

The experiment uses a fixed historical asset universe, common complete observations, a single training/evaluation split, zero trading costs, and zero risk-free rate. It does not establish that a model can outperform future markets. A model queried after the historical evaluation period may have knowledge of that period; the current implementation does not establish a point-in-time information boundary. Changing provider, model, prompt, price adjustments, or response sampling can change results. Optimization failures raise errors instead of returning failed weights.

## Reference

Ko, H., & Lee, J. (2024). *Can ChatGPT improve investment decisions? From a portfolio management perspective*. **Finance Research Letters, 64**, 105433. [https://doi.org/10.1016/j.frl.2024.105433](https://doi.org/10.1016/j.frl.2024.105433).

The bibliographic details were checked against the locally available paper text. The paper and its text are not redistributed here. Third-party libraries retain their own licenses; no license to those materials is implied.
