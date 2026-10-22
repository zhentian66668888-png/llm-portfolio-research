# Publication preparation

## 2026-10-22

- Retained the complete three original notebook sections and both Python scripts under neutral, topic-based paths. Added usage documentation, dependency declarations, ignored-file rules, and offline checks.
- Removed saved notebook outputs and submission-specific metadata. Standardized API configuration on environment variables; market downloads, the notebook experiment, and the connectivity example now require explicit flags.
- Replaced the notebook's silent random fallback with explicit response validation and propagated errors. Added timeouts and a concurrency limit; no failed or undersized model sample is replaced by a random draw.
- Corrected the notebook's `tangency` label to `mean_variance`: its original objective was mean minus one half variance. The script retains its separate true maximum-Sharpe tangency implementation. Both workflows now use six asset classes (including REITs) and log returns. The download end date includes the full January 2023 evaluation period.
- Use actual Welch-test p-values for notebook significance markers instead of fixed normal-distribution t-value thresholds. Retained the original threshold helper as historical code; it no longer supplies table markers.
- Fixed mode evaluates each predefined selection once and omits inferential tests for that singleton. Script tables distinguish predefined selections from LLM selections.
- Added strict ticker/cache/input validation, failure checks for optimizers, and explicit errors for missing or insufficient data. Moved script execution behind `main()` so numerical helpers can be imported without reading data or sending requests.
- Kept the PDF text-extraction functionality, with input/output command-line arguments instead of a hard-coded PDF path. No paper, market data, saved model response, local historical output, or credential was copied into the repository.

These corrections change some computations relative to the original submission. Historical output files are not presented as results of this corrected version. The full API experiment has not been rerun.
