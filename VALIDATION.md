# Offline validation

Checked on 2026-10-22 using Python 3.12.14, NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, and requests 2.34.2.

```text
python -m unittest discover -s tests -v
Ran 10 tests
OK
```

The checks cover asset-class diversity, strict ticker parsing, singleton inference suppression, constrained optimization and its objectives, explicit optimization failure, rejection of invalid LLM responses without random replacement, cache completeness and malformed records, notebook response-batch validation, compilation of all three complete notebook code sections, and a fixed-mode end-to-end run on deterministic synthetic prices. The synthetic run uses six baseline draws per portfolio size and temporary local files.

An additional fixed-mode diagnostic completed with 12 random baseline draws per portfolio size using a locally supplied historical price CSV. It produced all three summary tables and omitted inferential tests for the single predefined selection. The historical CSV and its output are not included.

The repository was statically checked for saved notebook output, submission-specific labels, absolute home-directory paths, private-key headers, and recognizable API-key patterns. No matches were found in the publication files. This does not replace a final review of future additions.

No market-data download, provider connectivity check, or live model experiment was run. The data-download route, asynchronous provider integration, and PDF extraction were retained but were not exercised end to end. Notebook sections were compiled and pure numerical/validation functions were inspected through isolated offline tests; the full notebook was not executed. These checks are not an empirical reproduction or a claim about investment performance.
