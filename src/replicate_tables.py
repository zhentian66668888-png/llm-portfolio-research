"""Offline fixed-selection diagnostics and optional LLM portfolio experiments."""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t, ttest_ind
import os
import json
import time
import re
import requests
import warnings
import sys

warnings.filterwarnings("ignore", message="Precision loss occurred*")

# --- Asset Category Mapping ---
asset_categories = {
    'AAPL': 'Stock', 'ADBE': 'Stock', 'BAC': 'Stock', 'CRM': 'Stock', 'JNJ': 'Stock', 
    'JPM': 'Stock', 'LLY': 'Stock', 'MA': 'Stock', 'MS': 'Stock', 'MSFT': 'Stock', 
    'NVDA': 'Stock', 'UNH': 'Stock', 'UNP': 'Stock', 'UPS': 'Stock', 'V': 'Stock',
    'ADA-USD': 'Crypto', 'BNB-USD': 'Crypto', 'BTC-USD': 'Crypto', 'ETH-USD': 'Crypto', 'XRP-USD': 'Crypto',
    'CPER': 'Commodity', 'GLD': 'Commodity', 'PALL': 'Commodity', 'PPLT': 'Commodity', 'SLV': 'Commodity',
    'CNYUSD=X': 'Forex', 'EURUSD=X': 'Forex', 'GBPUSD=X': 'Forex', 'JPYUSD=X': 'Forex', 'UUP': 'Forex',
    'AMT': 'REIT', 'CCI': 'REIT', 'EQIX': 'REIT', 'PLD': 'REIT', 'PSA': 'REIT',
    'CGB.MI': 'Bond', 'GOVT': 'Bond', 'IGLT.L': 'Bond', 'VETA.L': 'Bond', 'XJSE.MI': 'Bond'
}

def calculate_diversity_indices(assets):
    cats = [asset_categories[a] for a in assets]
    counts = pd.Series(cats).value_counts()
    p = counts / len(assets)
    simpson = 1 - (p**2).sum()
    shannon = -(p * np.log(p + 1e-10)).sum()
    return simpson, shannon

def get_performance_from_returns(portfolio_returns, annualization=252):
    exp_return = portfolio_returns.mean() * annualization
    risk = portfolio_returns.std() * np.sqrt(annualization)
    sr = exp_return / risk if risk != 0 else 0
    return exp_return, risk, sr

def get_portfolio_performance(returns, weights, annualization=252):
    portfolio_returns = returns.dot(weights).dropna()
    exp_return, risk, sr = get_performance_from_returns(portfolio_returns, annualization)
    return exp_return, risk, sr, portfolio_returns

def get_tangency_portfolio(mean_returns, cov_matrix, annualization=252):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, annualization)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, annualization):
        p_ret = np.sum(mean_returns * weights) * annualization
        p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * annualization, weights)))
        return -p_ret / p_std if p_std != 0 else 0

    result = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    return result.x

def get_mean_variance_portfolio(mean_returns, cov_matrix, risk_aversion=1.0, annualization=252):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_aversion, annualization)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))

    def neg_utility(weights, mean_returns, cov_matrix, risk_aversion, annualization):
        p_ret = np.sum(mean_returns * weights) * annualization
        p_var = np.dot(weights.T, np.dot(cov_matrix * annualization, weights))
        utility = p_ret - 0.5 * risk_aversion * p_var
        return -utility

    result = minimize(neg_utility, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    return result.x

def get_min_risk_portfolio(mean_returns, cov_matrix, annualization=252):
    num_assets = len(mean_returns)
    args = (cov_matrix, annualization)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))

    def portfolio_variance(weights, cov_matrix, annualization):
        return np.dot(weights.T, np.dot(cov_matrix * annualization, weights))

    result = minimize(portfolio_variance, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    return result.x

def average_pairwise_correlation(returns_df):
    k = returns_df.shape[1]
    if k < 2:
        return np.nan
    corr = returns_df.corr().values
    return corr[np.triu_indices(k, k=1)].mean()

def get_significance(p_val):
    if p_val < 0.01: return "***"
    if p_val < 0.05: return "**"
    if p_val < 0.1: return "*"
    return ""

def openai_chat_completion(messages, model, api_key, base_url):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def parse_tickers(text, asset_pool, k):
    try:
        obj = json.loads(text)
        tickers = obj.get("tickers", [])
        tickers = [str(x).strip() for x in tickers]
    except:
        candidates = re.findall(r"[A-Z0-9][A-Z0-9.=\\-]{1,15}", text.upper())
        tickers = []
        seen = set()
        for c in candidates:
            if c in asset_pool and c not in seen:
                seen.add(c)
                tickers.append(c)
    tickers = [t for t in tickers if t in asset_pool]
    uniq = []
    seen = set()
    for tkr in tickers:
        if tkr not in seen:
            seen.add(tkr)
            uniq.append(tkr)
    if len(uniq) != k or len(tickers) != k:
        return None
    return uniq

def load_cached_selections(path):
    by_k = {}
    if not os.path.exists(path):
        return by_k
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = int(obj["k"])
                tickers = list(obj["tickers"])
                by_k.setdefault(k, []).append(tickers)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid cache record on line {line_number}") from exc
    return by_k

def append_cached_selection(path, k, tickers, raw):
    rec = {"k": int(k), "tickers": tickers, "raw": raw}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def safe_ttest_ind(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan
    if np.allclose(a, a[0]) and np.allclose(b, b[0]):
        return np.nan, np.nan
    return ttest_ind(a, b, equal_var=False)

def get_chatgpt_selections(k, n, asset_pool, cache_path, mode, model, api_key, base_url, sleep_s):
    cached = load_cached_selections(cache_path).get(k, [])
    for tickers in cached:
        if parse_tickers(json.dumps({"tickers": tickers}), asset_pool, k) is None:
            raise ValueError(f"Invalid cached selection for k={k}")
    if mode == "file":
        if len(cached) < n:
            raise ValueError(f"Cache has {len(cached)} selections for k={k}; {n} required")
        return cached[:n]
    if mode != "api":
        raise ValueError("Selections must use file or api mode")
    selections = list(cached)
    pool_str = ", ".join(asset_pool)
    system_msg = "You are a finance research assistant. Respond with JSON only."
    user_template = (
        "Select exactly {k} assets from the following tickers to maximize diversification across asset classes "
        "for a portfolio. Output strictly as JSON: {{\"tickers\":[...]}} with tickers only. "
        "Tickers: {pool}"
    )
    while len(selections) < n:
        raw = openai_chat_completion(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": user_template.format(k=k, pool=pool_str)}],
            model=model, api_key=api_key, base_url=base_url,
        )
        tickers = parse_tickers(raw, asset_pool, k)
        if tickers is None:
            raise ValueError(f"Invalid LLM selection for k={k}; no random replacement was added")
        selections.append(tickers)
        append_cached_selection(cache_path, k, tickers, raw)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return selections[:n]

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", nargs="?", type=int, default=200, help="Random baseline draws and, in file/api mode, LLM draws per portfolio size")
    parser.add_argument("mode", nargs="?", choices=["fixed", "file", "api"], default="fixed")
    parser.add_argument("--prices", type=Path, default=Path("data/assets_prices.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/llm_selections.jsonl"))
    args = parser.parse_args(argv)
    if args.trials < 2:
        parser.error("trials must be at least 2")
    # Load data
    df = pd.read_csv(args.prices, index_col='Date', parse_dates=True)
    if df.empty or df.isna().any().any() or not np.isfinite(df.to_numpy(dtype=float)).all() or (df <= 0).any().any():
        raise ValueError("Prices must be a nonempty numeric table of finite positive values without missing data")
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and in ascending order")
    unknown = set(df.columns) - set(asset_categories)
    if unknown:
        raise ValueError(f"Unclassified assets: {sorted(unknown)}")
    returns = np.log(df / df.shift(1)).dropna()

    # Periods
    in_sample = returns.loc['2021-01-01':'2021-12-31']
    out_sample = returns.loc['2022-01-01':'2023-01-31']
    if len(in_sample) < 2 or len(out_sample) < 2:
        raise ValueError('Need at least two return observations in both 2021 and 2022-01 through 2023-01')

    asset_pool = list(returns.columns)
    chatgpt_assets = {
        2: ['BTC-USD', 'GLD'],
        3: ['BTC-USD', 'GLD', 'GOVT'],
        4: ['JPM', 'GLD', 'GOVT', 'BTC-USD']
    }

    results_table3 = []
    results_table4 = []
    results_table5 = []

    np.random.seed(42) # For reproducibility

    n_trials = args.trials
    chatgpt_mode = args.mode
    cache_path = args.cache
    chatgpt_model = os.getenv("OPENAI_MODEL", "")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    sleep_s = float(os.getenv("OPENAI_SLEEP_S", "0"))
    selection_label = "Fixed selection" if chatgpt_mode == "fixed" else "LLM selection"
    print(f"Mode: {chatgpt_mode}; random baseline draws per k: {n_trials}")
    if chatgpt_mode == "fixed":
        print("One predefined selection per k; t-tests are omitted because it is not an independent LLM sample.")

    weights_cache = {
        'tangency': {},
        'minrisk': {},
    }

    def selection_metrics(tickers):
        tickers = sorted(list(tickers))
        in_ret = in_sample[tickers]
        out_ret = out_sample[tickers]
        simpson, shannon = calculate_diversity_indices(tickers)
        corr = average_pairwise_correlation(in_ret)
        mean = in_ret.mean()
        cov = in_ret.cov()
        key = tuple(tickers)
        if key not in weights_cache['tangency']:
            weights_cache['tangency'][key] = get_tangency_portfolio(mean, cov)
        w_t = weights_cache['tangency'][key]
        t_exp, t_risk, t_sr, _ = get_portfolio_performance(out_ret, w_t)
        if key not in weights_cache['minrisk']:
            weights_cache['minrisk'][key] = get_min_risk_portfolio(mean, cov)
        w_m = weights_cache['minrisk'][key]
        m_exp, m_risk, m_sr, _ = get_portfolio_performance(out_ret, w_m)
        return simpson, shannon, corr, (t_exp, t_risk, t_sr), (m_exp, m_risk, m_sr)

    for k in [2, 3, 4]:
        if chatgpt_mode == "fixed":
            cg_selections = [chatgpt_assets[k]]
        else:
            if chatgpt_mode == "api" and not all([openai_api_key, openai_base_url, chatgpt_model]):
                raise RuntimeError("OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL are required in api mode")
            cg_selections = get_chatgpt_selections(
                k=k,
                n=n_trials,
                asset_pool=asset_pool,
                cache_path=cache_path,
                mode=chatgpt_mode,
                model=chatgpt_model,
                api_key=openai_api_key,
                base_url=openai_base_url,
                sleep_s=sleep_s,
            )

        print(f"Running {n_trials} trials for k={k}...")
        sys.stdout.flush()

        rand_simpson = []
        rand_shannon = []
        rand_corr = []
        rand_t_exp = []
        rand_t_risk = []
        rand_t_sr = []
        rand_m_exp = []
        rand_m_risk = []
        rand_m_sr = []

        cg_simpson = []
        cg_shannon = []
        cg_corr = []
        cg_t_exp = []
        cg_t_risk = []
        cg_t_sr = []
        cg_m_exp = []
        cg_m_risk = []
        cg_m_sr = []

        for tickers in cg_selections:
            try:
                s, h, c, (te, tr, ts), (me, mr, msr) = selection_metrics(tickers)
                cg_simpson.append(s)
                cg_shannon.append(h)
                cg_corr.append(c)
                cg_t_exp.append(te)
                cg_t_risk.append(tr)
                cg_t_sr.append(ts)
                cg_m_exp.append(me)
                cg_m_risk.append(mr)
                cg_m_sr.append(msr)
            except Exception as exc:
                raise RuntimeError(f"Cannot evaluate selection {list(tickers)}") from exc

        for _ in range(n_trials):
            tickers = np.random.choice(asset_pool, k, replace=False)
            try:
                s, h, c, (te, tr, ts), (me, mr, msr) = selection_metrics(tickers)
                rand_simpson.append(s)
                rand_shannon.append(h)
                rand_corr.append(c)
                rand_t_exp.append(te)
                rand_t_risk.append(tr)
                rand_t_sr.append(ts)
                rand_m_exp.append(me)
                rand_m_risk.append(mr)
                rand_m_sr.append(msr)
            except Exception as exc:
                raise RuntimeError(f"Cannot evaluate selection {list(tickers)}") from exc

        r_simpson_mean = float(np.mean(rand_simpson))
        r_shannon_mean = float(np.mean(rand_shannon))
        r_corr_mean = float(np.mean(rand_corr))
        c_simpson_mean = float(np.mean(cg_simpson))
        c_shannon_mean = float(np.mean(cg_shannon))
        c_corr_mean = float(np.mean(cg_corr))

        t_simpson, p_simpson = safe_ttest_ind(cg_simpson, rand_simpson)
        t_shannon, p_shannon = safe_ttest_ind(cg_shannon, rand_shannon)
        t_corr, p_corr = safe_ttest_ind(cg_corr, rand_corr)

        results_table3.append({
            'k': k,
            'Random_Simpson': r_simpson_mean, 'ChatGPT_Simpson': c_simpson_mean, 't_Simpson': t_simpson, 'p_Simpson': p_simpson,
            'Random_Shannon': r_shannon_mean, 'ChatGPT_Shannon': c_shannon_mean, 't_Shannon': t_shannon, 'p_Shannon': p_shannon,
            'Random_Corr': r_corr_mean, 'ChatGPT_Corr': c_corr_mean, 't_Corr': t_corr, 'p_Corr': p_corr
        })

        r_exp_t = float(np.mean(rand_t_exp))
        r_risk_t = float(np.mean(rand_t_risk))
        r_sr_t = float(np.mean(rand_t_sr))
        c_exp_t = float(np.mean(cg_t_exp))
        c_risk_t = float(np.mean(cg_t_risk))
        c_sr_t = float(np.mean(cg_t_sr))

        t_risk_t, p_risk_t = safe_ttest_ind(cg_t_risk, rand_t_risk)
        t_sr_t, p_sr_t = safe_ttest_ind(cg_t_sr, rand_t_sr)

        results_table4.append({
            'k': k,
            'Random_Exp': r_exp_t, 'Random_Risk': r_risk_t, 'Random_SR': r_sr_t,
            'ChatGPT_Exp': c_exp_t, 'ChatGPT_Risk': c_risk_t, 'ChatGPT_SR': c_sr_t,
            't_Risk': t_risk_t, 'p_Risk': p_risk_t,
            't_SR': t_sr_t, 'p_SR': p_sr_t
        })

        r_exp_m = float(np.mean(rand_m_exp))
        r_risk_m = float(np.mean(rand_m_risk))
        r_sr_m = float(np.mean(rand_m_sr))
        c_exp_m = float(np.mean(cg_m_exp))
        c_risk_m = float(np.mean(cg_m_risk))
        c_sr_m = float(np.mean(cg_m_sr))

        t_risk_m, p_risk_m = safe_ttest_ind(cg_m_risk, rand_m_risk)
        t_sr_m, p_sr_m = safe_ttest_ind(cg_m_sr, rand_m_sr)

        results_table5.append({
            'k': k,
            'Random_Exp': r_exp_m, 'Random_Risk': r_risk_m, 'Random_SR': r_sr_m,
            'ChatGPT_Exp': c_exp_m, 'ChatGPT_Risk': c_risk_m, 'ChatGPT_SR': c_sr_m,
            't_Risk': t_risk_m, 'p_Risk': p_risk_m,
            't_SR': t_sr_m, 'p_SR': p_sr_m
        })

    # --- Print Results ---

    def fmt_t(value, p_value, width=12):
        if not np.isfinite(value) or not np.isfinite(p_value):
            return "-"
        stars = get_significance(p_value)
        return f"{value:>{width - len(stars)}.3f}{stars}"

    def print_table3(table3_rows):
        print("\nTable 3")
        print("The results of three diversity indices.")
        header = f"{'':<22}{'Random':>10}{selection_label:>16}{'t-value':>12}"
        print(header)
        print("-" * len(header))

        print("Panel A: The Simpson diversity index")
        for res in table3_rows:
            tval = fmt_t(res['t_Simpson'], res['p_Simpson'])
            label = f"k = {res['k']}"
            print(f"{label:<22}{res['Random_Simpson']:>10.3f}{res['ChatGPT_Simpson']:>16.3f}{tval:>12}")
        print()

        print("Panel B: The Shannon diversity index")
        for res in table3_rows:
            tval = fmt_t(res['t_Shannon'], res['p_Shannon'])
            label = f"k = {res['k']}"
            print(f"{label:<22}{res['Random_Shannon']:>10.3f}{res['ChatGPT_Shannon']:>16.3f}{tval:>12}")
        print()

        print("Panel C: Average correlation")
        for res in table3_rows:
            tval = fmt_t(res['t_Corr'], res['p_Corr'])
            label = f"k = {res['k']}"
            print(f"{label:<22}{res['Random_Corr']:>10.3f}{res['ChatGPT_Corr']:>16.3f}{tval:>12}")

    def print_table45(title, subtitle, rows):
        print(f"\n{title}")
        print(subtitle)
        top = f"{'':<10}{'Expected return':>14}{'Risk':>12}{'':>12}{'Risk-adjusted return':>22}"
        sub = f"{'':<10}{'':>14}{'Stdev.':>12}{'t-value':>12}{'SR':>12}{'t-value':>12}"
        print(top)
        print(sub)
        print("-" * len(sub))

        row_fmt = "{name:<16}{exp:>14.4f}{risk:>12.4f}{t_risk:>12}{sr:>12.3f}{t_sr:>12}"
        for res in rows:
            k = res['k']
            panel = chr(64 + k - 1)
            print(f"Panel {panel}: k = {k}")
            print(row_fmt.format(
                name="Random",
                exp=res['Random_Exp'],
                risk=res['Random_Risk'],
                t_risk='-',
                sr=res['Random_SR'],
                t_sr='-',
            ))
            print(row_fmt.format(
                name=selection_label,
                exp=res['ChatGPT_Exp'],
                risk=res['ChatGPT_Risk'],
                t_risk=fmt_t(res['t_Risk'], res['p_Risk']),
                sr=res['ChatGPT_SR'],
                t_sr=fmt_t(res['t_SR'], res['p_SR']),
            ))
            print()

    print_table3(results_table3)
    print_table45(
        "Table 4",
        "The out-of-sample results of the tangency portfolio.",
        results_table4,
    )
    print_table45(
        "Table 5",
        "The out-of-sample results of the minimum risk portfolio.",
        results_table5,
    )


if __name__ == "__main__":
    main()
