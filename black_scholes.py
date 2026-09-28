"""Black-Scholes European derivative pricer via mutual numerical integral basis."""

import sys
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from scipy.stats import lognorm, norm
from dataclasses import dataclass
import time

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TOL = 1e-12

def call_option(
    K: float, thres: Union[float, Sequence[float], np.ndarray] = 0.0,
    vest_pct: Union[float, Sequence[float], np.ndarray] = 1.0, is_step: int = 1,
) -> Callable[[np.ndarray], np.ndarray]:
    """Unified Call Option Payoff with step (is_step=1) or linear interpolation (is_step=0) vesting."""
    t_arr, v_arr = np.atleast_1d(np.asarray(thres, dtype=float)), np.atleast_1d(np.asarray(vest_pct, dtype=float))
    if len(t_arr) != len(v_arr) or np.any(v_arr < 0.0):
        raise ValueError("thres and vest_pct must have identical lengths, and vest_pct >= 0.")
    v_padded = np.concatenate(([0.0], v_arr))

    def vesting_fn(ST: np.ndarray) -> np.ndarray:
        ST = np.asarray(ST, dtype=float)
        if is_step:
            return v_padded[np.searchsorted(t_arr, ST, side="right")]
        return np.interp(ST, t_arr, v_arr, left=0.0, right=v_arr[-1])

    def payoff(ST: np.ndarray) -> np.ndarray:
        return np.maximum(ST - K, 0.0) * vesting_fn(ST)

    payoff.vesting = vesting_fn
    return payoff


def make_mutual_grid(
    S0: float, T: float, r: float, q: float, sigma: float,
    N: int = 10000, st_min: float = 0.0, st_max: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Constructs shared asset price grid and lognormal PDF under BSM."""
    if T <= 0.0 or sigma <= 0.0:
        return np.array([S0]), np.array([1.0])
    mu = np.log(S0) + (r - q - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    st_max = float(S0 * np.exp((r - q) * T + 6.0 * s)) if st_max is None else st_max
    st_grid = np.linspace(st_min, st_max, N + 1)
    return st_grid, lognorm.pdf(st_grid, s=s, scale=np.exp(mu))


def make_delta_weight(
    S0: float, T: float, r: float, q: float, sigma: float,
    st_grid: np.ndarray, pdf: np.ndarray,
) -> np.ndarray:
    """Constructs LRM Delta kernel weight w_delta(ST) = pdf(ST) * z / (S0 * sigma^2 * T)."""
    if T <= 0.0 or sigma <= 0.0:
        return np.zeros_like(st_grid)
    z = np.log(np.maximum(st_grid, TOL) / S0) - (r - q - 0.5 * sigma**2) * T
    w = pdf * (z / (S0 * sigma**2 * T))
    if st_grid[0] <= 0.0:
        w[0] = 0.0
    return w


def bs_price_and_delta_multiple(
    S0: float, T: float, r: float, q: float, sigma: float,
    payoffs: Union[Dict[str, Callable], Sequence[Callable]],
    N: int = 10000, st_min: float = 0.0, st_max: float = None,
) -> Tuple[Any, Any]:
    """Simultaneously calculates prices and deltas for all securities in portfolio on a single mutual grid."""
    is_dict = isinstance(payoffs, dict)
    items = list(payoffs.items()) if is_dict else list(enumerate(payoffs))
    if T <= 0.0:
        eps = 1e-4
        p = {k: float(fn(np.array([S0]))[0]) for k, fn in items}
        d = {k: float((fn(np.array([S0 + eps]))[0] - fn(np.array([S0 - eps]))[0]) / (2.0 * eps)) for k, fn in items}
        return (p if is_dict else list(p.values())), (d if is_dict else list(d.values()))

    grid, pdf = make_mutual_grid(S0, T, r, q, sigma, N=N, st_min=st_min, st_max=st_max)
    w_delta = make_delta_weight(S0, T, r, q, sigma, grid, pdf)
    disc = np.exp(-r * T)
    p, d = {}, {}
    for k, fn in items:
        f = fn(grid)
        p[k], d[k] = float(disc * trapezoid(f * pdf, grid)), float(disc * trapezoid(f * w_delta, grid))

    return (p, d) if is_dict else (list(p.values()), list(d.values()))


def _single_threshold_cf(S0: float, K: float, thres: float, T: float, r: float, q: float, sigma: float) -> float:
    eff_k = max(thres, K)
    if eff_k <= 0.0:
        return float(S0 * np.exp(-q * T) - K * np.exp(-r * T))
    d1 = (np.log(S0 / eff_k) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T)))


def call_closed_form(
    S0: float, K: float, thres: Union[float, Sequence[float], np.ndarray] = 0.0,
    vest_pct: Union[float, Sequence[float], np.ndarray] = 1.0,
    T: float = 1.0, r: float = 0.05, q: float = 0.0, sigma: float = 0.2,
) -> float:
    """Closed-form benchmark for call options (uses small T proxy if T<=0)."""
    T_eff = 1e-8 if T <= 0.0 else T
    t_arr, v_arr = np.atleast_1d(np.asarray(thres, dtype=float)), np.atleast_1d(np.asarray(vest_pct, dtype=float))
    return float(sum(dv * _single_threshold_cf(S0, K, tj, T_eff, r, q, sigma) for tj, dv in zip(t_arr, np.diff(np.concatenate(([0.0], v_arr))))))


def call_closed_form_delta(
    S0: float, K: float, thres: Union[float, Sequence[float], np.ndarray] = 0.0,
    vest_pct: Union[float, Sequence[float], np.ndarray] = 1.0,
    T: float = 1.0, r: float = 0.05, q: float = 0.0, sigma: float = 0.2,
    eps: float = 1e-5,
) -> float:
    """Closed-form benchmark Delta via high-precision symmetric difference on call_closed_form."""
    p_up = call_closed_form(S0 + eps, K, thres, vest_pct, T, r, q, sigma)
    p_dn = call_closed_form(S0 - eps, K, thres, vest_pct, T, r, q, sigma)
    return float((p_up - p_dn) / (2.0 * eps))


def _unpack_portfolio(payoffs: Union[Sequence, Dict], shares: Union[Sequence, Dict]) -> Tuple[list, np.ndarray]:
    fn_list = list(payoffs.values()) if isinstance(payoffs, dict) else list(payoffs)
    sh_list = [shares[k] for k in payoffs.keys()] if isinstance(payoffs, dict) and isinstance(shares, dict) else list(shares)
    return fn_list, np.asarray(sh_list, dtype=float)


def equity_moments(
    S0: float, T: float, r: float, q: float, sigma: float,
    payoffs: Union[Sequence[Callable], Dict[str, Callable]],
    shares: Union[Sequence[float], Dict[str, float]],
    N: int = 10000, cm_idx: int = 0,
) -> Dict[str, Any]:
    """Calculates class fair values, total equity value (TEV0), and aggregate equity volatility (voleq)."""
    fn_list, sh_arr = _unpack_portfolio(payoffs, shares)
    if T <= 0.0 or sigma <= 0.0:
        fvs = np.array([float(fn(np.array([S0]))[0]) for fn in fn_list])
        tev0 = float(np.sum(fvs * sh_arr))
        return {
            "s0": float(fvs[cm_idx]), "tev0": tev0, "voleq": 0.0, "fair_values": fvs,
            "check_tev_ratio": 1.0, "skew_log": 0.0, "kurt_log": 0.0,
        }
    grid, pdf = make_mutual_grid(S0, T, r, q, sigma, N=N)
    f_vals = [fn(grid) for fn in fn_list]
    fvs = np.array([float(np.exp(-r * T) * trapezoid(fv * pdf, grid)) for fv in f_vals])
    tev0, tev_t = float(np.sum(fvs * sh_arr)), sum(s * fv for fv, s in zip(f_vals, sh_arr))
    
    # QC checks: integrated discounted terminal TEV vs sum of class fair values
    check_tev0 = float(np.exp(-r * T) * float(trapezoid(np.maximum(tev_t, 0.0) * pdf, grid)))
    check_tev = float(check_tev0 / max(tev0, TOL))

    log_r = np.log(np.maximum(tev_t / max(tev0, TOL), 1e-30))
    mean_log = float(trapezoid(log_r * pdf, grid))
    var_log = float(trapezoid((log_r - mean_log)**2 * pdf, grid))
    
    if var_log > TOL:
        skew_log = float(trapezoid((log_r - mean_log)**3 * pdf, grid) / var_log**1.5)
        kurt_log = float(trapezoid((log_r - mean_log)**4 * pdf, grid) / var_log**2 - 3.0)
    else:
        skew_log = 0.0
        kurt_log = 0.0

    voleq = float(np.sqrt(max(var_log, 0.0)) / np.sqrt(T))
    return {
        "s0": float(fvs[cm_idx]), "tev0": tev0, "voleq": voleq, "fair_values": fvs,
        "check_tev_ratio": check_tev, "skew_log": skew_log, "kurt_log": kurt_log,
    }


def calibrate_cca(
    payoffs: Union[Sequence[Callable], Dict[str, Callable]],
    shares: Union[Sequence[float], Dict[str, float]],
    s0_target: float, voleq_target: float, tev0_target: float,
    r: float, T: float, q: float = 0.0, cm_idx: int = 0,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None,
    N: int = 5000, method: str = "Nelder-Mead",
    s0_guess: float = None,
) -> Dict[str, Any]:
    """Calibrates underlying S0 and common asset volatility (volcm) to match target equity volatility and value."""
    fn_list, sh_arr = _unpack_portfolio(payoffs, shares)
    
    if s0_guess is None:
        s0_guess = tev0_target / max(np.sum(sh_arr), TOL)
        
    cm_ratio = sh_arr[cm_idx] / max(np.sum(sh_arr), TOL)
    sigma_guess = voleq_target * np.sqrt(cm_ratio)
        
    x0 = [s0_target if s0_target > 0 else s0_guess, sigma_guess]
    
    if bounds is None:
        bounds = ((0.01, max(10.0 * x0[0], 0.02)), (0.05, 1.50))

    def obj(x: Sequence[float]) -> float:
        m = equity_moments(x[0], T, r, q, x[1], fn_list, sh_arr, N=N, cm_idx=cm_idx)
        err_s0 = ((m["s0"] - s0_target) / max(s0_target, TOL))**2 if s0_target > 0 else 0.0
        err_tev = ((m["tev0"] - tev0_target) / max(tev0_target, TOL))**2 if s0_target <= 0 else 0.0
        err_vol = ((m["voleq"] - voleq_target) / max(voleq_target, TOL))**2
        return float(err_s0 + err_tev + err_vol)

    res = minimize(obj, x0, method=method, bounds=bounds, tol=1e-4)
    if not res.success and method != "Nelder-Mead":
        res = minimize(obj, x0, method="Nelder-Mead", bounds=bounds, tol=1e-4)
    opt = equity_moments(res.x[0], T, r, q, res.x[1], fn_list, sh_arr, N=N, cm_idx=cm_idx)
    opt.update({
        "volcm": float(res.x[1]),
        "asset_s0": float(res.x[0]),
        "iterations": int(res.nit),
        "check_ratios": [opt["check_tev_ratio"], opt["voleq"] / max(voleq_target, TOL)],
        "status": bool(res.success),
        "message": str(res.message),
    })
    return opt


def metric_to_portfolio(metric: np.ndarray) -> Tuple[list, list]:
    """Converts sample.py 7-column metric array into (payoffs, shares)."""
    pfs, shs = [], []
    for r in metric:
        shs.append(float(r[0]))
        if bool(r[4]) or r[5] == 0: pfs.append(call_option(K=r[1]))
        elif r[5] == 1: pfs.append(call_option(K=r[1], thres=r[2], vest_pct=1.0, is_step=1))
        else: pfs.append(call_option(K=r[1], thres=[r[2], r[3]], vest_pct=[0.0, 1.0], is_step=0))
    return pfs, shs


@dataclass
class MarketParams:
    """Market inputs & grid settings for Capital Structure OPM valuation."""
    S0: float = 71.0778
    T: float = 1.0
    r: float = 0.05
    q: float = 0.0
    sigma: float = 0.4608
    N: int = 20000


@dataclass
class CalibrationTargets:
    """Target aggregate equity market values for CCA calibration."""
    tev0: float = 1000.0
    voleq: float = 0.55
    s0: float = 0.0


def calc_portfolio_allocations(
    prices: Dict[str, float],
    deltas: Dict[str, float],
    shares: Dict[str, float],
    eq_in: float
) -> Dict[str, Dict[str, float]]:
    """Calculates relative capital structure allocations (Omega, Elasticity, Specific Volatility)."""
    tot_val = sum(prices[k] * shares[k] for k in prices)
    tot_delta = sum(deltas[k] * shares[k] for k in deltas)
    
    allocs = {}
    for k in prices:
        num, d, sh = prices[k], deltas[k], shares[k]
        omega = (d * sh) / tot_delta if tot_delta > TOL else 0.0
        elas = (d * tot_val) / (num * tot_delta) if (num > 1e-8 and tot_delta > 1e-8) else 0.0
        v_num = eq_in * elas
        allocs[k] = {"omega": omega, "elas": elas, "specific_vol": v_num}
    
    return allocs


def run_opm_analysis(portfolio: dict, params: MarketParams, targets: CalibrationTargets = None) -> str:
    """Runs OPM and returns a formatted report string."""
    is_cal = targets is not None
    payoffs = {k: v["fn"] for k, v in portfolio.items()}
    shares = {k: v["shares"] for k, v in portfolio.items()}
    tot_shares = sum(shares.values())

    if is_cal:
        t_cal = time.perf_counter()
        
        # Calculate gross-up initial guess (TEV0 + Proceeds) / Total Shares
        tot_proceeds = sum(sh * portfolio[k]["params"].get("K", 0.0) for k, sh in shares.items())
        guess = (targets.tev0 + tot_proceeds) / max(tot_shares, TOL)
        
        user_input_s0 = params.S0
        cal = calibrate_cca(payoffs, shares, s0_target=targets.s0, voleq_target=targets.voleq, tev0_target=targets.tev0, r=params.r, T=params.T, q=params.q, s0_guess=guess)
        
        if targets.s0 <= 0.0 and not cal["status"]:
            cal = calibrate_cca(payoffs, shares, s0_target=targets.s0, voleq_target=targets.voleq, tev0_target=targets.tev0, r=params.r, T=params.T, q=params.q, s0_guess=user_input_s0)
            
        params.S0, params.sigma = cal["asset_s0"], cal["volcm"]
        cal_ms, iters = (time.perf_counter() - t_cal) * 1000.0, cal["iterations"]

    S0, T, r, q, sigma, N = params.S0, params.T, params.r, params.q, params.sigma, params.N
    min_k = min(item["params"]["K"] for item in portfolio.values())

    t0 = time.perf_counter()
    prices, deltas = bs_price_and_delta_multiple(S0, T, r, q, sigma, payoffs, N=N, st_min=min_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    st_grid, pdf = (np.array([S0]), np.array([1.0])) if T <= 0.0 else make_mutual_grid(S0, T, r, q, sigma, N=N, st_min=min_k)

    tot_val_curr = sum(prices[k] * shares[k] for k in portfolio)
    tot_delta_curr = sum(deltas[k] * shares[k] for k in portfolio)
    eq_in = targets.voleq if is_cal else (sigma * S0 * tot_delta_curr / tot_val_curr)

    allocs = calc_portfolio_allocations(prices, deltas, shares, eq_in)

    rows, tot_val, tot_delta, tot_vested = [], 0.0, 0.0, 0.0
    for name, item in portfolio.items():
        p, sh = item["params"], item["shares"]
        num, d = prices[name], deltas[name]
        
        omega = allocs[name]["omega"]
        elas = allocs[name]["elas"]
        v_num = allocs[name]["specific_vol"]
        
        tot_val += num * sh
        tot_delta += d * sh
        avg_v = float(item["fn"].vesting(np.array([S0]))[0] if T <= 0.0 else trapezoid(item["fn"].vesting(st_grid) * pdf, st_grid)) * 100.0
        vested_sh = sh * (avg_v / 100.0)
        tot_vested += vested_sh
        rows.append((name, p["K"], item["th_str"], vested_sh, sh, (sh / tot_shares * 100.0) if tot_shares > 0 else 0.0, v_num, num, d, omega, elas, num * sh))

    tot_omega = sum(r[9] for r in rows)
    tot_shares_pct = sum(r[5] for r in rows)
    tot_elas = tot_omega * tot_val / tot_val if tot_val > 1e-8 else 1.0

    tot_vol = 0.0 if tot_val <= 1e-8 else sigma * S0 * tot_delta / tot_val
    eq_in = targets.voleq if is_cal else tot_vol

    user_tev = targets.tev0 if (is_cal and targets.s0 <= 0.0) else tot_val
    user_voleq = targets.voleq if is_cal else eq_in

    summary_str = f"Summary   : Total Equity = ${user_tev:,.2f} | Equity Vol = {user_voleq*100:.2f}% | Implied S0 = ${S0:.4f} | T = {T:.1f}y | r = {r*100:.1f}%"
    
    common_sh = list(portfolio.values())[0]["shares"]
    no_dil = user_tev / max(common_sh, TOL)
    out_dil = user_tev / max(tot_shares, TOL)
    vest_dil = user_tev / max(tot_vested, TOL)
    dilution_str = f"Dilution  : No Dilution: ${no_dil:.2f} | By OPM: ${S0:.2f} | By Vested Shares: ${vest_dil:.2f} | By Outstanding Shares: ${out_dil:.2f}"

    if is_cal:
        diag_str = f"Diagnosis : Converged in {iters} iters ({cal_ms:.1f} ms) | Batch Run Time = {elapsed_ms:.2f} ms"
    else:
        diag_str = f"Diagnosis : Batch Run Time = {elapsed_ms:.2f} ms"

    header_str = f"{'Class':<26} {'Shares':>10} {'Shares (%)':>10} {'Strike K':>8} {'Thres':>13} {'Vested Shares':>13} {'Delta (dV/dS0)':>15} {'Omega (Ω)':>11} {'Elasticity':>11} {'Specific Vol %':>14} {'FV (PS $)':>10} {'Total ($)':>11}"
    box_w = max(len(summary_str), len(dilution_str), len(diag_str), len(header_str))

    out = ["=" * box_w, summary_str, dilution_str, diag_str, "=" * box_w, header_str, "-" * box_w]
    for name, k, th_str, vested_sh, sh, sh_pct, v_num, num, d, omega, elas, val in rows:
        out.append(f"{name:<26} {sh:>10.2f} {sh_pct:>10.2f} {k:>8.2f} {th_str:>13} {vested_sh:>13.2f} {d:>15.4f} {f'{omega*100:.2f}%':>11} {elas:>11.4f} {v_num*100:>14.2f} {num:>10.4f} {val:>11.2f}")
    out.extend([
        "-" * box_w,
        f"{'TOTAL CAPITAL STRUCTURE':<26} {tot_shares:>10.2f} {tot_shares_pct:>10.2f} {'':>8} {'':>13} {tot_vested:>13.2f} {'':>15} {f'{tot_omega*100:.2f}%':>11} {'':>11} {eq_in*100:>14.2f} {'':>10} {tot_val:>11.2f}",
        "=" * box_w
    ])
    
    return "\n".join(out)
