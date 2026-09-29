import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from scipy.stats import lognorm, norm


# Ensure Windows console can correctly output UTF-8 characters.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


TOL = 1e-12


# -----------------------------------------------------------------------------
# Payoff / Vesting helpers
# -----------------------------------------------------------------------------


def _normalize_vesting_inputs(thres: Union[float, Sequence[float], np.ndarray], vest_pct: Union[float, Sequence[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize threshold and vesting percentage into 1D numpy arrays."""
    t_arr = np.atleast_1d(np.asarray(thres, dtype=float))
    v_arr = np.atleast_1d(np.asarray(vest_pct, dtype=float))
    return t_arr, v_arr


def call_option(K: float, thres: Union[float, Sequence[float], np.ndarray] = 0.0, vest_pct: Union[float, Sequence[float], np.ndarray] = 1.0, is_step: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    """Create Call Option payoff with step or linear vesting."""
    t_arr, v_arr = _normalize_vesting_inputs(thres, vest_pct)

    if len(t_arr) != len(v_arr):
        raise ValueError("thres and vest_pct must have identical lengths.")
    if np.any(np.diff(t_arr) < 0.0):
        raise ValueError("thres array must be in non-decreasing order.")
    if np.any(np.diff(v_arr) < 0.0):
        raise ValueError("vest_pct array must be in non-decreasing order.")
    if np.any(v_arr < 0.0):
        raise ValueError("vest_pct must be >= 0.0.")

    v_padded = np.concatenate(([0.0], v_arr))

    def vesting_fn(ST: np.ndarray) -> np.ndarray:
        ST = np.asarray(ST, dtype=float)

        if is_step:
            return v_padded[np.searchsorted(t_arr, ST, side="right")]

        return np.interp(ST, t_arr, v_arr, left=0.0, right=v_arr[-1])

    def payoff(ST: np.ndarray) -> np.ndarray:
        return np.maximum(ST - K, 0.0) * vesting_fn(ST)

    # Preserve original function metadata API to avoid changing external usage.
    payoff.vesting = vesting_fn
    payoff.K = K
    payoff.thres = t_arr
    payoff.vest_pct = v_arr

    return payoff


# -----------------------------------------------------------------------------
# Numerical integration helpers
# -----------------------------------------------------------------------------


def _extract_critical_prices(payoffs: Sequence[Callable]) -> Union[list, None]:
    """Extract K and vesting thresholds uniformly from payoffs."""
    critical_prices = []

    for fn in payoffs:
        if hasattr(fn, "K"):
            critical_prices.append(fn.K)
        if hasattr(fn, "thres"):
            critical_prices.extend(fn.thres)

    return critical_prices or None


def make_mutual_grid(S0: float, T: float, r: float, q: float, sigma: float, N: int = 5000, critical_prices: Sequence[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create shared terminal value grid and BSM lognormal PDF for all payoffs."""
    if S0 <= 0.0:
        raise ValueError(f"Underlying asset price S0 must be strictly positive, got {S0}")

    if T <= 0.0 or sigma <= 0.0:
        return np.array([S0]), np.array([1.0])

    mu = np.log(S0) + (r - q - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    Z = np.linspace(-8.0, 9.0, N + 1)
    st_grid = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if critical_prices is not None:
        cp = np.asarray(critical_prices)
        cp = cp[cp > TOL]
        st_grid = np.unique(np.concatenate((st_grid, cp)))

    return st_grid, lognorm.pdf(st_grid, s=s, scale=np.exp(mu))


def make_delta_weight(S0: float, T: float, r: float, q: float, sigma: float, st_grid: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    """Create LRM Delta kernel weight."""
    if T <= 0.0 or sigma <= 0.0:
        return np.zeros_like(st_grid)

    z = np.log(np.maximum(st_grid, TOL) / S0) - (r - q - 0.5 * sigma**2) * T
    w = pdf * (z / (S0 * sigma**2 * T))

    if st_grid[0] <= 0.0:
        w[0] = 0.0

    return w


@dataclass
class PricingGrid:
    """Shared grid, PDF, Delta weight, and discount factor for OPM integration."""

    st_grid: np.ndarray
    pdf: np.ndarray
    delta_weight: np.ndarray
    discount: float


def _build_pricing_grid(S0: float, T: float, r: float, q: float, sigma: float, N: int, critical_prices: Union[Sequence[float], None] = None) -> PricingGrid:
    """Build a pricing grid once, shared across price, delta, and vesting integration."""
    st_grid, pdf = make_mutual_grid(S0, T, r, q, sigma, N=N, critical_prices=critical_prices)
    delta_weight = make_delta_weight(S0, T, r, q, sigma, st_grid, pdf)

    return PricingGrid(st_grid=st_grid, pdf=pdf, delta_weight=delta_weight, discount=np.exp(-r * T))


def _price_and_delta_on_grid(payoff: Callable, pricing_grid: PricingGrid) -> Tuple[float, float]:
    """Calculate PV and Delta for a single payoff on the established shared grid."""
    payoff_values = payoff(pricing_grid.st_grid)

    price = pricing_grid.discount * trapezoid(payoff_values * pricing_grid.pdf, pricing_grid.st_grid)
    delta = pricing_grid.discount * trapezoid(payoff_values * pricing_grid.delta_weight, pricing_grid.st_grid)

    return float(price), float(delta)


# -----------------------------------------------------------------------------
# Pricing API
# -----------------------------------------------------------------------------


def bs_price_and_delta_multiple(
    S0: float, T: float, r: float, q: float, sigma: float, payoffs: Union[Dict[str, Callable], Sequence[Callable]], N: int = 5000, insert_critical: bool = True
) -> Tuple[Any, Any]:
    """Simultaneously calculate prices and Deltas for all payoffs in the portfolio using a single mutual grid."""
    is_dict = isinstance(payoffs, dict)
    items = list(payoffs.items()) if is_dict else list(enumerate(payoffs))

    # When T or sigma are invalid, fallback to intrinsic-value / finite-difference logic.
    if T <= 0.0 or sigma <= 0.0:
        eps = 1e-4
        T_eff = max(T, 0.0)
        fwd = S0 * np.exp((r - q) * T_eff)
        disc = np.exp(-r * T_eff)

        p = {k: float(disc * fn(np.array([fwd]))[0]) for k, fn in items}
        d = {k: float(disc * (fn(np.array([(S0 + eps) * np.exp((r - q) * T_eff)]))[0] - fn(np.array([(S0 - eps) * np.exp((r - q) * T_eff)]))[0]) / (2.0 * eps)) for k, fn in items}

        return (p if is_dict else list(p.values())), (d if is_dict else list(d.values()))

    critical_prices = _extract_critical_prices([fn for _, fn in items]) if insert_critical else None

    pricing_grid = _build_pricing_grid(S0, T, r, q, sigma, N=N, critical_prices=critical_prices)

    prices: Dict[Any, float] = {}
    deltas: Dict[Any, float] = {}

    for key, fn in items:
        prices[key], deltas[key] = _price_and_delta_on_grid(fn, pricing_grid)

    return (prices, deltas) if is_dict else (list(prices.values()), list(deltas.values()))


# -----------------------------------------------------------------------------
# Closed-form benchmark
# -----------------------------------------------------------------------------


def _single_threshold_cf(S0: float, K: float, thres: float, T: float, r: float, q: float, sigma: float) -> float:
    """Closed-form call benchmark under a single vesting threshold."""
    eff_k = max(thres, K)

    if eff_k <= 0.0:
        return float(S0 * np.exp(-q * T) - K * np.exp(-r * T))

    if sigma <= 0.0:
        fwd = S0 * np.exp((r - q) * T)
        return float(np.exp(-r * T) * (fwd - K)) if fwd >= eff_k else 0.0

    d1 = (np.log(S0 / eff_k) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return float(S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T)))


def call_closed_form(
    S0: float,
    K: float,
    thres: Union[float, Sequence[float], np.ndarray] = 0.0,
    vest_pct: Union[float, Sequence[float], np.ndarray] = 1.0,
    T: float = 1.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
) -> float:
    """Closed-form benchmark for call option; uses minimal maturity proxy when T<=0."""
    T_eff = 1e-8 if T <= 0.0 else T
    t_arr, v_arr = _normalize_vesting_inputs(thres, vest_pct)
    vesting_increments = np.diff(np.concatenate(([0.0], v_arr)))

    return float(sum(dv * _single_threshold_cf(S0, K, tj, T_eff, r, q, sigma) for tj, dv in zip(t_arr, vesting_increments)))


def call_closed_form_delta(
    S0: float,
    K: float,
    thres: Union[float, Sequence[float], np.ndarray] = 0.0,
    vest_pct: Union[float, Sequence[float], np.ndarray] = 1.0,
    T: float = 1.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
    eps: float = 1e-5,
) -> float:
    """Calculate Delta benchmark of call_closed_form via centered difference."""
    p_up = call_closed_form(S0 + eps, K, thres, vest_pct, T, r, q, sigma)
    p_dn = call_closed_form(S0 - eps, K, thres, vest_pct, T, r, q, sigma)
    return float((p_up - p_dn) / (2.0 * eps))


# -----------------------------------------------------------------------------
# Portfolio helpers
# -----------------------------------------------------------------------------


def _unpack_portfolio(payoffs: Union[Sequence, Dict], shares: Union[Sequence, Dict]) -> Tuple[list, np.ndarray]:
    """Unpack dict/sequence portfolio into payoff list and shares array."""
    if isinstance(payoffs, dict):
        fn_list = list(payoffs.values())

        if isinstance(shares, dict):
            sh_list = [shares[k] for k in payoffs]
        else:
            sh_list = list(shares)
    else:
        fn_list = list(payoffs)
        sh_list = list(shares)

    return fn_list, np.asarray(sh_list, dtype=float)


def _calculate_portfolio_totals(prices: Dict[str, float], deltas: Dict[str, float], shares: Dict[str, float]) -> Tuple[float, float]:
    """Calculate total value and total Delta of the portfolio."""
    total_value = sum(prices[k] * shares[k] for k in prices)
    total_delta = sum(deltas[k] * shares[k] for k in prices)
    return total_value, total_delta


def equity_moments(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    payoffs: Union[Sequence[Callable], Dict[str, Callable]],
    shares: Union[Sequence[float], Dict[str, float]],
    N: int = 5000,
    cm_idx: int = 0,
    insert_critical: bool = True,
) -> Dict[str, Any]:
    """Calculate FV for each class, total equity value, and aggregate equity volatility."""
    fn_list, sh_arr = _unpack_portfolio(payoffs, shares)
    prices, deltas = bs_price_and_delta_multiple(S0, T, r, q, sigma, fn_list, N=N, insert_critical=insert_critical)

    fvs = np.array(prices)
    ds = np.array(deltas)
    tev0 = float(np.sum(fvs * sh_arr))
    tev_delta = float(np.sum(ds * sh_arr))

    voleq = sigma * S0 * tev_delta / max(tev0, TOL)

    return {"s0": float(fvs[cm_idx]), "tev0": tev0, "tev_delta": tev_delta, "voleq": voleq, "fair_values": fvs, "check_tev_ratio": 1.0}


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------


def calibrate_cca(
    payoffs: Union[Sequence[Callable], Dict[str, Callable]], shares: Union[Sequence[float], Dict[str, float]],
    s0_target: float, voleq_target: float, tev0_target: float, r: float, T: float, q: float = 0.0, cm_idx: int = 0,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None, N: int = 5000, method: str = "Nelder-Mead", s0_guess: float = None,
) -> Dict[str, Any]:
    """Calibrate underlying S0 and common asset volatility to match target equity metrics."""
    fn_list, sh_arr = _unpack_portfolio(payoffs, shares)

    if s0_guess is None:
        s0_guess = tev0_target / max(np.sum(sh_arr), TOL)

    sigma_guess = voleq_target
    x0 = [s0_target if s0_target > 0 else s0_guess, sigma_guess]

    if bounds is None:
        bounds = ((0.01, max(10.0 * x0[0], 0.02)), (0.05, 2.00))

    def obj(x: Sequence[float]) -> float:
        moments = equity_moments(x[0], T, r, q, x[1], fn_list, sh_arr, N=N, cm_idx=cm_idx)

        err_s0 = ((moments["s0"] - s0_target) / max(s0_target, TOL)) ** 2 if s0_target > 0 else 0.0
        err_tev = ((moments["tev0"] - tev0_target) / max(tev0_target, TOL)) ** 2 if s0_target <= 0 else 0.0
        err_vol = ((moments["voleq"] - voleq_target) / max(voleq_target, TOL)) ** 2

        return float(err_s0 + err_tev + err_vol)

    res = minimize(obj, x0, method=method, bounds=bounds, tol=1e-4)

    if not res.success and method != "Nelder-Mead":
        res = minimize(obj, x0, method="Nelder-Mead", bounds=bounds, tol=1e-4)

    opt = equity_moments(res.x[0], T, r, q, res.x[1], fn_list, sh_arr, N=N, cm_idx=cm_idx)

    opt.update(
        {
            "volcm": float(res.x[1]),
            "asset_s0": float(res.x[0]),
            "iterations": int(res.nit),
            "check_ratios": [opt["check_tev_ratio"], opt["voleq"] / max(voleq_target, TOL)],
            "status": bool(res.success),
            "message": str(res.message),
        }
    )

    return opt


# -----------------------------------------------------------------------------
# Input conversion
# -----------------------------------------------------------------------------

# Field positions for metric. Preserve original 7-column input contract, naming the magic indices.
METRIC_SHARES = 0
METRIC_STRIKE = 1
METRIC_THRESH_LOW = 2
METRIC_THRESH_HIGH = 3
METRIC_IS_COMMON = 4
METRIC_VESTING_TYPE = 5


def metric_to_portfolio(metric: np.ndarray) -> Tuple[list, list]:
    """Convert 7-column metric array from sample.py into (payoffs, shares)."""
    payoffs = []
    shares = []

    for row in metric:
        shares.append(float(row[METRIC_SHARES]))

        strike = row[METRIC_STRIKE]
        low_threshold = row[METRIC_THRESH_LOW]
        high_threshold = row[METRIC_THRESH_HIGH]
        is_common = bool(row[METRIC_IS_COMMON])
        vesting_type = row[METRIC_VESTING_TYPE]

        if is_common or vesting_type == 0:
            payoffs.append(call_option(K=strike))
        elif vesting_type == 1:
            payoffs.append(call_option(K=strike, thres=low_threshold, vest_pct=1.0, is_step=1))
        else:
            payoffs.append(call_option(K=strike, thres=[low_threshold, high_threshold], vest_pct=[0.0, 1.0], is_step=0))

    return payoffs, shares


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class MarketParams:
    """Market inputs and integration settings for Capital Structure OPM."""

    S0: float = 62.4253
    T: float = 2.0
    r: float = 0.05
    q: float = 0.0
    sigma: float = 0.6901
    N: int = 10000


@dataclass
class CalibrationTargets:
    """Target equity market values / volatility used for CCA calibration."""

    tev0: float = 850.0
    voleq: float = 0.80
    s0: float = 0.0


# -----------------------------------------------------------------------------
# Portfolio allocation
# -----------------------------------------------------------------------------


def calc_portfolio_allocations(prices: Dict[str, float], deltas: Dict[str, float], shares: Dict[str, float], eq_in: float, total_value: float, total_delta: float) -> Dict[str, Dict[str, float]]:
    """Calculate Omega, Elasticity, and Specific Volatility allocation."""

    allocs = {}

    for key, price in prices.items():
        delta = deltas[key]
        share = shares[key]

        omega = (delta * share) / total_delta if total_delta > TOL else 0.0
        elas = (delta * total_value) / (price * total_delta) if price > 1e-8 and total_delta > 1e-8 else 0.0
        specific_vol = eq_in * elas

        allocs[key] = {"omega": omega, "elas": elas, "specific_vol": specific_vol}

    return allocs


# -----------------------------------------------------------------------------
# High-level analysis helpers
# -----------------------------------------------------------------------------


def _validate_common_stock(portfolio: dict) -> None:
    """Ensure the first class in the portfolio matches the expected payoff structure for Common Stock."""
    if len(portfolio) == 0:
        return

    first_name = next(iter(portfolio))
    cm_fn = portfolio[first_name]["fn"]
    K_val = getattr(cm_fn, "K", -1.0)
    thres_val = getattr(cm_fn, "thres", np.array([-1.0]))
    vest_val = getattr(cm_fn, "vest_pct", np.array([-1.0]))

    if K_val != 0.0 or not np.array_equal(thres_val, [0.0]) or not np.array_equal(vest_val, [1.0]):
        raise ValueError(
            f"Validation Error: The first portfolio class (Index 0) MUST be the Common Stock. "
            f"Expected: Strike K=0.0, thres=0.0, vest_pct=1.0 (100%). "
            f"Got: K={K_val}, thres={thres_val.tolist()}, vest_pct={vest_val.tolist()} "
            f"for class '{first_name}'."
        )


def _run_calibration(portfolio: dict, payoffs: Dict[str, Callable], shares: Dict[str, float], params: MarketParams, targets: CalibrationTargets) -> Tuple[Dict[str, Any], float]:
    """Execute CCA calibration and preserve the original fallback logic."""
    t_cal = time.perf_counter()

    total_shares = sum(shares.values())

    # Gross-up initial guess: add option proceeds back to TEV then divide by total shares.
    total_proceeds = sum(sh * portfolio[key]["params"].get("K", 0.0) for key, sh in shares.items())
    guess = (targets.tev0 + total_proceeds) / max(total_shares, TOL)

    user_input_s0 = params.S0

    cal = calibrate_cca(payoffs, shares, s0_target=targets.s0, voleq_target=targets.voleq, tev0_target=targets.tev0, r=params.r, T=params.T, q=params.q, s0_guess=guess)

    if targets.s0 <= 0.0 and not cal["status"]:
        cal = calibrate_cca(payoffs, shares, s0_target=targets.s0, voleq_target=targets.voleq, tev0_target=targets.tev0, r=params.r, T=params.T, q=params.q, s0_guess=user_input_s0)

    params.S0 = cal["asset_s0"]
    params.sigma = cal["volcm"]

    elapsed_ms = (time.perf_counter() - t_cal) * 1000.0
    return cal, elapsed_ms


def _build_report_rows(
    portfolio: dict, prices: Dict[str, float], deltas: Dict[str, float], shares: Dict[str, float],
    total_shares: float, allocations: Dict[str, Dict[str, float]], S0: float, T: float, pricing_grid: PricingGrid,
) -> Tuple[list, float]:
    """Build report rows and synchronously calculate vested shares."""
    rows, total_vested = [], 0.0
    for name, item in portfolio.items():
        sh, fv, d, alloc = item["shares"], prices[name], deltas[name], allocations[name]
        avg_v = float(item["fn"].vesting(np.array([S0]))[0]) if T <= 0.0 else float(trapezoid(item["fn"].vesting(pricing_grid.st_grid) * pricing_grid.pdf, pricing_grid.st_grid))
        vested_sh = sh * avg_v
        total_vested += vested_sh
        sh_pct = sh / total_shares * 100.0 if total_shares > 0 else 0.0
        rows.append((name, item["params"]["K"], item["th_str"], vested_sh, sh, sh_pct, alloc["specific_vol"], fv, d, alloc["omega"], alloc["elas"], fv * sh))
    return rows, total_vested


def _format_report(
    summary_str: str, dilution_str: str, diag_str: str, rows: list, total_shares: float,
    total_shares_pct: float, total_vested: float, total_omega: float, eq_in: float, total_value: float,
) -> str:
    """Format analysis results into the original text report."""
    header_str = (
        f"{'Class':<26} {'Shares':>10} {'Shares (%)':>10} {'Strike K':>8} {'Thres':>13} {'Vested Shares':>13} "
        f"{'Delta (dV/dS0)':>15} {'Omega (Ω)':>11} {'Elasticity':>11} {'Specific Vol %':>14} {'FV (PS $)':>10} {'Total ($)':>11}"
    )
    box_w = max(len(summary_str), len(dilution_str), len(diag_str), len(header_str))
    out = ["=" * box_w, summary_str, dilution_str, diag_str, "=" * box_w, header_str, "-" * box_w]
    for name, k, th_str, vested_sh, sh, sh_pct, v_num, num, d, omega, elas, val in rows:
        out.append(f"{name:<26} {sh:>10.2f} {sh_pct:>10.2f} {k:>8.2f} {th_str:>13} {vested_sh:>13.2f} {d:>15.4f} {f'{omega * 100:.2f}%':>11} {elas:>11.4f} {v_num * 100:>14.2f} {num:>10.4f} {val:>11.2f}")
    out.extend([
        "-" * box_w,
        f"{'TOTAL CAPITAL STRUCTURE':<26} {total_shares:>10.2f} {total_shares_pct:>10.2f} {'':>8} {'':>13} {total_vested:>13.2f} {'':>15} {f'{total_omega * 100:.2f}%':>11} {'':>11} {eq_in * 100:>14.2f} {'':>10} {total_value:>11.2f}",
        "=" * box_w,
    ])
    return "\n".join(out)


# -----------------------------------------------------------------------------
# High-level OPM analysis
# -----------------------------------------------------------------------------


def _execute_pricing(S0: float, T: float, r: float, q: float, sigma: float, N: int, payoffs: dict, insert_critical: bool) -> Tuple[dict, dict, PricingGrid, float]:
    t0 = time.perf_counter()
    if T <= 0.0 or sigma <= 0.0:
        prices, deltas = bs_price_and_delta_multiple(S0, T, r, q, sigma, payoffs, N=N, insert_critical=insert_critical)
        pricing_grid = _build_pricing_grid(S0, T, r, q, sigma, N=N, critical_prices=None)
    else:
        critical_prices = _extract_critical_prices(list(payoffs.values())) if insert_critical else None
        pricing_grid = _build_pricing_grid(S0, T, r, q, sigma, N=N, critical_prices=critical_prices)
        prices, deltas = {}, {}
        for key, fn in payoffs.items():
            prices[key], deltas[key] = _price_and_delta_on_grid(fn, pricing_grid)
    return prices, deltas, pricing_grid, (time.perf_counter() - t0) * 1000.0


def run_opm_analysis(portfolio: dict, params: MarketParams, targets: CalibrationTargets = None, insert_critical: bool = True) -> str:
    """Run OPM analysis and return formatted report string."""
    is_cal = targets is not None
    _validate_common_stock(portfolio)
    payoffs, shares = {k: v["fn"] for k, v in portfolio.items()}, {k: v["shares"] for k, v in portfolio.items()}
    total_shares = sum(shares.values())
    cal, cal_ms, iters = None, 0.0, 0

    if is_cal:
        cal, cal_ms = _run_calibration(portfolio, payoffs, shares, params, targets)
        iters = cal["iterations"]

    S0, T, r, q, sigma, N = params.S0, params.T, params.r, params.q, params.sigma, params.N
    prices, deltas, pricing_grid, elapsed_ms = _execute_pricing(S0, T, r, q, sigma, N, payoffs, insert_critical)
    total_value, total_delta = _calculate_portfolio_totals(prices, deltas, shares)

    eq_in = targets.voleq if is_cal else sigma * S0 * total_delta / total_value
    allocations = calc_portfolio_allocations(prices, deltas, shares, eq_in, total_value, total_delta)
    rows, total_vested = _build_report_rows(portfolio, prices, deltas, shares, total_shares, allocations, S0, T, pricing_grid)

    total_omega, total_shares_pct = sum(r[9] for r in rows), sum(r[5] for r in rows)
    total_vol = 0.0 if total_value <= 1e-8 else sigma * S0 * total_delta / total_value
    eq_in = targets.voleq if is_cal else total_vol
    user_tev = targets.tev0 if is_cal and targets.s0 <= 0.0 else total_value

    summary_str = f"Summary   : Total Equity = ${user_tev:,.2f} | Equity Vol = {eq_in * 100:.2f}% | Implied S0 = ${S0:.4f} | T = {T:.1f}y | r = {r * 100:.1f}%"
    common_shares = list(portfolio.values())[0]["shares"]
    no_dil, out_dil, vest_dil = user_tev / max(common_shares, TOL), user_tev / max(total_shares, TOL), user_tev / max(total_vested, TOL)
    dilution_str = f"Dilution  : No Dilution: ${no_dil:.2f} | By OPM: ${S0:.2f} | By Vested Shares: ${vest_dil:.2f} | By Outstanding Shares: ${out_dil:.2f}"
    diag_str = f"Diagnosis : Converged in {iters} iters ({cal_ms:.1f} ms) | Batch Run Time = {elapsed_ms:.2f} ms" if is_cal else f"Diagnosis : Batch Run Time = {elapsed_ms:.2f} ms"

    return _format_report(summary_str, dilution_str, diag_str, rows, total_shares, total_shares_pct, total_vested, total_omega, eq_in, total_value)
