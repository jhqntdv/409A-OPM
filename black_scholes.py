import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import least_squares, minimize
from scipy.stats import norm


# Ensure Windows console can correctly output UTF-8 characters.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


TOL = 1e-12
Z_MIN, Z_MAX = -8.0, 9.0


# Data Models / Classes


@dataclass
class PricingGrid:
    """Shared standard-normal grid, terminal prices, density, Delta weight, and discount factor."""

    z_grid: np.ndarray
    st_grid: np.ndarray
    z_pdf: np.ndarray
    delta_weight: np.ndarray
    discount: float


@dataclass
class MarketParams:
    """Market inputs and integration settings for Capital Structure OPM."""

    S0: float = 75.00
    T: float = 2.0
    r: float = 0.05
    q: float = 0.0
    sigma: float = 0.80
    N: int = 5000


@dataclass
class CalibrationTargets:
    """Target equity market values / volatility used for CCA calibration."""

    tev0: float = 850.0
    voleq: float = 0.80
    s0: float = 0.0


# Payoff / Vesting helpers


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
    payoff.is_step = is_step

    return payoff


# Numerical integration helpers


def _extract_critical_prices(payoffs: Sequence[Callable]) -> Union[list, None]:
    """Extract K and vesting thresholds uniformly from payoffs."""
    critical_prices = []

    for fn in payoffs:
        if hasattr(fn, "K"):
            critical_prices.append(fn.K)
        if hasattr(fn, "thres"):
            if getattr(fn, "is_step", 0):
                for t in fn.thres:
                    critical_prices.extend([t - 1e-7, t])
            else:
                critical_prices.extend(fn.thres)

    return critical_prices or None


def _build_pricing_grid(S0: float, T: float, r: float, q: float, sigma: float, N: int = 5000, critical_prices: Union[Sequence[float], None] = None) -> PricingGrid:
    """Build one shared z-space pricing grid for price, Delta, and vesting integration."""
    if S0 <= 0.0:
        raise ValueError(f"Underlying asset price S0 must be strictly positive, got {S0}")
    if N < 2:
        raise ValueError("N must be at least 2.")

    T_eff = max(T, 0.0)
    disc = np.exp(-r * T_eff)
    if T <= 0.0 or sigma <= 0.0:
        fwd = S0 * np.exp((r - q) * T_eff)
        return PricingGrid(
            z_grid=np.array([0.0]),
            st_grid=np.array([fwd]),
            z_pdf=np.array([1.0]),
            delta_weight=np.array([0.0]),
            discount=disc,
        )

    sqrt_t = np.sqrt(T)
    drift = (r - q - 0.5 * sigma**2) * T
    # [-8, 9] is the base range; extend the right tail only for extreme sigma*sqrt(T).
    z_max = max(Z_MAX, sigma * sqrt_t + 6.0)
    z_grid = np.linspace(Z_MIN, z_max, N + 1)

    if critical_prices is not None:
        cp = np.asarray(critical_prices, dtype=float)
        cp = cp[np.isfinite(cp) & (cp > TOL)]
        if cp.size:
            z_cp = (np.log(cp / S0) - drift) / (sigma * sqrt_t)
            z_cp = z_cp[(z_cp > Z_MIN) & (z_cp < z_max)]
            if z_cp.size:
                z_grid = np.unique(np.concatenate((z_grid, z_cp)))

    st_grid = S0 * np.exp(drift + sigma * sqrt_t * z_grid)
    z_pdf = norm.pdf(z_grid)
    delta_weight = z_pdf * z_grid / (S0 * sigma * sqrt_t)
    return PricingGrid(z_grid=z_grid, st_grid=st_grid, z_pdf=z_pdf, delta_weight=delta_weight, discount=disc)


def make_mutual_grid(S0: float, T: float, r: float, q: float, sigma: float, N: int = 5000, critical_prices: Union[Sequence[float], None] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create a z-spaced terminal-price grid and its lognormal PDF (legacy two-value API)."""
    grid = _build_pricing_grid(S0, T, r, q, sigma, N=N, critical_prices=critical_prices)
    if T <= 0.0 or sigma <= 0.0:
        st_pdf = np.array([1.0])
    else:
        st_pdf = grid.z_pdf / (grid.st_grid * sigma * np.sqrt(T))
    return grid.st_grid, st_pdf


def _price_and_delta_on_grid(payoff: Callable, pricing_grid: PricingGrid) -> Tuple[float, float]:
    """Calculate PV and Delta for one payoff on the shared z-space grid."""
    payoff_values = payoff(pricing_grid.st_grid)
    if pricing_grid.z_grid.size == 1:
        return float(pricing_grid.discount * payoff_values[0]), 0.0

    price = pricing_grid.discount * trapezoid(payoff_values * pricing_grid.z_pdf, pricing_grid.z_grid)
    delta = pricing_grid.discount * trapezoid(payoff_values * pricing_grid.delta_weight, pricing_grid.z_grid)
    return float(price), float(delta)


# Pricing API


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


# Closed-form benchmark


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


def call_closed_form(S0: float, K: float, thres: Union[float, Sequence[float], np.ndarray] = 0.0, vest_pct: Union[float, Sequence[float], np.ndarray] = 1.0, T: float = 1.0, r: float = 0.05, q: float = 0.0, sigma: float = 0.2) -> float:
    """Closed-form benchmark for call option; uses minimal maturity proxy when T<=0."""
    T_eff = 1e-8 if T <= 0.0 else T
    t_arr, v_arr = _normalize_vesting_inputs(thres, vest_pct)
    vesting_increments = np.diff(np.concatenate(([0.0], v_arr)))

    return float(sum(dv * _single_threshold_cf(S0, K, tj, T_eff, r, q, sigma) for tj, dv in zip(t_arr, vesting_increments)))


def call_closed_form_delta(S0: float, K: float, thres: Union[float, Sequence[float], np.ndarray] = 0.0, vest_pct: Union[float, Sequence[float], np.ndarray] = 1.0, T: float = 1.0, r: float = 0.05, q: float = 0.0, sigma: float = 0.2, eps: float = 1e-5) -> float:
    """Calculate Delta benchmark of call_closed_form via centered difference."""
    p_up = call_closed_form(S0 + eps, K, thres, vest_pct, T, r, q, sigma)
    p_dn = call_closed_form(S0 - eps, K, thres, vest_pct, T, r, q, sigma)
    return float((p_up - p_dn) / (2.0 * eps))


# Portfolio helpers


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
    """Calculate class fair values, TEV, TEV Delta, and Merton-style equity volatility."""
    fn_list, sh_arr = _unpack_portfolio(payoffs, shares)
    prices, deltas = bs_price_and_delta_multiple(S0, T, r, q, sigma, fn_list, N=N, insert_critical=insert_critical)

    fvs, ds = np.asarray(prices, dtype=float), np.asarray(deltas, dtype=float)
    tev0 = float(np.dot(fvs, sh_arr))
    tev_delta = float(np.dot(ds, sh_arr))
    voleq = 0.0 if tev0 <= TOL else float(sigma * S0 * tev_delta / tev0)

    return {
        "s0": float(fvs[cm_idx]), "tev0": tev0, "tev_delta": tev_delta,
        "tev_elasticity": 0.0 if tev0 <= TOL else float(S0 * tev_delta / tev0),
        "voleq": voleq, "fair_values": fvs,
    }


# Calibration


def calibrate_cca(
    payoffs: Union[Sequence[Callable], Dict[str, Callable]], shares: Union[Sequence[float], Dict[str, float]],
    s0_target: float, voleq_target: float, tev0_target: float, r: float, T: float, q: float = 0.0, cm_idx: int = 0,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None, N: int = 5000, method: str = "trf", s0_guess: float = None,
) -> Dict[str, Any]:
    """Calibrate underlying S0 and asset volatility to equity value/common value and equity-vol targets."""
    fn_list, sh_arr = _unpack_portfolio(payoffs, shares)
    if voleq_target <= 0.0:
        raise ValueError("voleq_target must be positive for calibration.")
    if s0_target <= 0.0 and tev0_target <= 0.0:
        raise ValueError("tev0_target must be positive when s0_target is not provided.")

    if s0_guess is None:
        s0_guess = tev0_target / max(np.sum(sh_arr), TOL)
    x0 = np.array([s0_target if s0_target > 0 else s0_guess, max(voleq_target, 0.05)], dtype=float)

    if bounds is None:
        common_shares = max(sh_arr[cm_idx], TOL)
        s0_hi = max(10.0 * x0[0], 10.0 * tev0_target / common_shares, 0.02)
        sigma_hi = max(2.0, 1.5 * voleq_target)
        bounds = ((0.01, s0_hi), (0.01, sigma_hi))

    lower = np.array([bounds[0][0], bounds[1][0]], dtype=float)
    upper = np.array([bounds[0][1], bounds[1][1]], dtype=float)
    if np.any(lower >= upper):
        raise ValueError("Each calibration lower bound must be below its upper bound.")
    x0 = np.clip(x0, lower + 1e-10, upper - 1e-10)

    def residuals(x: Sequence[float], insert_critical: bool = False) -> np.ndarray:
        m = equity_moments(x[0], T, r, q, x[1], fn_list, sh_arr, N=N, cm_idx=cm_idx, insert_critical=insert_critical)
        value_resid = (m["s0"] - s0_target) / s0_target if s0_target > 0 else (m["tev0"] - tev0_target) / tev0_target
        vol_resid = (m["voleq"] - voleq_target) / voleq_target
        return np.array([value_resid, vol_resid], dtype=float)

    # A fixed z-grid makes the finite-difference Jacobian smooth. Reprice with critical
    # points afterward; use the robust fallback only if that formal repricing is off.
    res = least_squares(residuals, x0, bounds=(lower, upper), method=method, x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10)
    final_resid = residuals(res.x, insert_critical=True)
    solver, iterations, nfev = "least_squares", int(res.nfev), int(res.nfev)
    optimizer_success = bool(res.success)
    active_mask = res.active_mask.astype(int)
    message = str(res.message)

    # Weird threshold-heavy portfolios can make finite-difference Jacobians locally noisy.
    # Keep least_squares as the fast path and use Nelder-Mead only when the fit is not acceptable.
    if (not res.success) or np.any(np.abs(final_resid) > np.array([1e-3, 5e-3])):
        def obj(x: Sequence[float]) -> float:
            rr = residuals(x, insert_critical=True)
            return float(np.dot(rr, rr))

        nm = minimize(obj, res.x, method="Nelder-Mead", bounds=list(zip(lower, upper)), tol=1e-8, options={"maxiter": 150})
        nm_resid = residuals(nm.x, insert_critical=True)
        if np.dot(nm_resid, nm_resid) < np.dot(final_resid, final_resid):
            res_x, final_resid = nm.x, nm_resid
            solver, iterations, nfev = "Nelder-Mead fallback", int(nm.nit), int(res.nfev + nm.nfev)
            optimizer_success, active_mask, message = bool(nm.success), np.zeros(2, dtype=int), str(nm.message)
            active_mask[np.isclose(res_x, lower, rtol=0.0, atol=1e-8)] = -1
            active_mask[np.isclose(res_x, upper, rtol=0.0, atol=1e-8)] = 1
        else:
            res_x = res.x
    else:
        res_x = res.x

    opt = equity_moments(res_x[0], T, r, q, res_x[1], fn_list, sh_arr, N=N, cm_idx=cm_idx)
    bound_hit = bool(np.any(active_mask != 0))
    calibration_ok = bool(optimizer_success and np.all(np.abs(final_resid) <= np.array([1e-3, 5e-3])))

    opt.update({
        "volcm": float(res_x[1]), "asset_s0": float(res_x[0]),
        "iterations": iterations, "function_evals": nfev, "solver": solver,
        "value_residual": float(final_resid[0]), "vol_residual": float(final_resid[1]),
        "bound_hit": bound_hit, "active_mask": active_mask.astype(int).tolist(),
        "optimizer_success": optimizer_success, "status": calibration_ok,
        "message": message, "bounds": bounds,
    })
    return opt


# Input conversion

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


# Portfolio allocation


def calc_portfolio_allocations(prices: Dict[str, float], deltas: Dict[str, float], shares: Dict[str, float], equity_vol: float, total_value: float, total_delta: float) -> Dict[str, Dict[str, float]]:
    """Calculate Omega, Elasticity, and Specific Volatility allocation."""

    allocs = {}

    for key, price in prices.items():
        delta = deltas[key]
        share = shares[key]

        omega = (delta * share) / total_delta if total_delta > TOL else 0.0
        elas = (delta * total_value) / (price * total_delta) if price > 1e-8 and total_delta > 1e-8 else 0.0
        specific_vol = equity_vol * elas

        allocs[key] = {"omega": omega, "elas": elas, "specific_vol": specific_vol}

    return allocs


# High-level analysis helpers


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

    cal = calibrate_cca(payoffs, shares, s0_target=targets.s0, voleq_target=targets.voleq, tev0_target=targets.tev0,
                        r=params.r, T=params.T, q=params.q, N=params.N, s0_guess=guess)

    if targets.s0 <= 0.0 and not cal["status"]:
        cal = calibrate_cca(payoffs, shares, s0_target=targets.s0, voleq_target=targets.voleq, tev0_target=targets.tev0,
                            r=params.r, T=params.T, q=params.q, N=params.N, s0_guess=user_input_s0)

    params.S0 = cal["asset_s0"]
    params.sigma = cal["volcm"]

    elapsed_ms = (time.perf_counter() - t_cal) * 1000.0
    return cal, elapsed_ms


def _build_report_rows(
    portfolio: dict, prices: Dict[str, float], deltas: Dict[str, float], shares: Dict[str, float],
    total_shares: float, allocations: Dict[str, Dict[str, float]], pricing_grid: PricingGrid,
) -> Tuple[list, float]:
    """Build instrument rows and probability-weighted expected vested shares."""
    rows, total_vested = [], 0.0
    deterministic = pricing_grid.z_grid.size == 1
    for name, item in portfolio.items():
        sh, fv, d, alloc = item["shares"], prices[name], deltas[name], allocations[name]
        vest = item["fn"].vesting(pricing_grid.st_grid)
        avg_v = float(vest[0]) if deterministic else float(trapezoid(vest * pricing_grid.z_pdf, pricing_grid.z_grid))
        vested_sh = sh * avg_v
        total_vested += vested_sh
        sh_pct = sh / total_shares * 100.0 if total_shares > 0 else 0.0
        rows.append((name, item["params"]["K"], item["th_str"], vested_sh, sh, sh_pct,
                     alloc["specific_vol"], fv, d, alloc["omega"], alloc["elas"], fv * sh))
    return rows, total_vested


def _format_report(
    summary_str: str, per_share_str: str, diag_str: str, rows: list, total_shares: float,
    total_shares_pct: float, total_vested: float, total_omega: float, model_eq_vol: float, total_value: float,
    calibration_str: str = "", warning_str: str = "",
) -> str:
    """Format the compact practitioner-facing text report."""
    header_str = (
        f"{'Class':<26} {'Shares':>14} {'Shares (%)':>10} {'Strike K':>10} {'Thres':>13} {'Exp. Vested':>14} "
        f"{'Delta (dV/dS0)':>15} {'Omega (%)':>11} {'Elasticity':>11} {'Model Vol %':>13} {'FV / Per Share ($)':>18} {'Total ($)':>15}"
    )
    info = [summary_str]
    if calibration_str:
        info.append(calibration_str)
    info.extend([per_share_str, diag_str])
    if warning_str:
        info.append(warning_str)
    box_w = max([len(header_str)] + [len(x) for x in info])
    out = ["=" * box_w, *info, "=" * box_w, header_str, "-" * box_w]
    for name, k, th_str, vested_sh, sh, sh_pct, v_num, fv, d, omega, elas, val in rows:
        out.append(
            f"{name:<26} {sh:>14,.2f} {sh_pct:>10.2f} {k:>10.2f} {th_str:>13} {vested_sh:>14,.2f} "
            f"{d:>15.4f} {omega * 100:>11.2f} {elas:>11.4f} {v_num * 100:>13.2f} {fv:>18.4f} {val:>15,.2f}"
        )
    out.extend([
        "-" * box_w,
        f"{'TOTAL CAPITAL STRUCTURE':<26} {total_shares:>14,.2f} {total_shares_pct:>10.2f} {'':>10} {'':>13} {total_vested:>14,.2f} "
        f"{'':>15} {total_omega * 100:>11.2f} {'':>11} {model_eq_vol * 100:>13.2f} {'':>18} {total_value:>15,.2f}",
        "=" * box_w,
    ])
    return "\n".join(out)


# High-level OPM analysis


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


def run_opm_analysis(portfolio: dict, params: MarketParams = None, targets: CalibrationTargets = None, insert_critical: bool = True) -> str:
    """Run OPM analysis and return a compact valuation and calibration report."""
    if params is None:
        params = MarketParams()
    is_cal = targets is not None
    _validate_common_stock(portfolio)
    payoffs = {k: v["fn"] for k, v in portfolio.items()}
    shares = {k: v["shares"] for k, v in portfolio.items()}
    total_shares = sum(shares.values())
    cal, cal_ms, iters = None, 0.0, 0

    if is_cal:
        cal, cal_ms = _run_calibration(portfolio, payoffs, shares, params, targets)
        iters = cal["iterations"]

    S0, T, r, q, sigma, N = params.S0, params.T, params.r, params.q, params.sigma, params.N
    prices, deltas, pricing_grid, elapsed_ms = _execute_pricing(S0, T, r, q, sigma, N, payoffs, insert_critical)
    total_value, total_delta = _calculate_portfolio_totals(prices, deltas, shares)
    model_eq_vol = 0.0 if total_value <= TOL else sigma * S0 * total_delta / total_value

    allocations = calc_portfolio_allocations(prices, deltas, shares, model_eq_vol, total_value, total_delta)
    rows, total_vested = _build_report_rows(portfolio, prices, deltas, shares, total_shares, allocations, pricing_grid)
    total_omega, total_shares_pct = sum(r[9] for r in rows), sum(r[5] for r in rows)

    first_name = next(iter(portfolio))
    common_fv = prices[first_name]
    common_shares = portfolio[first_name]["shares"]
    tev_elasticity = 0.0 if total_value <= TOL else S0 * total_delta / total_value

    summary_str = (
        f"Summary   : Model TEV = ${total_value:,.2f} | Model Equity Vol = {model_eq_vol * 100:.2f}% | "
        f"Underlying S0 = ${S0:.4f} | Asset Vol = {sigma * 100:.2f}% | T = {T:.2f}y | r = {r * 100:.2f}%"
    )
    per_share_str = (
        f"Per Share : TEV/Common Shares = ${total_value / max(common_shares, TOL):,.2f} | Common FV = ${common_fv:,.4f} | "
        f"TEV/Exp. Vested Shares = ${total_value / max(total_vested, TOL):,.2f} | "
        f"TEV/Outstanding Shares = ${total_value / max(total_shares, TOL):,.2f} | TEV Elasticity = {tev_elasticity:.4f}x"
    )
    diag_str = f"Diagnosis : Converged in {iters} iters ({cal_ms:.1f} ms) | Batch Run Time = {elapsed_ms:.2f} ms" if is_cal else f"Diagnosis : Batch Run Time = {elapsed_ms:.2f} ms"

    calibration_str, warning_str = "", ""
    if is_cal:
        if targets.s0 > 0.0:
            value_resid = common_fv / targets.s0 - 1.0
            calibration_str = (
                f"Calibration: Common FV Target = ${targets.s0:,.4f} | Model = ${common_fv:,.4f} | "
                f"Equity Vol Target = {targets.voleq * 100:.2f}% | Model = {model_eq_vol * 100:.2f}%"
            )
        else:
            value_resid = total_value / targets.tev0 - 1.0
            calibration_str = (
                f"Calibration: TEV Target = ${targets.tev0:,.2f} | Model = ${total_value:,.2f} | "
                f"Equity Vol Target = {targets.voleq * 100:.2f}% | Model = {model_eq_vol * 100:.2f}%"
            )
        vol_resid = model_eq_vol / targets.voleq - 1.0
        warnings = []
        if not cal["optimizer_success"]:
            warnings.append("optimizer did not report success")
        if abs(value_resid) > 1e-3:
            warnings.append(f"value residual {value_resid:+.2%}")
        if abs(vol_resid) > 5e-3:
            warnings.append(f"equity-vol residual {vol_resid:+.2%}")
        if cal["bound_hit"]:
            labels = ["S0", "asset vol"]
            hits = [labels[i] for i, flag in enumerate(cal["active_mask"]) if flag != 0]
            warnings.append(f"calibration bound active ({', '.join(hits)})")
        if warnings:
            warning_str = "Warning   : " + " | ".join(warnings)

    return _format_report(summary_str, per_share_str, diag_str, rows, total_shares, total_shares_pct,
                          total_vested, total_omega, model_eq_vol, total_value, calibration_str, warning_str)

