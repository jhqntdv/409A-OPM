# Capital Structure OPM - Trapezoidal Pricing Engine & LRM Greeks

<div align="right">
  <strong>Language / 語言:</strong>
  <b>English</b> | <a href="opm_cn.md">繁體中文</a>
</div>

> A Python engine for batch valuation and risk sensitivity analysis of complex corporate capital structures using the Trapezoidal Rule and Likelihood Ratio Method (LRM).

---

## Table of Contents

1. [Introduction](#1-introduction)
   - [1.1 OPM Capital Structure Tree](#11-opm-capital-structure-tree)
   - [1.2 CCA (OPM) for Private Company Practice](#12-cca-opm-for-private-company-practice)
2. [Trapezoidal Rule Pricing](#2-trapezoidal-rule-pricing)
3. [Extension to Exotic Stock-Based Compensation Options](#3-extension-to-exotic-stock-based-compensation-options)
4. [Design Architecture](#4-design-architecture)
5. [CCA Calibration and Volatility Allocation](#5-cca-calibration-and-volatility-allocation)
6. [Complete Execution Example](#6-complete-execution-example)
7. [Dependencies & Setup](#7-dependencies--setup)

---

## 1. Introduction

In the valuation of private companies and complex equity structures, the Option Pricing Model (OPM) and Contingent Claim Analysis (CCA) are the core methodologies under International Financial Reporting Standards (IFRS 2) and US GAAP (ASC 718 / AICPA Valuation Guide) for allocating fair value across multiple equity classes.

This project provides a high-performance numerical integration and Likelihood Ratio Method (LRM) engine capable of simultaneously batch-computing present values (fair values) and risk sensitivities (Delta) for dozens of option classes with diverse terms across a single shared grid.

### 1.1 OPM Capital Structure Tree

The hierarchical relationship of the 50-class capital structure implemented in `portfolio.py` is as follows:

```
Total Enterprise / Equity Value (TEV, S0)
├── Class 01: Common Stock
│   ├── Share Weight: 10% - 100% (scenario-dependent, default 50% / 10.0 shares)
│   ├── Strike Price: K = 0.0 (no exercise threshold)
│   └── Vesting Condition: 100% fully vested (vest_pct = 1.0)
│
├── Class 02 - 08: Stock Options (Employee Stock Options / Vanilla Options)
│   ├── Number of Classes: 7 classes (0.50 shares each, 3.5 shares total)
│   ├── Strike Range: K in [10.0, 40.0]
│   └── Vesting Condition: 100% fully vested (no price hurdle, thres = 0.0)
│
├── Class 09 - 10: Early Vest Options
│   ├── Number of Classes: 2 classes (0.50 shares each, 1.0 shares total)
│   ├── Strike Range: K in [45.0, 50.0]
│   ├── Hurdle Condition: thres = max(K - 20.0, 0.1)
│   └── Vesting Percentage: 50% vested once hurdle is met (vest_pct = 0.50)
│
├── Class 11 - 20: Warrants (Threshold Warrants)
│   ├── Number of Classes: 10 classes (0.20 shares each, 2.0 shares total)
│   ├── Strike Range: K in [55.0, 100.0]
│   ├── Hurdle Condition: thres = K + 50.0
│   └── Vesting Percentage: 85% vested once hurdle is met (vest_pct = 0.85)
│
├── Class 21 - 25: Mimic Linear Options
│   ├── Number of Classes: 5 classes (0.15 shares each, 0.75 shares total)
│   ├── Strike Range: K in [105.0, 125.0]
│   ├── Hurdle Range: thres = [K + 10.0 .. K + 110.0] (100-step linear approximation)
│   └── Vesting Percentage: Linear step vesting (0.01 to 1.0)
│
├── Class 26 - 30: True Linear Options
│   ├── Number of Classes: 5 classes (0.15 shares each, 0.75 shares total)
│   ├── Strike Range: K in [130.0, 150.0]
│   ├── Hurdle Range: thres = [K + 10.0 -> K + 110.0] (continuous linear)
│   └── Vesting Percentage: Continuous linear vesting (0.0 to 1.0)
│
└── Class 31 - 50: Incentive Units (Management Incentive Units)
    ├── Number of Classes: 20 classes (0.10 shares each, 2.0 shares total)
    ├── Strike Range: K in [155.0, 250.0]
    ├── Two-Tier Hurdles: thres = [K + 30.0, K + 150.0]
    └── Step Vesting: 50% vests above K + 30; 100% vests above K + 150
```

### 1.2 CCA (OPM) for Private Company Practice

In private company equity valuation (e.g., 409A and IFRS 2), valuation specialists typically assume that Total Equity Value (TEV) follows a geometric Brownian motion (with log-normally distributed returns) and perform waterfall allocations using either Black-Scholes OPM breakpoint analysis or Monte Carlo simulation. In capital structures consisting of common shares alongside various options and warrants (or where preferred shares are treated as senior debt-like claims), practitioners often use overall equity volatility (derived from guideline public companies) as a direct proxy for common stock volatility in their simulations. This approximation is generally acceptable only when common equity represents the dominant share class (e.g., common constitutes 80%+ of TEV with minimal dilution) and leverage is negligible. In multi-tier or heavily diluted structures, however, common stock — as a junior residual claim — carries inherently higher volatility than total equity volatility, and directly substituting equity volatility risks systematically understating fair values (FVs) of options and incentive awards.

A naive fix might be to apply Merton's closed-form formula to invert $(S_0, \sigma_A)$ from observable equity data. However, Merton's closed form applies only to a single class of common equity modeled as a plain call option on assets. When the capital structure includes multi-class, multi-threshold, and step-vesting exotic awards (as in this engine), no closed-form solution exists. Numerical integration is the only tractable and exact approach.

To resolve this, our engine introduces an explicit **CCA calibration layer**: we calibrate the unobserved underlying parameters $(S_0, \sigma_A)$ jointly against the observable target TEV ($\text{TEV}_0$) and equity volatility ($\sigma_{eq}$), then apply Merton's leverage elasticity to derive exact class-specific volatilities. Unlike Monte Carlo simulation — which is stochastic and introduces random noise — **numerical integration is fully deterministic**, producing a smooth optimization landscape that enables stable, fast convergence. Per the AICPA 2019 Valuation Guide, class-specific volatilities should reflect the true economic risk of each security class, a requirement this calibration framework is designed to satisfy.

---

## 2. Trapezoidal Rule Pricing

### 2.1 Core Intuition: Numerical Integration as Pricing

The Black-Scholes closed-form solution establishes that an option's present value is the **discounted expected value** of the terminal payoff $f(S_T)$ under the risk-neutral measure:

$$V = e^{-rT} \, \mathbb{E}^{\mathbb{Q}}[f(S_T)] = e^{-rT} \int_0^{\infty} f(S_T) \cdot p(S_T) \, dS_T$$

where $p(S_T)$ is the log-normal probability density function (PDF). This integral can be computed directly via numerical evaluation using the **Trapezoidal Rule**:

$$\int_a^b g(x)\,dx \approx \sum_{i=0}^{N-1} \frac{g(x_i) + g(x_{i+1})}{2} \cdot (x_{i+1} - x_i)$$

### 2.2 Pricing Vanilla Options

Taking a European call option $f(S_T) = \max(S_T - K,\ 0)$ as an example:

```python
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import lognorm

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

# Step 1: Construct the underlying asset log-normal grid and PDF
mu    = np.log(S0) + (r - 0.5 * sigma**2) * T
s     = sigma * np.sqrt(T)
grid  = np.linspace(0, S0 * np.exp(6 * s), 10001)   # Underlying asset price range
pdf   = lognorm.pdf(grid, s=s, scale=np.exp(mu))     # Risk-neutral PDF

# Step 2: Compute terminal payoff vector
payoff = np.maximum(grid - K, 0.0)

# Step 3: Compute present value via trapezoidal integration
price = np.exp(-r * T) * trapezoid(payoff * pdf, grid)
print(f"Trapezoidal price = {price:.4f}")  # ~ 10.4506 (matches Black-Scholes closed-form)
```

In this project, this workflow is encapsulated inside `bs_price_and_delta_multiple()`:

```python
from black_scholes import bs_price_and_delta_multiple, call_option

vanilla_call = call_option(K=100.0)
prices, deltas = bs_price_and_delta_multiple(S0=100, T=1, r=0.05, q=0, sigma=0.20, payoffs={"call": vanilla_call})
price, delta = prices["call"], deltas["call"]
```

---

## 3. Extension to Exotic Stock-Based Compensation Options

Equity compensation plans (such as Employee Stock Options [ESOs] and Incentive Units [RSUs/PSUs]) introduce two additional conditions beyond standard call options:

| Condition | Mathematical Meaning | Code Parameter |
| :--- | :--- | :--- |
| **Strike Price** $K$ | The per-share amount paid by the employee upon exercise | `K` |
| **Hurdle / Threshold** $T_j$ | The underlying asset price must **exceed** this level for vesting to trigger | `thres` |
| **Vesting Percentage** $v_j$ | The fraction of units that convert into shares once the threshold is satisfied | `vest_pct` |

### 3.1 Payoff Function of Exotic Options

The `call_option()` factory function in this project unifies these conditions into:

$$f(S_T) = \max(S_T - K,\ 0) \times v(S_T)$$

where $v(S_T)$ is a **vesting function** that determines the effective vesting percentage based on asset price hurdle levels:

```
       v(S_T)
  1.00 ┤            ┌─────────────────  (Full Vesting)
  0.85 ┤      ┌─────┘
  0.00 ┤──────┘
       └──────┬──────┬────────────────> S_T
              K    thres
```

### 3.2 Three Option Archetypes in `portfolio.py`

```python
from black_scholes import call_option

# Type 1: Plain Vanilla Stock Option (Class 02-08)
# No threshold barrier; fully vested upon exercise
vanilla = call_option(K=50.0, thres=0.0, vest_pct=1.0, is_step=1)

# Type 2: Warrants (Class 11-20)
# Asset price must breach hurdle (thres = K + 50) to vest 85%
warrants = call_option(K=100.0, thres=150.0, vest_pct=0.85, is_step=1)
#   payoff(S_T) = max(S_T - 100, 0) * (0.85 if S_T > 150 else 0)

# Type 3: Incentive Units (Class 31-50)
# Two-tier step vesting: vests 50% above thres[0]=K+30, 100% above thres[1]=K+150
incentive = call_option(K=100.0, thres=[130.0, 250.0], vest_pct=[0.50, 1.00], is_step=1)
#   payoff(S_T) = max(S_T - 100, 0) * {0.0 if S_T<130 | 0.5 if 130<=S_T<250 | 1.0 if S_T>=250}
```

Due to the **step discontinuities** in the payoff functions, conventional finite differencing fails to provide stable derivatives. The trapezoidal rule only evaluates pointwise values of the discontinuous payoff $f(S_T)$ and **requires zero differentiation of the payoff function**, making it inherently ideal for such exotic structures.

---

## 4. Design Architecture

### 4.1 Performance Breakthrough: The Mutual Grid

Constructing separate integration grids for each option would require $50 \times N$ evaluations across 50 classes.
The key insight of this engine is that: **the PDF vector $p(S_T)$ and LRM weight $w_\Delta(S_T)$ depend solely on underlying asset parameters and are completely invariant to individual option contract specifications.**

Therefore, we construct a single shared "Mutual Grid" once; all 50 classes are evaluated across this identical $(S_T, p)$ pair, reducing computational complexity to $O(N + M)$:

```
make_mutual_grid(S0, T, r, q, sigma, N)
     │
     ├── st_grid [N+1]    Shared asset price nodes
     └── pdf    [N+1]    Shared PDF vector
              │
     ┌────────┼────────┬─────────┐
     │        │        │         │
  payoff_1  payoff_2  payoff_3  ... (50 classes)
     │        │        │
  f1*pdf   f2*pdf   f3*pdf   (Vector dot products: maximum SIMD/vectorization)
     │        │        │
  trapezoid  trapezoid  trapezoid (Trapezoidal integration)
     │        │        │
    FV_1     FV_2     FV_3
```

### 4.2 LRM Delta: Differentiating Without Touching the Payoff Function

The standard approach for computing Delta relies on finite difference approximations $\frac{V(S_0 + \varepsilon) - V(S_0 - \varepsilon)}{2\varepsilon}$, requiring two complete re-pricing runs.

The breakthrough of the Likelihood Ratio Method (LRM) is shifting differentiation with respect to $S_0$ from the payoff $f(S_T)$ onto the density $p(S_T)$:

$$\Delta = \frac{\partial V}{\partial S_0} = e^{-rT} \int_0^\infty f(S_T) \underbrace{\left[ p(S_T) \cdot \frac{z}{S_0 \sigma^2 T} \right]}_{w_\Delta(S_T)} dS_T$$

where $z = \ln(S_T/S_0) - (r - q - \frac{1}{2}\sigma^2)T$ is the centered log-return.

This means computing Delta merely requires swapping $f \cdot p$ with $f \cdot w_\Delta$, **requiring zero additional re-evaluations or bumps**. In code:

```python
# make_delta_weight() builds the w_delta vector of equal length to pdf; created once for all classes
w_delta = pdf * (z / (S0 * sigma**2 * T))   # Core LRM Delta weight vector

# Deltas for all 50 classes are evaluated across this identical w_delta
d[k] = disc * trapezoid(fn(grid) * w_delta, grid)   # Requires only 1 integration pass
```

### 4.3 Module Structure

```
opm/
├── black_scholes.py     # Core engine
│   ├── call_option()                  - Unified exotic option payoff factory
│   ├── make_mutual_grid()             - Constructs shared log-normal PDF grid
│   ├── make_delta_weight()            - LRM Delta integration kernel (w_delta)
│   ├── bs_price_and_delta_multiple()  - Public API: simultaneous FV and Delta calculation
│   ├── equity_moments()              - Calculates TEV0 and implied equity volatility
│   ├── calibrate_cca()               - Nelder-Mead joint calibration of S0 & sigma
│   ├── MarketParams                  - Dataclass: market inputs and grid settings
│   ├── CalibrationTargets            - Dataclass: target TEV and equity volatility
│   ├── calc_portfolio_allocations()  - Omega, Elasticity, and Specific Vol per class
│   └── run_opm_analysis()            - Comprehensive text report generation
│
├── portfolio.py         # Capital structure definitions
│   └── build_capital_structure() - Builds the 50-class equity compensation portfolio
│
├── generate_report.py   # HTML report generation
└── main.py              # Entry point: runs reports and opens browser automatically
```

---

## 5. CCA Calibration and Volatility Allocation

### 5.1 Calibration Objective

Under the Contingent Claim Analysis (CCA) framework, the unobserved underlying parameters $(S_0, \sigma_A)$ must be inverted from observable market inputs (target equity fair value $\text{TEV}_0^{\text{target}}$ and equity volatility $\sigma_{eq}^{\text{target}}$). `calibrate_cca()` uses Nelder-Mead simplex optimization to minimize the following loss function:

$$\mathcal{L}(S_0, \sigma_A) = \underbrace{\left(\frac{\text{TEV}(S_0, \sigma_A) - \text{TEV}_0^{\text{target}}}{\text{TEV}_0^{\text{target}}}\right)^2}_{\text{Market Value Error}} + \underbrace{\left(\frac{\sigma_{eq}(S_0, \sigma_A) - \sigma_{eq}^{\text{target}}}{\sigma_{eq}^{\text{target}}}\right)^2}_{\text{Volatility Error}}$$

where $\sigma_{eq}$ is computed via the **Merton leverage-elasticity formula**, using the portfolio-level Delta from numerical integration:

$$\sigma_{eq}(S_0, \sigma_A) = \sigma_A \cdot \frac{S_0 \cdot \Delta_{\text{total}}}{\text{TEV}_0}$$

where $\Delta_{\text{total}} = \sum_i N_i \Delta_i$ is the aggregate portfolio Delta computed on the shared mutual grid. This is numerically exact for any payoff structure and directly consistent with the Itô-chain-rule derivation of equity volatility.

To accelerate Nelder-Mead convergence, the solver uses a gross-up initial guess for $S_0$; $\sigma_A$ is initialized directly to the target equity volatility:
- $S_0^{\text{guess}} = \frac{\text{TEV}_0 + \text{Total Proceeds}}{\text{Total Shares}}$ (Gross-up proxy)
- $\sigma_A^{\text{guess}} = \sigma_{eq}^{\text{target}}$ (Direct initialization; leverage adjustment is left to the optimizer)

### 5.2 Class-Specific Volatility Allocation (Option B)

Post-calibration, the engine allocates the aggregate $\sigma_{eq}^{\text{target}}$ to individual classes via **relative leverage projection**:

$$\sigma_{i,\text{alloc}} = \sigma_{eq}^{\text{target}} \times \frac{\text{TEV}_0 \cdot \Delta_i}{V_i \cdot \Delta_{\text{total}}}$$

**Why does this guarantee that the weighted sum equals the target?**

$$\sum_i \frac{N_i V_i}{\text{TEV}_0} \cdot \sigma_{i,\text{alloc}} = \sigma_{eq}^{\text{target}} \cdot \frac{\sum_i N_i \Delta_i}{\Delta_{\text{total}}} = \sigma_{eq}^{\text{target}} \cdot 1 = \sigma_{eq}^{\text{target}}$$

## 6. Complete Execution Example

The following is a complete end-to-end pricing and calibration example with **10 share classes**, using the same market inputs as `generate_report.py` (50/50 Scenario 4). Classes 2–8 are vanilla Stock Options; Classes 9–10 are Early Vest Options with a vesting threshold.

```python
from black_scholes import call_option, MarketParams, CalibrationTargets, run_opm_analysis

# --- 1. Define Capital Structure (10 Share Classes, 50/50 Common/Other) ---
portfolio = {
    "Class 01 Common Stock": {
        "params": {"K": 0.0, "thres": 0.0, "vest_pct": 1.0, "is_step": 1},
        "th_str": "-", "shares": 10.0, "fn": call_option(K=0.0),
    },
    "Class 02 Stock Option": {
        "params": {"K": 10.0, "thres": 0.0, "vest_pct": 1.0, "is_step": 1},
        "th_str": "-", "shares": 1.1111, "fn": call_option(K=10.0),
    },
    "Class 03 Stock Option": {
        "params": {"K": 15.0, "thres": 0.0, "vest_pct": 1.0, "is_step": 1},
        "th_str": "-", "shares": 1.1111, "fn": call_option(K=15.0),
    },
    "Class 04 Stock Option": {
        "params": {"K": 20.0, "thres": 0.0, "vest_pct": 1.0, "is_step": 1},
        "th_str": "-", "shares": 1.1111, "fn": call_option(K=20.0),
    },
    "Class 05 Stock Option": {
        "params": {"K": 25.0, "thres": 0.0, "vest_pct": 1.0, "is_step": 1},
        "th_str": "-", "shares": 1.1111, "fn": call_option(K=25.0),
    },
    "Class 06 Stock Option": {
        "params": {"K": 30.0, "thres": 0.0, "vest_pct": 1.0, "is_step": 1},
        "th_str": "-", "shares": 1.1111, "fn": call_option(K=30.0),
    },
    "Class 07 Stock Option": {
        "params": {"K": 35.0, "thres": 0.0, "vest_pct": 1.0, "is_step": 1},
        "th_str": "-", "shares": 1.1111, "fn": call_option(K=35.0),
    },
    "Class 08 Stock Option": {
        "params": {"K": 40.0, "thres": 0.0, "vest_pct": 1.0, "is_step": 1},
        "th_str": "-", "shares": 1.1111, "fn": call_option(K=40.0),
    },
    "Class 09 Early Vest Option": {
        # Vests 50% once asset price exceeds the threshold of 25
        "params": {"K": 45.0, "thres": 25.0, "vest_pct": 0.5, "is_step": 1},
        "th_str": "25", "shares": 1.1111, "fn": call_option(K=45.0, thres=25.0, vest_pct=0.5),
    },
    "Class 10 Early Vest Option": {
        # Vests 50% once asset price exceeds the threshold of 30
        "params": {"K": 50.0, "thres": 30.0, "vest_pct": 0.5, "is_step": 1},
        "th_str": "30", "shares": 1.1111, "fn": call_option(K=50.0, thres=30.0, vest_pct=0.5),
    },
}

# --- 2. Set Market Parameters and Calibration Targets ---
# Inputs match generate_report.py Scenario 4 (50/50 structure)
params  = MarketParams(S0=80, sigma=0.85, T=2.0, r=0.05, q=0.0, N=10000)
targets = CalibrationTargets(tev0=850.0, voleq=0.80)  # Target equity $850, vol 80%

# --- 3. Execute Calibration and Generate Report ---
report = run_opm_analysis(portfolio, params, targets)
print(report)
```

**Example Output:**

```
===================================================================================================================================================================
Summary   : Total Equity = $850.00 | Equity Vol = 80.00% | Implied S0 = $54.4445 | T = 2.0y | r = 5.0%
Dilution  : No Dilution: $85.00 | By OPM: $54.44 | By Vested Shares: $46.03 | By Outstanding Shares: $42.50
Diagnosis : Converged in 38 iters (38.1 ms) | Batch Run Time = 0.87 ms
===================================================================================================================================================================
Class                          Shares Shares (%) Strike K         Thres Vested Shares  Delta (dV/dS0)   Omega (Ω)  Elasticity Specific Vol %  FV (PS $)   Total ($)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
Class 01 Common Stock           10.00      50.00     0.00             -         10.00          1.0000      55.74%      0.8702          69.62    54.4446      544.45
Class 02 Stock Option            1.11       5.56    10.00             -          1.11          0.9897       6.13%      1.0268          82.15    45.6646       50.74
Class 03 Stock Option            1.11       5.56    15.00             -          1.11          0.9715       6.02%      1.1024          88.19    41.7512       46.39
Class 04 Stock Option            1.11       5.56    20.00             -          1.11          0.9464       5.86%      1.1727          93.82    38.2349       42.48
Class 05 Stock Option            1.11       5.56    25.00             -          1.11          0.9169       5.68%      1.2378          99.03    35.0941       38.99
Class 06 Stock Option            1.11       5.56    30.00             -          1.11          0.8848       5.48%      1.2981         103.85    32.2916       35.88
Class 07 Stock Option            1.11       5.56    35.00             -          1.11          0.8514       5.27%      1.3542         108.34    29.7880       33.10
Class 08 Stock Option            1.11       5.56    40.00             -          1.11          0.8178       5.06%      1.4066         112.53    27.5462       30.61
Class 09 Early Vest Option       1.11       5.56    45.00            25          0.36          0.3922       2.43%      1.4556         116.45    12.7666       14.19
Class 10 Early Vest Option       1.11       5.56    50.00            30          0.33          0.3759       2.33%      1.5017         120.14    11.8601       13.18
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
TOTAL CAPITAL STRUCTURE         20.00     100.00                                18.47                     100.00%                      80.00                 850.00
===================================================================================================================================================================
```

**Note:** The equity fair-value weighted sum of `Specific Vol %` across all classes equals exactly the target input of **80.00%**, satisfying audit trail and compliance verification.

---


## 7. Dependencies & Setup

This project uses Python `>= 3.12` and standard numerical packages (`numpy`, `scipy`).

### Installation

```bash
pip install -r requirements.txt
```

### Quick Run

```bash
# Generate comprehensive HTML report across 5 capital structure scenarios with one click and auto-open in browser
python main.py
```
