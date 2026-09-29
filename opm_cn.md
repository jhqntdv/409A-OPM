# Capital Structure OPM - Trapezoidal Pricing Engine & LRM Greeks

<div align="right">
  <strong>Language / 語言:</strong>
  <a href="README.md">English</a> | <b>繁體中文</b>
</div>

> 使用梯形積分法 (Trapezoidal Rule) 與似然比方法 (Likelihood Ratio Method) 對複雜企業資本結構進行批次估值與風險分析的 Python 引擎。

---

## 目錄 (Table of Contents)

1. [簡介 (Introduction)](#1-簡介-introduction)
   - [1.1 OPM 資本結構樹 (OPM Capital Structure Tree)](#11-opm-資本結構樹-opm-capital-structure-tree)
   - [1.2 CCA (OPM) for private company practice](#12-cca-opm-for-private-company-practice)
2. [梯形積分定價法 (Trapezoidal Rule Pricing)](#2-梯形積分定價法)
3. [推廣至股權薪酬的異型期權](#3-推廣至股權薪酬的異型期權)
4. [設計架構 (Design Architecture)](#4-設計架構)
5. [CCA 校準與波動率分配](#5-cca-校準與波動率分配)
6. [完整執行範例](#6-完整執行範例)

---

## 1. 簡介 (Introduction)

在非上市公司與複雜股權結構的估值中，選擇權定價模型 (Option Pricing Model, OPM) 與或有請求權分析 (Contingent Claim Analysis, CCA) 是國際財務報告準則 (IFRS 2) 與美國會計準則 (ASC 718 / AICPA 估值指南) 中針對多重權益類別分配公平價值的核心方法。

本專案提供一套高效能數值積分與似然比方法 (LRM) 引擎，能在單一共享網格上同時完成數十種不同條款期權類別的現值與風險敏感度 (Delta) 批次計算。

### 1.1 OPM 資本結構樹 (OPM Capital Structure Tree)

本專案 `portfolio.py` 實作的 50 類別資本結構層級樹狀關係如下：

```
Total Enterprise / Equity Value (TEV, S0)
├── Class 01: Common Stock (普通股)
│   ├── 股數權重: 10% - 100% (依情境配置，預設 50% / 10.0 股)
│   ├── 履約價: K = 0.0 (無履約門檻)
│   └── 歸屬條件: 100% 完全歸屬 (vest_pct = 1.0)
│
├── Class 02 - 08: Stock Options (員工認股權 / 香草期權)
│   ├── 類別數量: 7 個類別 (每類別 0.50 股，共 3.5 股)
│   ├── 履約價區間: K in [10.0, 40.0]
│   └── 歸屬條件: 100% 完全歸屬 (無股價門檻，thres = 0.0)
│
├── Class 09 - 10: Early Vest Options (提早歸屬期權)
│   ├── 類別數量: 2 個類別 (每類別 0.50 股，共 1.0 股)
│   ├── 履約價區間: K in [45.0, 50.0]
│   ├── 門檻條件: thres = max(K - 20.0, 0.1)
│   └── 歸屬比例: 達到門檻後歸屬 50% (vest_pct = 0.50)
│
├── Class 11 - 20: Warrants (認股權證 / 附門檻期權)
│   ├── 類別數量: 10 個類別 (每類別 0.20 股，共 2.0 股)
│   ├── 履約價區間: K in [55.0, 100.0]
│   ├── 門檻條件: thres = K + 50.0
│   └── 歸屬比例: 達到門檻後歸屬 85% (vest_pct = 0.85)
│
├── Class 21 - 25: Mimic Linear (擬真線性歸屬)
│   ├── 類別數量: 5 個類別 (每類別 0.15 股，共 0.75 股)
│   ├── 履約價區間: K in [105.0, 125.0]
│   ├── 門檻區間: thres = [K + 10.0 .. K + 110.0] (100 階梯近似)
│   └── 歸屬比例: 線性階梯歸屬 (0.01 至 1.0)
│
├── Class 26 - 30: True Linear (真實連續線性歸屬)
│   ├── 類別數量: 5 個類別 (每類別 0.15 股，共 0.75 股)
│   ├── 履約價區間: K in [130.0, 150.0]
│   ├── 門檻區間: thres = [K + 10.0 -> K + 110.0] (連續線性)
│   └── 歸屬比例: 連續線性歸屬 (0.0 至 1.0)
│
└── Class 31 - 50: Incentive Units (管理層激勵股份單位)
    ├── 類別數量: 20 個類別 (每類別 0.10 股，共 2.0 股)
    ├── 履約價區間: K in [155.0, 250.0]
    ├── 雙階梯門檻: thres = [K + 30.0, K + 150.0]
    └── 階梯歸屬: 超過 K + 30 歸屬 50%，超過 K + 150 歸屬 100%
```

### 1.2 CCA (OPM) for private company practice

在未上市公司股權估值實務（如 409A 與 IFRS 2 規範）中，估值專家通常假設全公司總股權價值（TEV）服從幾何布朗運動，並透過 OPM 折點分析或蒙地卡羅模擬執行瀑布流分配（Waterfall Allocation）。實務上常將可比公眾公司的「總股權波動率（Equity Volatility）」直接作為普通股波動率代理使用。此做法僅在普通股占 TEV 比例較高（例如 80% 以上）、稀釋程度低且槓桿微弱時，誤差尚在可接受範圍。一旦資本結構含多層次或高稀釋度的期權與特別股，普通股作為殘值請求權的真實波動率將高於總股權波動率，直接套用將導致系統性低估各類別公平價值（Fair Value）。

有人或許認為直接套用 Merton 封閉解即可反推 $(S_0, \sigma_A)$，然而 Merton 封閉解僅適用於單一普通股類別（視為普通看漲期權）。本專案的資本結構包含多類別、多層門檻、多段階梯歸屬的異型期權，**不存在封閉解**，唯有數值積分方能精確求解。

為此，本引擎引入嚴謹的 **CCA 反推校準層**：同時對目標 $\text{TEV}_0$ 與 $\sigma_{eq}$ 進行聯合優化，校準底層未觀測參數 $(S_0, \sigma_A)$，再透過 Merton 槓桿彈性公式嚴謹派生各類別特定波動率。相較於蒙地卡羅模擬，**數值積分為確定性計算，無隨機噪聲**，損失函數曲面光滑，校準過程穩定且高效。根據 AICPA 2019 估值指南，各類別估值應採用反映其真實經濟風險的特定波動率，本框架正是為滿足此合規要求而設計。

---

## 2. 梯形積分定價法

### 2.1 核心直觀：數值積分即定價

Black-Scholes 的封閉解告訴我們，期權現值是到期支付 $f(S_T)$ 在風險中性測度下的**期望折現值**：

$$V = e^{-rT} \, \mathbb{E}^{\mathbb{Q}}[f(S_T)] = e^{-rT} \int_0^{\infty} f(S_T) \cdot p(S_T) \, dS_T$$

其中 $p(S_T)$ 是對數常態分佈的 PDF。這個積分可以用**梯形法則 (Trapezoidal Rule)** 直接數值計算：

$$\int_a^b g(x)\,dx \approx \sum_{i=0}^{N-1} \frac{g(x_i) + g(x_{i+1})}{2} \cdot (x_{i+1} - x_i)$$

### 2.2 對普通香草期權定價

以歐式看漲期權 $f(S_T) = \max(S_T - K,\ 0)$ 為例：

```python
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import lognorm

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

# Step 1: 建立底層資產的對數常態網格與 PDF
mu    = np.log(S0) + (r - 0.5 * sigma**2) * T
s     = sigma * np.sqrt(T)
grid  = np.linspace(0, S0 * np.exp(6 * s), 10001)   # 底層資產價格區間
pdf   = lognorm.pdf(grid, s=s, scale=np.exp(mu))     # 風險中性 PDF

# Step 2: 計算到期支付向量
payoff = np.maximum(grid - K, 0.0)

# Step 3: 梯形積分求現值
price = np.exp(-r * T) * trapezoid(payoff * pdf, grid)
print(f"Trapezoidal price = {price:.4f}")  # ~ 10.4506 (與 Black-Scholes 封閉解吻合)
```

在本專案中，這個流程被封裝於 `bs_price_and_delta_multiple()`：

```python
from black_scholes import bs_price_and_delta_multiple, call_option

vanilla_call = call_option(K=100.0)
prices, deltas = bs_price_and_delta_multiple(S0=100, T=1, r=0.05, q=0, sigma=0.20, payoffs={"call": vanilla_call})
price, delta = prices["call"], deltas["call"]
```

---

## 3. 推廣至股權薪酬的異型期權

股票薪酬計畫（如員工股票期權 ESO、激勵單位 RSU/PSU）在標準看漲期權之上額外增加了兩層條件：

| 條件 | 數學含義 | 程式碼參數 |
| :--- | :--- | :--- |
| **履約價 Strike** $K$ | 員工執行期權時須支付的每股金額 | `K` |
| **門檻 Threshold** $T_j$ | 底層資產價格須**超過**此水位，歸屬才會啟動 | `thres` |
| **歸屬比例 Vesting** $v_j$ | 達到門檻後可轉換的期權比例 | `vest_pct` |

### 3.1 異型期權的到期支付函數

本專案的 `call_option()` 函式將上述條件統一為：

$$f(S_T) = \max(S_T - K,\ 0) \times v(S_T)$$

其中 $v(S_T)$ 是一個**歸屬函數 (Vesting Function)**，根據門檻水位決定當前的歸屬比例：

```
       v(S_T)
  1.00 ┤            ┌─────────────────  (完全歸屬)
  0.85 ┤      ┌─────┘
  0.00 ┤──────┘
       └──────┬──────┬────────────────> S_T
             K    thres
```

### 3.2 `portfolio.py` 中的三種期權類型

```python
from black_scholes import call_option

# 類型 1: 普通股票期權 (Class 02-08)
# 無門檻限制，行使即全額歸屬
vanilla = call_option(K=50.0, thres=0.0, vest_pct=1.0, is_step=1)

# 類型 2: 認股權證 (Class 11-20)
# 資產價格須超過門檻 (thres = K + 50) 才可歸屬 85%
warrants = call_option(K=100.0, thres=150.0, vest_pct=0.85, is_step=1)
#   payoff(S_T) = max(S_T - 100, 0) * (0.85 if S_T > 150 else 0)

# 類型 3: 激勵股份單位 (Class 31-50)
# 兩段式歸屬：超過 thres[0]=K+30 歸屬 50%，超過 thres[1]=K+150 歸屬 100%
incentive = call_option(K=100.0, thres=[130.0, 250.0], vest_pct=[0.50, 1.00], is_step=1)
#   payoff(S_T) = max(S_T - 100, 0) * {0.0 if S_T<130 | 0.5 if 130<=S_T<250 | 1.0 if S_T>=250}
```

由於支付函數的**階梯不連續性**，傳統有限差分無法穩定求導。梯形積分法只需評估不連續的 $f(S_T)$ 的函數值，**完全無需對支付函數微分**，因此天然適合此類異型期權。

---

## 4. 設計架構

### 4.1 相互網格 (Mutual Grid) 的效能突破

若對每個期權各自建立積分網格，50 個類別需要 $50 \times N$ 次計算。
本引擎的關鍵洞察是：**PDF 向量 $p(S_T)$ 與 LRM 權重 $w_\Delta(S_T)$ 完全由底層資產參數決定，與個別期權合約無關。**

因此，我們只需建立一次共享的「相互網格 (Mutual Grid)」，所有 50 個類別皆在同一組 $(S_T, p)$ 上計算，複雜度降至 $O(N + M)$：

```
make_mutual_grid(S0, T, r, q, sigma, N)
     │
     ├── st_grid [N+1]    共享的資產價格節點
     └── pdf    [N+1]    共享的 PDF 向量
              │
     ┌────────┼────────┬─────────┐
     │        │        │         │
  payoff_1  payoff_2  payoff_3  ... (50 個類別)
     │        │        │
  f1*pdf   f2*pdf   f3*pdf   (向量點積: 極致向量化)
     │        │        │
  trapezoid  trapezoid  trapezoid (梯形積分)
     │        │        │
    FV_1     FV_2     FV_3
```

### 4.2 LRM Delta: 不對支付函數求導

計算 Delta 的傳統方式是有限差分 $\frac{V(S_0 + \varepsilon) - V(S_0 - \varepsilon)}{2\varepsilon}$，需要兩次完整重新定價。

LRM 的突破是將對 $S_0$ 的微分由 $f(S_T)$ 轉移至 $p(S_T)$：

$$\Delta = \frac{\partial V}{\partial S_0} = e^{-rT} \int_0^\infty f(S_T) \underbrace{\left[ p(S_T) \cdot \frac{z}{S_0 \sigma^2 T} \right]}_{w_\Delta(S_T)} dS_T$$

其中 $z = \ln(S_T/S_0) - (r - q - \frac{1}{2}\sigma^2)T$ 為中心化對數回報。

這意味著 Delta 的計算只需把 $f \cdot p$ 換成 $f \cdot w_\Delta$，**不需要任何額外的重新定價**。在程式碼中：

```python
# make_delta_weight() 建立的 w_delta 向量與 pdf 等長，建立一次即可供所有類別使用
w_delta = pdf * (z / (S0 * sigma**2 * T))   # LRM Delta 核心權重

# 50 個類別的 Delta 全部可以用同一組 w_delta 計算
d[k] = disc * trapezoid(fn(grid) * w_delta, grid)   # 只需 1 次積分
```

### 4.3 模組結構

```
opm/
├── black_scholes.py     # 核心引擎
│   ├── call_option()                  - 統一異型期權支付工廠
│   ├── make_mutual_grid()             - 建立共享對數常態 PDF 網格
│   ├── make_delta_weight()            - LRM Delta 積分核 (w_delta)
│   ├── bs_price_and_delta_multiple()  - 公開 API：同時計算 FV 與 Delta
│   ├── equity_moments()              - 計算 TEV0 及隱含股權波動率
│   ├── calibrate_cca()               - Nelder-Mead 校準 S0 & sigma
│   ├── MarketParams                  - Dataclass：市場輸入與網格設定
│   ├── CalibrationTargets            - Dataclass：目標 TEV 與股權波動率
│   ├── calc_portfolio_allocations()  - 計算各類別 Omega、Elasticity、Specific Vol
│   └── run_opm_analysis()            - 完整報表字串輸出
│
├── portfolio.py         # 資本結構定義
│   └── build_capital_structure() - 建立 50 類別的股票薪酬投資組合
│
├── generate_report.py   # HTML 報表生成
└── main.py              # 入口點：呼叫報表、自動開啟瀏覽器
```

---

## 5. CCA 校準與波動率分配

### 5.1 校準目標

在 Contingent Claim Analysis (CCA) 框架下，底層資產的**未觀測參數** $(S_0, \sigma_A)$ 必須從已觀測的市場資料（股權公平價值 $\text{TEV}_0$、股權波動率 $\sigma_{eq}$）反推。`calibrate_cca()` 使用 Nelder-Mead 最優化，最小化以下損失函數：

$$\mathcal{L}(S_0, \sigma_A) = \underbrace{\left(\frac{\text{TEV}(S_0, \sigma_A) - \text{TEV}_0^{\text{target}}}{\text{TEV}_0^{\text{target}}}\right)^2}_{\text{市值誤差}} + \underbrace{\left(\frac{\sigma_{eq}(S_0, \sigma_A) - \sigma_{eq}^{\text{target}}}{\sigma_{eq}^{\text{target}}}\right)^2}_{\text{波動率誤差}}$$

其中 $\sigma_{eq}$ 採用 **Merton 槓桿彈性公式**，以數值積分所得的投資組合 Delta 計算：

$$\sigma_{eq}(S_0, \sigma_A) = \sigma_A \cdot \frac{S_0 \cdot \Delta_{\text{total}}}{\text{TEV}_0}$$

其中 $\Delta_{\text{total}} = \sum_i N_i \Delta_i$ 為在共享相互網格上計算的組合總 Delta。此公式對任意支付結構均精確成立，並直接對應 Itô 連鎖法則下的股權波動率推導。

為加速 Nelder-Mead 收斂，本演算法以還原基準價格作為 $S_0$ 的初始猜測；$\sigma_A$ 則直接以目標股權波動率初始化：
- $S_0^{\text{guess}} = \frac{\text{TEV}_0 + \text{總行使價收益}}{\text{總股數}}$ (還原基準價格)
- $\sigma_A^{\text{guess}} = \sigma_{eq}^{\text{target}}$ (直接初始化；槓桿調整交由優化器自行求解)

### 5.2 各類別特定波動率的分配 (Option B)

校準後，系統以「**相對槓桿投影**」將總體 $\sigma_{eq}^{\text{target}}$ 分配至各類別：

$$\sigma_{i,\text{alloc}} = \sigma_{eq}^{\text{target}} \times \frac{\text{TEV}_0 \cdot \Delta_i}{V_i \cdot \Delta_{\text{total}}}$$

**為何這樣做保證加總等於目標？**

$$\sum_i \frac{N_i V_i}{\text{TEV}_0} \cdot \sigma_{i,\text{alloc}} = \sigma_{eq}^{\text{target}} \cdot \frac{\sum_i N_i \Delta_i}{\Delta_{\text{total}}} = \sigma_{eq}^{\text{target}} \cdot 1 = \sigma_{eq}^{\text{target}}$$

## 6. 完整執行範例

以下是一個包含 **10 種股份類別**的完整定價與校準範例，輸入參數與 `generate_report.py` 的情境 4（50/50 結構）完全一致。第 2–8 類為普通股票期權；第 9–10 類為附有歸屬條件的「提前歸屬期權 (Early Vest Option)」。

```python
from black_scholes import call_option, MarketParams, CalibrationTargets, run_opm_analysis

# --- 1. 定義資本結構（10 個股份類別，50/50 普通股 / 其他）---
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
        # 資產價格超過 25 時歸屬 50%
        "params": {"K": 45.0, "thres": 25.0, "vest_pct": 0.5, "is_step": 1},
        "th_str": "25", "shares": 1.1111, "fn": call_option(K=45.0, thres=25.0, vest_pct=0.5),
    },
    "Class 10 Early Vest Option": {
        # 資產價格超過 30 時歸屬 50%
        "params": {"K": 50.0, "thres": 30.0, "vest_pct": 0.5, "is_step": 1},
        "th_str": "30", "shares": 1.1111, "fn": call_option(K=50.0, thres=30.0, vest_pct=0.5),
    },
}

# --- 2. 設定市場參數與校準目標 ---
# 輸入與 generate_report.py 情境 4（50/50 結構）完全一致
params  = MarketParams(S0=80, sigma=0.85, T=2.0, r=0.05, q=0.0, N=10000)
targets = CalibrationTargets(tev0=850.0, voleq=0.80)  # 目標股權市值 $850，波動率 80%

# --- 3. 執行校準與報表 ---
report = run_opm_analysis(portfolio, params, targets)
print(report)
```

**輸出範例：**

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

**注意：** 所有類別的 `Specific Vol %` 以股權市值加權後的總和，精確等於輸入的目標值 **80.00%**，可供財務審計直接使用。

---


## 7. 相依套件與安裝 (Dependencies & Setup)

本專案使用 Python `>= 3.12` 與標準科學計算套件 (`numpy`, `scipy`)。

### 安裝

```bash
pip install -r requirements.txt
```

### 快速執行 (Quick Run)

```bash
# 一鍵產生包含 5 種資本結構情境的完整 HTML 報表，並自動在瀏覽器開啟
python main.py
```

