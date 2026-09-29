"""Capital Structure & Portfolio Definitions for OPM Analysis."""

import numpy as np
from black_scholes import call_option


def build_capital_structure(common_ratio: float = 0.50) -> dict:
    """Builds unified 50-class capital structure (Common fixed at 10.0 shs)."""
    strikes = list(range(10, 255, 5))
    other_classes = {}
    for idx, k_raw in enumerate(strikes, 2):
        k = float(k_raw)
        if idx <= 8:
            name, params, th_str, base_sh = f"Class {idx:02d} Stock Option", {"K": k, "thres": 0.0, "vest_pct": 1.0, "is_step": 1}, "-", 0.50
        elif 9 <= idx <= 10:
            th = max(k - 20.0, 0.1)
            name, params, th_str, base_sh = f"Class {idx:02d} Early Vest Option", {"K": k, "thres": th, "vest_pct": 0.50, "is_step": 1}, f"{th:.0f}", 0.50
        elif 11 <= idx <= 20:
            th = k + 50.0
            name, params, th_str, base_sh = f"Class {idx:02d} Warrants", {"K": k, "thres": th, "vest_pct": 0.85, "is_step": 1}, f"{th:.0f}", 0.20
        elif 21 <= idx <= 25:
            th_start, th_end = k + 10.0, k + 110.0
            thres_arr = np.linspace(th_start, th_end, 100).tolist()
            vest_arr = np.linspace(0.01, 1.0, 100).tolist()
            name, params, th_str, base_sh = f"Class {idx:02d} Mimic Linear", {"K": k, "thres": thres_arr, "vest_pct": vest_arr, "is_step": 1}, f"[{th_start:.0f}..{th_end:.0f}]", 0.15
        elif 26 <= idx <= 30:
            th_start, th_end = k + 10.0, k + 110.0
            name, params, th_str, base_sh = f"Class {idx:02d} True Linear", {"K": k, "thres": [th_start, th_end], "vest_pct": [0.0, 1.0], "is_step": 0}, f"[{th_start:.0f}->{th_end:.0f}]", 0.15
        else:
            th1, th2 = k + 30.0, k + 150.0
            name, params, th_str, base_sh = f"Class {idx:02d} Incentive Unit", {"K": k, "thres": [th1, th2], "vest_pct": [0.50, 1.00], "is_step": 1}, f"[{th1:.0f},{th2:.0f}]", 0.10
        other_classes[name] = {"params": params, "th_str": th_str, "base_shares": base_sh}

    sh_common = 10.0
    tot_base_other = sum(item["base_shares"] for item in other_classes.values())
    
    if common_ratio >= 1.0:
        multiplier = 0.0
    else:
        target_tot_other = sh_common * (1.0 - common_ratio) / common_ratio
        multiplier = target_tot_other / tot_base_other if tot_base_other > 0 else 0.0

    for item in other_classes.values():
        item["shares"] = item["base_shares"] * multiplier
        item["fn"] = call_option(**item["params"])
        del item["base_shares"]

    portfolio = {
        "Class 01 Common Stock": {
            "params": {"K": 0.0, "thres": 0.0, "vest_pct": 1.0, "is_step": 1},
            "th_str": "-", "shares": sh_common, "fn": call_option(K=0.0, thres=0.0, vest_pct=1.0),
        }
    }
    portfolio.update(other_classes)
    return portfolio

def build_capital_structure_100_0() -> dict:
    return build_capital_structure(common_ratio=1.0)

def build_capital_structure_90_10() -> dict:
    return build_capital_structure(common_ratio=0.90)

def build_capital_structure_75_25() -> dict:
    return build_capital_structure(common_ratio=0.75)

def build_capital_structure_50_50() -> dict:
    return build_capital_structure(common_ratio=0.50)

def build_capital_structure_25_75() -> dict:
    return build_capital_structure(common_ratio=0.25)