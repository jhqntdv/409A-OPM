"""Capital Structure & Portfolio Definitions for OPM Analysis."""

import numpy as np
from black_scholes import call_option


def build_capital_structure(common_ratio: float = 0.50) -> dict:
    """Builds unified 50-class capital structure (Common fixed at 10.0 shs)."""
    other_classes = {}

    # 3 PRSUs (K=0.0, high threshold)
    prsu_thresholds = {2: 30.0, 3: 50.0, 4: 80.0}
    for idx in range(2, 5):
        name = f"Class {idx:02d} PRSUs"
        params = {"K": 0.0, "thres": prsu_thresholds[idx], "vest_pct": 1.0, "is_step": 1}
        other_classes[name] = {"params": params, "th_str": f"{prsu_thresholds[idx]:.0f} (100%)", "base_shares": 0.15}

    # 4 Warrants (K=0.01)
    for idx in range(5, 9):
        name = f"Class {idx:02d} Warrants"
        params = {"K": 0.01, "thres": 0.0, "vest_pct": 1.0, "is_step": 1}
        other_classes[name] = {"params": params, "th_str": "-", "base_shares": 0.50}

    # 12 Stock Options (K = 2.0 to 13.0)
    for idx in range(9, 21):
        k = float(idx - 7)
        name = f"Class {idx:02d} Stock Options"
        params = {"K": k, "thres": 0.0, "vest_pct": 1.0, "is_step": 1}
        other_classes[name] = {"params": params, "th_str": "-", "base_shares": 0.50}

    # 20 PIUs (K = 18.0 to 56.0, step 2.0)
    for idx in range(21, 41):
        k = 18.0 + (idx - 21) * 2.0
        is_step = 1 if idx % 2 == 0 else 0
        name_suffix = " (Linear)" if is_step == 0 else ""
        name = f"Class {idx:02d} PIUs{name_suffix}"
        
        if idx >= 36:
            # 5 tranches
            th1 = k - 2.0 if k - 2.0 > 0 else 0.5
            th2, th3, th4, th5 = k + 2.0, k + 6.0, k + 10.0, k + 14.0
            params = {"K": k, "thres": [th1, th2, th3, th4, th5], "vest_pct": [0.20, 0.40, 0.60, 0.80, 1.0], "is_step": is_step}
            th_str = f"[{th1:.0f}...{th5:.0f}]"
        else:
            # 3 tranches
            th1, th2, th3 = k - 2.0 if k - 2.0 > 0 else 0.5, k + 5.0, k + 10.0
            params = {"K": k, "thres": [th1, th2, th3], "vest_pct": [0.33, 0.66, 1.0], "is_step": is_step}
            th_str = f"[{th1:.0f},{th2:.0f},{th3:.0f}]"
            
        other_classes[name] = {"params": params, "th_str": th_str, "base_shares": 0.20}

    # 10 Super PIUs (K = 60.0 to 150.0)
    for idx in range(41, 51):
        k = 60.0 + (idx - 41) * 10.0
        multiple = 7 + (idx - 41) % 4 # 7x, 8x, 9x, 10x
        th = float(multiple * 10) 
        if th < k: th = k + 20.0
        name = f"Class {idx:02d} Super PIUs"
        params = {"K": k, "thres": th, "vest_pct": 1.0, "is_step": 1}
        other_classes[name] = {"params": params, "th_str": f"{th:.0f} (100%)", "base_shares": 0.10}

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

def build_capital_structure_with_pref_100(common_ratio: float = 0.50) -> dict:
    from black_scholes import preferred_convertible
    p = build_capital_structure(common_ratio)
    new_p = {
        "Class 00 Series A Preferred": {
            "params": {"K": 0.0},
            "th_str": "-",
            "shares": 5.0,
            "fn": preferred_convertible(curr_face=20.0, conv_price=20.0, issue_face=20.0, pik=0.08, lp_multiple=1.0, T=2.0, r=0.05, rky=0.15, freq=2)
        }
    }
    new_p.update(p)
    return new_p

def build_capital_structure_pref_100_0() -> dict:
    return build_capital_structure_with_pref_100(1.0)

def build_capital_structure_pref_90_10() -> dict:
    return build_capital_structure_with_pref_100(0.90)

def build_capital_structure_pref_75_25() -> dict:
    return build_capital_structure_with_pref_100(0.75)

def build_capital_structure_pref_50_50() -> dict:
    return build_capital_structure_with_pref_100(0.50)

def build_capital_structure_pref_25_75() -> dict:
    return build_capital_structure_with_pref_100(0.25)

def build_simple_preferred_common_only(pref_face: float = 550.0) -> dict:
    from black_scholes import preferred_convertible, call_option
    return {
        "Class 00 Series A Preferred": {
            "params": {"K": 0.0},
            "th_str": "-",
            "shares": pref_face / 20.0,
            "fn": preferred_convertible(curr_face=20.0, conv_price=20.0, issue_face=20.0, pik=0.08, lp_multiple=1.0, T=2.0, r=0.05, rky=0.15, freq=2)
        },
        "Class 01 Common Stock": {
            "params": {"K": 0.0, "thres": 0.0, "vest_pct": 1.0, "is_step": 1},
            "th_str": "-", 
            "shares": 10.0, 
            "fn": call_option(K=0.0, thres=0.0, vest_pct=1.0),
        }
    }

def build_simple_pref_common_550() -> dict:
    return build_simple_preferred_common_only(550.0)

def build_simple_pref_common_250() -> dict:
    return build_simple_preferred_common_only(250.0)