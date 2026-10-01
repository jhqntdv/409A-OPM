"""Generate interactive Capital Structure OPM (CCA) multi-scenario report."""

import html
import re
from black_scholes import MarketParams, CalibrationTargets, run_opm_analysis
from portfolio import (
    build_capital_structure_100_0,
    build_capital_structure_90_10,
    build_capital_structure_75_25,
    build_capital_structure_50_50,
    build_capital_structure_25_75,
)

VOLATILITIES = [("40", 0.40, "40%"), ("80", 0.80, "80%"), ("120", 1.20, "120%"), ("160", 1.60, "160%")]
SCENARIOS = [
    ("s1", "100% (100/0)", "Scenario 1: 100/0 Structure (100% Common)", build_capital_structure_100_0),
    ("s2", "90% (90/10)", "Scenario 2: 90/10 Structure (90% Common)", build_capital_structure_90_10),
    ("s3", "75% (75/25)", "Scenario 3: 75/25 Structure (75% Common)", build_capital_structure_75_25),
    ("s4", "50% (50/50)", "Scenario 4: 50/50 Structure (50% Common)", build_capital_structure_50_50),
    ("s5", "25% (25/75)", "Scenario 5: 25/75 Structure (25% Common)", build_capital_structure_25_75),
]


def generate_html_report(filename="index.html"):
    """Compute combinations for two TEVs and display side-by-side clickable sensitivity matrices."""
    TEVS = [(550, "High Equity Value ($550M)"), (250, "Low Equity Value ($250M)")]
    
    reports = {
        (tev_val, v_k, s_k): run_opm_analysis(
            fn(),
            MarketParams(S0=50.0, sigma=v_val, T=2.0, r=0.05, q=0.0, N=5000),
            CalibrationTargets(tev0=tev_val, voleq=v_val),
        )
        for tev_val, _ in TEVS
        for v_k, v_val, _ in VOLATILITIES
        for s_k, _, _, fn in SCENARIOS
    }

    common_fvs = {}
    for (tev_val, v_k, s_k), rep in reports.items():
        m = re.search(r"Common FV = \$([0-9,.]+)", rep)
        common_fvs[(tev_val, v_k, s_k)] = f"${float(m.group(1)):.4f}" if m else "-"

    matrix_wrappers = ""
    for tev_val, tev_label in TEVS:
        matrix_rows = "".join(
            f'<tr><th class="matrix-y">{v_lbl}</th>' + "".join(
                f'<td class="matrix-cell{" active" if (tev_val == 550 and vk == "80" and sk == "s4") else ""}" '
                f'data-tev="{tev_val}" data-v="{vk}" data-s="{sk}" onclick="selectCell({tev_val}, \'{vk}\', \'{sk}\')">{common_fvs.get((tev_val, vk, sk), "-")}</td>'
                for sk, _, _, _ in SCENARIOS
            ) + '</tr>'
            for vk, _, v_lbl in VOLATILITIES
        )
        
        matrix_wrappers += f"""
    <div class="matrix-wrap">
      <div class="matrix-title">{tev_label} - Common Stock Value Per Share ($)</div>
      <table class="matrix-table">
        <thead>
          <tr>
            <th rowspan="2">Equity Volatility</th>
            <th colspan="{len(SCENARIOS)}">% of Common Shares in Capital Structure</th>
          </tr>
          <tr>
            {"".join(f"<th>{hdr}</th>" for _, hdr, _, _ in SCENARIOS)}
          </tr>
        </thead>
        <tbody>
          {matrix_rows}
        </tbody>
      </table>
    </div>"""

    tables = "".join(
        f'<div class="table-view" id="t-{tev_val}-{vk}-{sk}" style="display: {"block" if (tev_val == 550 and vk == "80" and sk == "s4") else "none"};">'
        f'<div class="table-title">{tev_label} | {html.escape(title)}</div>'
        f'<pre>{html.escape(reports[(tev_val, vk, sk)])}</pre>'
        f'</div>'
        for tev_val, tev_label in TEVS
        for vk, _, _ in VOLATILITIES
        for sk, _, title, _ in SCENARIOS
    )

    content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Option Pricing for Private Capital Structure (aka CCA)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; background: #fff; color: #000; }}
  .header-row {{ border-bottom: 2px solid #111; padding-bottom: 12px; margin-bottom: 16px; }}
  .title {{ font-size: 20px; font-weight: 700; margin: 0; }}
  .matrices-container {{ display: flex; flex-direction: row; gap: 40px; justify-content: flex-start; margin-bottom: 24px; flex-wrap: wrap; }}
  .matrix-wrap {{ display: flex; flex-direction: column; align-items: center; margin: 0; }}
  .matrix-title {{ font-size: 13px; font-weight: 700; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.03em; color: #222; }}
  .matrix-table {{ border-collapse: collapse; font-size: 12.5px; }}
  .matrix-table th, .matrix-table td {{ border: 1px solid #777; padding: 7px 16px; }}
  .matrix-table th {{ background: #f0f0f0; font-weight: 600; color: #111; text-align: center; }}
  .matrix-cell {{ font-family: Consolas, monospace; cursor: pointer; text-align: right; background: #fff; user-select: none; }}
  .matrix-cell:hover {{ background: #f0f0f0; }}
  .matrix-cell.active {{ background: #000; color: #fff; font-weight: bold; }}
  .table-title {{ font-size: 13.5px; font-weight: 600; padding: 7px 12px; background: #f4f4f4; border: 1px solid #ccc; border-bottom: none; }}
  pre {{ font-family: Consolas, "Courier New", monospace; font-size: 12.5px; line-height: 1.42; background: #fff; border: 1px solid #ccc; padding: 14px; margin: 0; overflow-x: auto; white-space: pre; }}
</style>
</head>
<body>
  <div class="header-row">
    <h1 class="title">Option Pricing for Private Capital Structure (aka CCA)</h1>
  </div>
  <div class="matrices-container">
    {matrix_wrappers}
  </div>
  <div id="table-container">
    {tables}
  </div>
<script>
  function selectCell(tev, v, s) {{
    document.querySelectorAll(".table-view").forEach(el => el.style.display = "none");
    const active = document.getElementById("t-" + tev + "-" + v + "-" + s);
    if (active) active.style.display = "block";
    document.querySelectorAll(".matrix-cell").forEach(c => {{
      c.classList.toggle("active", parseFloat(c.getAttribute("data-tev")) === tev && c.getAttribute("data-v") === v && c.getAttribute("data-s") === s);
    }});
  }}
</script>
</body>
</html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Report generated: {filename}")


if __name__ == "__main__":
    generate_html_report()
