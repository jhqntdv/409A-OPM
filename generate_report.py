from black_scholes import MarketParams, CalibrationTargets, run_opm_analysis
from portfolio import (
    build_capital_structure_100_0,
    build_capital_structure_90_10,
    build_capital_structure_75_25,
    build_capital_structure_50_50,
    build_capital_structure_25_75
)

def run_to_string(portfolio_func, title):
    report_string = f"<h2>{title}</h2>\n<pre>\n"
    portfolio = portfolio_func()
    params = MarketParams(S0=80, sigma=0.85, T=2.0, r=0.05, q=0.0, N=10000)
    targets = CalibrationTargets(tev0=850.0, voleq=0.80)
    
    report_string += run_opm_analysis(portfolio, params, targets)
    report_string += "\n</pre>\n"
    
    return report_string

def generate_html_report(filename="capital_structure_report.html"):
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Capital Structure</title>
    <style>
        body { font-family: sans-serif; margin: 20px; color: #333; }
        h1 { border-bottom: 2px solid #ccc; padding-bottom: 10px; }
        h2 { color: #0056b3; margin-top: 40px; }
        pre { 
            font-family: Consolas, 'Courier New', monospace; 
            font-size: 13px; 
            background: #f8f9fa; 
            padding: 15px; 
            border: 1px solid #ddd;
            border-radius: 5px; 
            overflow-x: auto; 
            line-height: 1.4;
        }
    </style>
</head>
<body>
    <h1>Capital Structure</h1>
"""
    scenarios = [
        (build_capital_structure_100_0, "Scenario 1: 100/0 Structure (100% Common)"),
        (build_capital_structure_90_10, "Scenario 2: 90/10 Structure (90% Common)"),
        (build_capital_structure_75_25, "Scenario 3: 75/25 Structure (75% Common)"),
        (build_capital_structure_50_50, "Scenario 4: 50/50 Structure (50% Common)"),
        (build_capital_structure_25_75, "Scenario 5: 25/75 Structure (25% Common)")
    ]
    for func, title in scenarios:
        html_content += run_to_string(func, title)
    
    html_content += "</body>\n</html>"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Report generated: {filename}")

if __name__ == "__main__":
    generate_html_report()
