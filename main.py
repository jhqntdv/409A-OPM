"""Capital Structure (OPM) Analysis: 50-Class Batch Valuation & Delta on Mutual Grid."""

import os
import webbrowser
from generate_report import generate_html_report

def main():
    # 1. Generate the report
    generate_html_report()
    
    # 2. Auto-pop the HTML in the default browser
    report_path = os.path.abspath("index.html")
    webbrowser.open(f"file:///{report_path.replace(os.sep, '/')}")

if __name__ == "__main__":
    main()
