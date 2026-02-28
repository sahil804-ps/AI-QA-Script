"""
dashboard_gen.py — Generates a premium HTML dashboard from AI-QA JSON reports.
v2: Evolution Phase with Advanced Metrics (Consistency Score, Agreement Index, Risk Index).
"""

import json
import os
import glob
from datetime import datetime

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI QA Framework Dashboard v2</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --primary: #38bdf8;
            --secondary: #818cf8;
            --accent: #f472b6;
            --success: #22c55e;
            --warn: #f59e0b;
            --danger: #ef4444;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 2rem;
            line-height: 1.5;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            margin-bottom: 3rem;
            text-align: left;
            border-left: 5px solid var(--primary);
            padding-left: 1.5rem;
        }

        h1 {
            font-weight: 800;
            font-size: 2.5rem;
            letter-spacing: -0.05rem;
            margin: 0;
        }

        .framework-tag {
            color: var(--primary);
            text-transform: uppercase;
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.2rem;
        }

        .timestamp {
            color: var(--text-dim);
            font-size: 0.9rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }

        .stat-card {
            background: linear-gradient(145deg, var(--card-bg), #111827);
            padding: 2rem;
            border-radius: 1.5rem;
            border: 1px solid #334155;
            position: relative;
            overflow: hidden;
        }

        .stat-card::after {
            content: '';
            position: absolute;
            top: 0; right: 0;
            width: 100px; height: 100px;
            background: var(--primary);
            filter: blur(80px);
            opacity: 0.1;
        }

        .stat-val {
            font-size: 2.5rem;
            font-weight: 800;
            display: block;
            margin-bottom: 0.5rem;
        }

        .stat-label {
            color: var(--text-dim);
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.1rem;
        }

        .report-section {
            display: grid;
            gap: 1.5rem;
        }

        .card {
            background: var(--card-bg);
            border-radius: 1.25rem;
            border: 1px solid #334155;
            padding: 2rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .card:hover {
            transform: scale(1.01);
            border-color: var(--secondary);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }

        .badge {
            padding: 0.4rem 1rem;
            border-radius: 0.75rem;
            font-size: 0.7rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .badge-PASS { background: rgba(34, 197, 94, 0.15); color: var(--success); border: 1px solid var(--success); }
        .badge-WARN { background: rgba(245, 158, 11, 0.15); color: var(--warn); border: 1px solid var(--warn); }
        .badge-FAIL { background: rgba(239, 68, 68, 0.15); color: var(--danger); border: 1px solid var(--danger); }

        .prompt-text {
            font-weight: 600;
            font-size: 1.2rem;
            color: var(--text-main);
        }

        .metrics-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            background: #0f172a;
            padding: 1rem;
            border-radius: 1rem;
        }

        .metric-item {
            display: flex;
            flex-direction: column;
        }

        .metric-val {
            font-weight: 800;
            font-size: 1.1rem;
        }

        .metric-label {
            font-size: 0.65rem;
            color: var(--text-dim);
            text-transform: uppercase;
            font-weight: 600;
        }

        .drift-up { color: var(--success); }
        .drift-down { color: var(--danger); }

        footer {
            margin-top: 5rem;
            padding-top: 2rem;
            border-top: 1px solid #334155;
            text-align: center;
            color: var(--text-dim);
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="framework-tag">AI Evaluation Framework v2</div>
            <h1>Quality Audit Dashboard</h1>
            <div class="timestamp">Session: {{timestamp}}</div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-val" style="color: var(--primary)">{{total_tests}}</span>
                <span class="stat-label">Total Prompts</span>
            </div>
            <div class="stat-card">
                <span class="stat-val" style="color: var(--secondary)">{{avg_consistency}}%</span>
                <span class="stat-label">Avg Consistency</span>
            </div>
            <div class="stat-card">
                <span class="stat-val" style="color: var(--accent)">{{avg_agreement}}%</span>
                <span class="stat-label">Model Agreement Index</span>
            </div>
            <div class="stat-card">
                <span class="stat-val" style="color: var(--success)">{{passed_tests}}</span>
                <span class="stat-label">Framework Verdict: PASS</span>
            </div>
        </div>

        <div class="report-section">
            {{report_items}}
        </div>

        <footer>
            Built with 🧰 AI-QA-Toolkit Evolution Phase. Data-driven AI Reliability.
        </footer>
    </div>
</body>
</html>
"""

def generate_dashboard(report_dir: str):
    json_files = glob.glob(os.path.join(report_dir, "*.json"))
    if not json_files:
        return None

    reports = []
    for f in sorted(json_files):
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                reports.append(json.load(jf))
        except: continue

    total = len(reports)
    passed = sum(1 for r in reports if r.get("overall_verdict") == "PASS")
    
    # Calculate Averages for Framework Summary
    cons_scores = [r.get("consistency", {}).get("consistency_score", 0) for r in reports if r.get("consistency")]
    agree_scores = [r.get("consistency", {}).get("agreement_index", 0) for r in reports if r.get("consistency")]
    
    avg_cons = (sum(cons_scores) / len(cons_scores)) if cons_scores else 0
    avg_agree = (sum(agree_scores) / len(agree_scores)) if agree_scores else 0

    report_html = ""
    for r in reports:
        prompt = r.get("prompt", "N/A")
        verdict = r.get("overall_verdict", "UNKNOWN")
        cons = r.get("consistency", {}).get("consistency_score", 0)
        agree = r.get("consistency", {}).get("agreement_index", 0)
        risk = r.get("hallucination", [{}])[0].get("risk_index", 0)
        risk_color = "var(--success)" if risk < 30 else "var(--danger)" if risk > 70 else "var(--warn)"
        conf = r.get("confidence", [{}])[0].get("score", 0)

        report_html += f'''
        <div class="card">
            <div class="card-header">
                <div class="prompt-text">"{prompt}"</div>
                <span class="badge badge-{verdict}">{verdict}</span>
            </div>
            
            <div class="metrics-row">
                <div class="metric-item">
                    <span class="metric-label">Consistency</span>
                    <span class="metric-val" style="color: var(--secondary)">{cons}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Agreement Index</span>
                    <span class="metric-val" style="color: var(--accent)">{agree}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Hallucination Risk</span>
                    <span class="metric-val" style="color: {risk_color}">{risk}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Avg Confidence</span>
                    <span class="metric-val">{conf}</span>
                </div>
            </div>
        </div>
        '''

    html = HTML_TEMPLATE.replace("{{timestamp}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    html = html.replace("{{total_tests}}", str(total))
    html = html.replace("{{passed_tests}}", str(passed))
    html = html.replace("{{avg_consistency}}", f"{avg_cons:.1f}")
    html = html.replace("{{avg_agreement}}", f"{avg_agree:.1f}")
    html = html.replace("{{report_items}}", report_html)

    output_path = os.path.join(report_dir, "dashboard.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = generate_dashboard(sys.argv[1])
        if path:
            print(f"Dashboard created: {path}")
