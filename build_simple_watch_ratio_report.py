from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = "validation_predictions_comparison.csv"
OUTPUT_HTML = "validation_watch_ratio_simple.html"
REPORT_DIR = "prediction_reports"


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def _error_badge(value):
    value = float(value)
    if value < 0.10:
        tone = "good"
    elif value < 0.30:
        tone = "mid"
    else:
        tone = "bad"
    return f"<span class='badge {tone}'>{_fmt(value)}</span>"


def _compare_bar(actual, predicted, cap=2.0):
    actual = float(actual)
    predicted = float(predicted)
    actual_width = max(0.0, min(actual, cap)) / cap * 100.0
    predicted_width = max(0.0, min(predicted, cap)) / cap * 100.0
    return f"""
    <div class="compare-cell">
      <div class="compare-values">
        <span class="label actual">Actual {_fmt(actual)}</span>
        <span class="label pred">Pred {_fmt(predicted)}</span>
      </div>
      <div class="track">
        <div class="fill actual" style="width:{actual_width:.1f}%"></div>
        <div class="fill pred" style="width:{predicted_width:.1f}%"></div>
      </div>
    </div>
    """


def _build_html(df):
    mae = float(df["baseline_abs_error"].mean())
    medae = float(df["baseline_abs_error"].median())
    p90 = float(df["baseline_abs_error"].quantile(0.9))

    rows_html = []
    for row in df.itertuples(index=False):
        search_text = " ".join(
            [
                str(row.eval_row_index),
                str(row.user_id),
                str(row.video_id),
                str(row.timestamp),
            ]
        ).lower()
        rows_html.append(
            "<tr data-search='{}'>".format(escape(search_text)) +
            f"<td>{row.eval_row_index}</td>" +
            f"<td>{row.user_id}</td>" +
            f"<td>{row.video_id}</td>" +
            f"<td>{_fmt(row.timestamp, 0)}</td>" +
            f"<td>{_compare_bar(row.actual_watch_ratio, row.baseline_pred_watch_ratio)}</td>" +
            f"<td>{_fmt(row.actual_watch_ratio)}</td>" +
            f"<td>{_fmt(row.baseline_pred_watch_ratio)}</td>" +
            f"<td>{_error_badge(row.baseline_abs_error)}</td>" +
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Watch Ratio 简洁对比表</title>
  <style>
    :root {{
      --bg: #f7f4ee;
      --panel: #fffdfa;
      --ink: #1f2937;
      --muted: #667085;
      --line: #e6decf;
      --actual: #2563eb;
      --pred: #059669;
      --good: #157f3b;
      --mid: #a16207;
      --bad: #b42318;
      --shadow: 0 12px 30px rgba(53, 38, 14, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 30%),
        radial-gradient(circle at top right, rgba(5, 150, 105, 0.10), transparent 28%),
        var(--bg);
      color: var(--ink);
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(5, 150, 105, 0.10));
      border: 1px solid rgba(37, 99, 235, 0.15);
      border-radius: 24px;
      padding: 24px;
      box-shadow: var(--shadow);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      line-height: 1.15;
    }}
    .sub {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 22px 0;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    .card-label {{
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .card-value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 16px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(255,255,255,0.75);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin: 18px 0 12px;
    }}
    .toolbar input {{
      flex: 1 1 340px;
      min-width: 260px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.92);
      font-size: 14px;
    }}
    .count {{
      color: var(--muted);
      font-size: 14px;
    }}
    .table-wrap {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      overflow: auto;
      box-shadow: var(--shadow);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 980px;
    }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #faf6ef;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid #eee5d8;
      text-align: left;
      vertical-align: middle;
      font-size: 13px;
      white-space: nowrap;
    }}
    tbody tr:hover {{
      background: rgba(37, 99, 235, 0.04);
    }}
    .compare-cell {{
      min-width: 290px;
    }}
    .compare-values {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
      font-variant-numeric: tabular-nums;
      font-size: 12px;
    }}
    .label.actual {{
      color: var(--actual);
      font-weight: 700;
    }}
    .label.pred {{
      color: var(--pred);
      font-weight: 700;
    }}
    .track {{
      position: relative;
      width: 270px;
      height: 14px;
      border-radius: 999px;
      background: #ece5d8;
      overflow: hidden;
    }}
    .fill {{
      position: absolute;
      top: 0;
      left: 0;
      height: 100%;
      border-radius: 999px;
      opacity: 0.88;
    }}
    .fill.actual {{
      background: var(--actual);
    }}
    .fill.pred {{
      background: var(--pred);
      mix-blend-mode: multiply;
    }}
    .badge {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}
    .badge.good {{
      background: rgba(21, 127, 59, 0.12);
      color: var(--good);
    }}
    .badge.mid {{
      background: rgba(161, 98, 7, 0.14);
      color: var(--mid);
    }}
    .badge.bad {{
      background: rgba(180, 35, 24, 0.12);
      color: var(--bad);
    }}
    @media (max-width: 900px) {{
      .page {{
        padding: 18px 14px 24px;
      }}
      h1 {{
        font-size: 24px;
      }}
      .track {{
        width: 220px;
      }}
      .compare-cell {{
        min-width: 240px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Watch Ratio 简洁对比表</h1>
      <p class="sub">
        这份版本只保留最关键的信息：每一行真实 <code>watch_ratio</code>、模型预测 <code>watch_ratio</code>，
        以及它们之间的绝对误差。蓝色代表真实值，绿色代表预测值。
      </p>
      <div class="legend">
        <div class="legend-item"><span class="dot" style="background: var(--actual);"></span>真实 watch_ratio</div>
        <div class="legend-item"><span class="dot" style="background: var(--pred);"></span>预测 watch_ratio</div>
      </div>
    </section>

    <section class="cards">
      <div class="card"><div class="card-label">Rows</div><div class="card-value">{len(df)}</div></div>
      <div class="card"><div class="card-label">Users</div><div class="card-value">{int(df['user_id'].nunique())}</div></div>
      <div class="card"><div class="card-label">Mean Abs Error</div><div class="card-value">{_fmt(mae)}</div></div>
      <div class="card"><div class="card-label">Median Abs Error</div><div class="card-value">{_fmt(medae)}</div></div>
      <div class="card"><div class="card-label">P90 Abs Error</div><div class="card-value">{_fmt(p90)}</div></div>
    </section>

    <div class="toolbar">
      <input id="searchBox" type="text" placeholder="按 row / user_id / video_id / timestamp 搜索...">
      <div class="count">Visible rows: <span id="visibleCount">{len(df)}</span> / {len(df)}</div>
    </div>

    <div class="table-wrap">
      <table id="simpleTable">
        <thead>
          <tr>
            <th>Row</th>
            <th>User</th>
            <th>Video</th>
            <th>Timestamp</th>
            <th>Actual vs Predicted</th>
            <th>Actual</th>
            <th>Predicted</th>
            <th>Abs Error</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const searchBox = document.getElementById("searchBox");
    const visibleCount = document.getElementById("visibleCount");
    const rows = Array.from(document.querySelectorAll("#simpleTable tbody tr"));

    function applyFilter() {{
      const q = searchBox.value.trim().toLowerCase();
      let visible = 0;
      rows.forEach((row) => {{
        const hay = row.dataset.search || "";
        const show = q === "" || hay.includes(q);
        row.style.display = show ? "" : "none";
        if (show) visible += 1;
      }});
      visibleCount.textContent = String(visible);
    }}

    searchBox.addEventListener("input", applyFilter);
  </script>
</body>
</html>
"""


def main():
    project_root = Path(__file__).resolve().parent
    report_dir = project_root / REPORT_DIR
    input_csv = report_dir / INPUT_CSV
    output_html = report_dir / OUTPUT_HTML

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing input CSV: {input_csv}. Please generate validation_predictions_comparison.csv first."
        )

    df = pd.read_csv(input_csv)
    html = _build_html(df)
    output_html.write_text(html, encoding="utf-8")

    print(f"input_csv: {input_csv}")
    print(f"output_html: {output_html}")
    print(f"rows: {len(df)}")
    print(f"users: {int(df['user_id'].nunique())}")


if __name__ == "__main__":
    main()
