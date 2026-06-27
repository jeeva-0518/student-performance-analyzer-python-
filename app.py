from __future__ import annotations

import csv
import html
import io
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8765
STORE: dict[str, "StoredAnalysis"] = {}


@dataclass
class StoredAnalysis:
    file_name: str
    source: pd.DataFrame
    threshold: float
    analysis: dict[str, Any]


@dataclass
class UploadedFile:
    filename: str
    content: bytes


@dataclass
class ParsedForm:
    fields: dict[str, str]
    files: dict[str, UploadedFile]

    def get(self, name: str, default: str = "") -> str:
        return self.fields.get(name, default)


def clean_header(value: Any, index: int) -> str:
    text = str(value).strip()
    return text or f"Column {index + 1}"


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_header(column, index) for index, column in enumerate(df.columns)]
    df = df.dropna(how="all")
    return df.reset_index(drop=True)


def detect_subject_columns(df: pd.DataFrame) -> list[str]:
    non_subject = re.compile(
        r"(roll|reg|admission|id|name|email|phone|mobile|class|section|gender|dob|date|attendance|rank|grade|result|remarks)",
        re.IGNORECASE,
    )
    subjects: list[str] = []

    for column in df.columns:
        if non_subject.search(column):
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue

        numeric_ratio = len(valid) / max(len(df), 1)
        plausible_marks = bool(((valid >= 0) & (valid <= 100)).all())
        if numeric_ratio >= 0.65 and plausible_marks:
            subjects.append(column)

    return subjects


def detect_name_column(df: pd.DataFrame, subjects: list[str]) -> str:
    for column in df.columns:
        if column not in subjects and re.search(r"(student|name)", column, re.IGNORECASE):
            return column

    for column in df.columns:
        if column in subjects:
            continue
        values = df[column].fillna("").astype(str)
        if values.str.contains(r"[A-Za-z]", regex=True).any():
            return column

    return next((column for column in df.columns if column not in subjects), df.columns[0])


def grade_from_average(averages: pd.Series) -> pd.Series:
    conditions = [
        averages >= 90,
        averages >= 80,
        averages >= 70,
        averages >= 60,
        averages >= 50,
    ]
    return pd.Series(np.select(conditions, ["A+", "A", "B+", "B", "C"], default="F"), index=averages.index)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "0"
    if not np.isfinite(number):
        return "0"
    rounded = round(number, 1)
    return str(int(rounded)) if rounded.is_integer() else f"{rounded:.1f}"


def pct(part: float, total: float) -> float:
    return float((part / total) * 100) if total else 0.0


def analyze_dataframe(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    df = normalize_dataframe(df)
    if df.empty:
        raise ValueError("The marksheet does not contain student rows.")

    subjects = detect_subject_columns(df)
    if not subjects:
        raise ValueError("No numeric subject columns were detected. Add marks columns like English, Maths, or Science.")

    name_column = detect_name_column(df, subjects)
    id_columns = [column for column in df.columns if column not in subjects and column != name_column]
    visible_id_columns = id_columns[:2]

    marks = df[subjects].apply(pd.to_numeric, errors="coerce")
    averages = marks.mean(axis=1).fillna(0)
    totals = marks.sum(axis=1).fillna(0)
    failed_subjects = marks.lt(threshold) & marks.notna()
    risk = averages.lt(threshold) | failed_subjects.any(axis=1)
    grades = grade_from_average(averages)

    students = pd.DataFrame(
        {
            "Student": df[name_column].fillna("").astype(str).str.strip(),
            "Total": totals,
            "Average": averages,
            "Grade": grades,
            "Status": np.where(risk, "At Risk", "On Track"),
            "Subjects Below Threshold": failed_subjects.apply(
                lambda row: ", ".join(row.index[row].tolist()) or "Overall average",
                axis=1,
            ),
        }
    )
    students.loc[students["Student"].eq(""), "Student"] = [
        f"Student {index + 1}" for index in students.index[students["Student"].eq("")]
    ]

    for column in reversed(visible_id_columns):
        students.insert(1, column, df[column].fillna("").astype(str))

    for subject in reversed(subjects):
        students.insert(1 + len(visible_id_columns), subject, marks[subject].fillna(0))

    students = students.sort_values("Average", ascending=False).reset_index(drop=True)
    students.insert(0, "Rank", np.arange(1, len(students) + 1))

    subject_rows = []
    for subject in subjects:
        values = marks[subject].dropna()
        subject_rows.append(
            {
                "Subject": subject,
                "Average": float(values.mean()) if not values.empty else 0.0,
                "Highest": float(values.max()) if not values.empty else 0.0,
                "Lowest": float(values.min()) if not values.empty else 0.0,
                "Pass Rate": pct(float((values >= threshold).sum()), float(len(values))),
                "Std Dev": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
            }
        )

    subject_stats = pd.DataFrame(subject_rows).sort_values("Average", ascending=False).reset_index(drop=True)
    strongest = subject_stats.iloc[0].to_dict()
    weakest = subject_stats.iloc[-1].to_dict()
    most_consistent = subject_stats.sort_values("Std Dev").iloc[0].to_dict()

    grade_order = ["A+", "A", "B+", "B", "C", "F"]
    grade_distribution = students["Grade"].value_counts().reindex(grade_order, fill_value=0).to_dict()
    risk_students = students[students["Status"].eq("At Risk")].sort_values("Average").reset_index(drop=True)

    metrics = {
        "student_count": int(len(students)),
        "subject_count": int(len(subjects)),
        "class_average": float(averages.mean()),
        "median": float(np.median(averages)),
        "highest": float(averages.max()),
        "lowest": float(averages.min()),
        "pass_rate": pct(float((~risk).sum()), float(len(students))),
        "risk_count": int(risk.sum()),
        "std_dev": float(np.std(averages)),
    }

    risk_phrase = (
        f"{metrics['risk_count']} student{'s' if metrics['risk_count'] != 1 else ''} need targeted academic support."
        if metrics["risk_count"]
        else "No students are below the selected risk threshold."
    )
    summary = (
        f"The class average is {fmt(metrics['class_average'])}% across {metrics['subject_count']} subjects. "
        f"{strongest['Subject']} is the strongest subject at {fmt(strongest['Average'])}%, while "
        f"{weakest['Subject']} needs the most attention at {fmt(weakest['Average'])}%. {risk_phrase}"
    )

    return {
        "subjects": subjects,
        "visible_id_columns": visible_id_columns,
        "students": students,
        "subject_stats": subject_stats,
        "strongest": strongest,
        "weakest": weakest,
        "most_consistent": most_consistent,
        "grade_distribution": grade_distribution,
        "risk_students": risk_students,
        "metrics": metrics,
        "summary": summary,
    }


def read_uploaded_dataframe(file_name: str, payload: bytes) -> pd.DataFrame:
    extension = Path(file_name).suffix.lower()
    stream = io.BytesIO(payload)

    if extension == ".csv":
        return pd.read_csv(stream)
    if extension == ".xlsx":
        return pd.read_excel(stream)
    raise ValueError("Please upload a CSV or XLSX marksheet.")


def load_demo_dataframe() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / "sample-marks.csv")


def store_analysis(file_name: str, source: pd.DataFrame, threshold: float) -> str:
    token = uuid.uuid4().hex
    STORE[token] = StoredAnalysis(
        file_name=file_name,
        source=source.copy(),
        threshold=threshold,
        analysis=analyze_dataframe(source, threshold),
    )
    return token


def recalculate(token: str, threshold: float) -> None:
    stored = STORE[token]
    stored.threshold = threshold
    stored.analysis = analyze_dataframe(stored.source, threshold)


def h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def grade_color(grade: str) -> str:
    return {
        "A+": "#1d8a7a",
        "A": "#2f6f9f",
        "B+": "#6b7f2a",
        "B": "#b7791f",
        "C": "#d9822b",
        "F": "#d95f4f",
    }.get(grade, "#2f6f9f")


def svg_bar_chart(rows: list[dict[str, Any]], max_value: float, suffix: str = "") -> str:
    width = 720
    height = 360
    pad_left = 58
    pad_top = 32
    pad_bottom = 94
    pad_right = 24
    chart_width = width - pad_left - pad_right
    chart_height = height - pad_top - pad_bottom
    gap = 18
    count = max(len(rows), 1)
    bar_width = max(24, (chart_width - gap * (count + 1)) / count)
    max_value = max(max_value, 1)

    pieces = [
        f'<svg class="svg-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Bar chart">',
        '<rect width="720" height="360" rx="8" fill="#fbfdfd"></rect>',
    ]
    for index in range(5):
        y = pad_top + chart_height - (chart_height * index / 4)
        label = round(max_value * index / 4)
        pieces.append(f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" stroke="#d9e2e8"></line>')
        pieces.append(f'<text x="16" y="{y + 5:.1f}" fill="#61717f" font-size="14">{label}</text>')

    for index, row in enumerate(rows):
        value = float(row["value"])
        label = str(row["label"])
        color = row.get("color", "#2f6f9f")
        x = pad_left + gap + index * (bar_width + gap)
        bar_height = (value / max_value) * chart_height
        y = pad_top + chart_height - bar_height
        text_x = x + bar_width / 2
        display_label = label if len(label) <= 18 else f"{label[:17]}."
        pieces.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"></rect>',
                f'<text x="{text_x:.1f}" y="{max(18, y - 8):.1f}" text-anchor="middle" fill="#1f2933" font-size="15" font-weight="700">{fmt(value)}{suffix}</text>',
                f'<text x="{text_x:.1f}" y="{height - 20}" text-anchor="middle" fill="#314956" font-size="13">{h(display_label)}</text>',
            ]
        )

    pieces.append("</svg>")
    return "".join(pieces)


def metric_card(label: str, value: str, note: str) -> str:
    return f"""
    <article class="metric-card">
      <span>{h(label)}</span>
      <strong>{h(value)}</strong>
      <small>{h(note)}</small>
    </article>
    """


def render_empty_state() -> str:
    return """
    <section class="empty-state">
      <h2>What this analyzer does</h2>
      <div class="requirement-grid">
        <div><strong>Data ingestion</strong><span>CSV and Excel upload handled by Python.</span></div>
        <div><strong>Calculations</strong><span>Pandas and NumPy calculate averages, grades, spread, and pass rate.</span></div>
        <div><strong>Insights</strong><span>Strongest and weakest subjects are detected automatically.</span></div>
        <div><strong>Risk flags</strong><span>Students below the chosen threshold are highlighted for support.</span></div>
        <div><strong>Reporting</strong><span>Download an executive HTML report and analyzed CSV file.</span></div>
      </div>
    </section>
    """


def render_dashboard(token: str, stored: StoredAnalysis, search: str = "") -> str:
    analysis = stored.analysis
    metrics = analysis["metrics"]
    subject_stats = analysis["subject_stats"]
    students = analysis["students"]
    risk_students = analysis["risk_students"]
    threshold = stored.threshold

    if search:
        mask = students.astype(str).apply(lambda row: row.str.lower().str.contains(search.lower()).any(), axis=1)
        visible_students = students[mask]
    else:
        visible_students = students

    grade_rows = [
        {"label": grade, "value": count, "color": grade_color(grade)}
        for grade, count in analysis["grade_distribution"].items()
    ]
    subject_rows = [
        {
            "label": row["Subject"],
            "value": row["Average"],
            "color": "#1d8a7a" if index == 0 else "#d95f4f" if index == len(subject_stats) - 1 else "#2f6f9f",
        }
        for index, row in subject_stats.iterrows()
    ]

    metric_html = "".join(
        [
            metric_card("Class Average", f"{fmt(metrics['class_average'])}%", f"{metrics['student_count']} students"),
            metric_card("Median Score", f"{fmt(metrics['median'])}%", f"Std dev {fmt(metrics['std_dev'])}"),
            metric_card("Highest Average", f"{fmt(metrics['highest'])}%", "Top student score"),
            metric_card("Pass Rate", f"{fmt(metrics['pass_rate'])}%", f"Threshold {fmt(threshold)}%"),
            metric_card("Risk Count", str(metrics["risk_count"]), "Need extra support"),
        ]
    )

    subject_insights = "".join(
        [
            insight_item("Strongest", analysis["strongest"]["Subject"], f"{fmt(analysis['strongest']['Average'])}% avg"),
            insight_item("Weakest", analysis["weakest"]["Subject"], f"{fmt(analysis['weakest']['Average'])}% avg"),
            insight_item(
                "Most Consistent",
                analysis["most_consistent"]["Subject"],
                f"{fmt(analysis['most_consistent']['Std Dev'])} std dev",
            ),
        ]
    )

    if risk_students.empty:
        risk_insights = '<div class="insight-item"><strong>No students flagged</strong><span>Good</span></div>'
    else:
        risk_insights = "".join(
            insight_item(row["Grade"], row["Student"], f"{fmt(row['Average'])}%")
            for _, row in risk_students.head(3).iterrows()
        )

    table_headers = "".join(f"<th>{h(column)}</th>" for column in visible_students.columns)
    table_rows = []
    for _, row in visible_students.iterrows():
        cells = []
        for column in visible_students.columns:
            value = row[column]
            if column in {"Average"}:
                cells.append(f"<td>{fmt(value)}%</td>")
            elif column == "Grade":
                cells.append(f'<td><span class="grade">{h(value)}</span></td>')
            elif column == "Status":
                css = "status-risk" if value == "At Risk" else "status-ok"
                cells.append(f'<td><span class="status {css}">{h(value)}</span></td>')
            elif isinstance(value, (int, float, np.number)):
                cells.append(f"<td>{fmt(value)}</td>")
            else:
                cells.append(f"<td>{h(value)}</td>")
        table_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <section class="dashboard">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Dashboard</p>
          <h2>{h(Path(stored.file_name).stem)}</h2>
        </div>
        <div class="actions">
          <a class="secondary-button link-button" href="/report?token={h(token)}">Download Report</a>
          <a class="secondary-button link-button" href="/download-csv?token={h(token)}">Download Analyzed CSV</a>
        </div>
      </div>

      <div class="metric-grid">{metric_html}</div>

      <div class="insight-grid">
        <article class="insight-block"><h3>Subject Insights</h3><div class="insight-list">{subject_insights}</div></article>
        <article class="insight-block"><h3>Risk Analysis</h3><div class="insight-list">{risk_insights}</div></article>
        <article class="insight-block"><h3>Executive Summary</h3><p>{h(analysis["summary"])}</p></article>
      </div>

      <div class="chart-grid">
        <article class="chart-panel">
          <div class="panel-heading"><h3>Grade Distribution</h3><span>{metrics['student_count']} students</span></div>
          {svg_bar_chart(grade_rows, max(analysis["grade_distribution"].values()) or 1)}
        </article>
        <article class="chart-panel">
          <div class="panel-heading"><h3>Subject Averages</h3><span>{len(analysis["subjects"])} subjects</span></div>
          {svg_bar_chart(subject_rows, 100, "%")}
        </article>
      </div>

      <section class="table-section">
        <div class="panel-heading">
          <div><h3>Student Results</h3><span>{len(visible_students)} shown</span></div>
          <form class="search-form" action="/view" method="get">
            <input type="hidden" name="token" value="{h(token)}">
            <input type="search" name="search" value="{h(search)}" placeholder="Search students" aria-label="Search students">
            <button class="secondary-button" type="submit">Search</button>
          </form>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr>{table_headers}</tr></thead>
            <tbody>{''.join(table_rows)}</tbody>
          </table>
        </div>
      </section>
    </section>
    """


def insight_item(label: Any, title: Any, value: Any) -> str:
    return f"""
    <div class="insight-item">
      <div><small>{h(label)}</small><br><strong>{h(title)}</strong></div>
      <span>{h(value)}</span>
    </div>
    """


def render_upload_panel(token: str | None = None, threshold: float = 50) -> str:
    recalc_form = ""
    if token:
        recalc_form = f"""
        <form action="/recalculate" method="post">
          <input type="hidden" name="token" value="{h(token)}">
          <label for="current-threshold">Current data threshold</label>
          <div class="threshold-row">
            <input id="current-threshold" name="threshold" class="number-input" type="number" min="0" max="100" value="{h(fmt(threshold))}">
            <output>{h(fmt(threshold))}%</output>
          </div>
          <button class="secondary-button" type="submit">Recalculate Current Data</button>
        </form>
        """

    return f"""
    <section class="workflow-band">
      <div class="upload-panel">
        <form class="drop-zone" action="/analyze" method="post" enctype="multipart/form-data">
          <input class="file-input" name="file" type="file" accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" required>
          <div class="upload-icon" aria-hidden="true">+</div>
          <h2>Upload marksheet</h2>
          <p>Select a CSV or XLSX file. Python will process it with pandas and NumPy.</p>
          <label for="upload-threshold">Risk threshold</label>
          <input id="upload-threshold" name="threshold" class="number-input" type="number" min="0" max="100" value="{h(fmt(threshold))}">
          <button class="primary-button" type="submit">Analyze File</button>
        </form>

        <div class="settings-panel">
          <form action="/demo" method="post">
            <label for="demo-threshold">Risk threshold</label>
            <div class="threshold-row">
              <input id="demo-threshold" name="threshold" class="number-input" type="number" min="0" max="100" value="{h(fmt(threshold))}">
              <output>{h(fmt(threshold))}%</output>
            </div>
            <button class="secondary-button" type="submit">Load Demo Data</button>
          </form>
          {recalc_form}
          <a class="ghost-button link-button" href="/">Reset</a>
        </div>
      </div>
    </section>
    """


def render_page(
    token: str | None = None,
    stored: StoredAnalysis | None = None,
    search: str = "",
    error: str | None = None,
) -> str:
    threshold = stored.threshold if stored else 50
    dashboard = render_dashboard(token, stored, search) if token and stored else render_empty_state()
    error_html = f'<div class="toast show">{h(error)}</div>' if error else ""
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Student Performance Analyzer</title>
    <link rel="stylesheet" href="/styles.css">
  </head>
  <body>
    <header class="topbar">
      <div>
        <p class="eyebrow">Team 6</p>
        <h1>Student Performance Analyzer</h1>
        <p class="intro">
          Upload a marksheet, calculate class and subject performance, flag students who need support,
          and export a classroom summary report.
        </p>
      </div>
    </header>
    <main>
      {render_upload_panel(token, threshold)}
      {dashboard}
    </main>
    {error_html}
  </body>
</html>"""


def render_report(stored: StoredAnalysis) -> str:
    analysis = stored.analysis
    metrics = analysis["metrics"]
    subject_rows = "".join(
        f"<tr><td>{h(row['Subject'])}</td><td>{fmt(row['Average'])}%</td><td>{fmt(row['Highest'])}</td>"
        f"<td>{fmt(row['Lowest'])}</td><td>{fmt(row['Pass Rate'])}%</td></tr>"
        for _, row in analysis["subject_stats"].iterrows()
    )
    risk = analysis["risk_students"]
    if risk.empty:
        risk_rows = "<p>No students were flagged at the selected threshold.</p>"
    else:
        risk_rows = (
            "<table><thead><tr><th>Student</th><th>Average</th><th>Grade</th><th>Subjects Below Threshold</th></tr></thead><tbody>"
            + "".join(
                f"<tr><td>{h(row['Student'])}</td><td>{fmt(row['Average'])}%</td><td>{h(row['Grade'])}</td>"
                f"<td>{h(row['Subjects Below Threshold'])}</td></tr>"
                for _, row in risk.iterrows()
            )
            + "</tbody></table>"
        )

    student_rows = "".join(
        f"<tr><td>{h(row['Student'])}</td><td>{fmt(row['Total'])}</td><td>{fmt(row['Average'])}%</td>"
        f"<td>{h(row['Grade'])}</td><td>{h(row['Status'])}</td></tr>"
        for _, row in analysis["students"].iterrows()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Student Performance Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#1f2933;line-height:1.45}}
h1{{margin-bottom:4px}} h2{{margin-top:28px}}
table{{border-collapse:collapse;width:100%;margin-top:12px}}
th,td{{border:1px solid #d9e2e8;padding:8px;text-align:left}}
th{{background:#eef5f4}} .metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.metric{{border:1px solid #d9e2e8;padding:12px}}
</style>
</head>
<body>
<h1>Student Performance Analyzer Report</h1>
<p><strong>Dataset:</strong> {h(stored.file_name)} | <strong>Risk threshold:</strong> {fmt(stored.threshold)}%</p>
<div class="metrics">
<div class="metric"><strong>Class average</strong><br>{fmt(metrics['class_average'])}%</div>
<div class="metric"><strong>Pass rate</strong><br>{fmt(metrics['pass_rate'])}%</div>
<div class="metric"><strong>Strongest subject</strong><br>{h(analysis['strongest']['Subject'])} ({fmt(analysis['strongest']['Average'])}%)</div>
<div class="metric"><strong>Weakest subject</strong><br>{h(analysis['weakest']['Subject'])} ({fmt(analysis['weakest']['Average'])}%)</div>
</div>
<h2>Executive Summary</h2>
<p>{h(analysis['summary'])}</p>
<h2>Subject Performance</h2>
<table><thead><tr><th>Subject</th><th>Average</th><th>Highest</th><th>Lowest</th><th>Pass Rate</th></tr></thead><tbody>{subject_rows}</tbody></table>
<h2>Students Requiring Support</h2>
{risk_rows}
<h2>Complete Student Results</h2>
<table><thead><tr><th>Student</th><th>Total</th><th>Average</th><th>Grade</th><th>Status</th></tr></thead><tbody>{student_rows}</tbody></table>
</body>
</html>"""


def analyzed_csv(stored: StoredAnalysis) -> str:
    output = io.StringIO()
    stored.analysis["students"].to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    return output.getvalue()


def parse_post_form(headers: Any, body: bytes) -> ParsedForm:
    content_type = headers.get("Content-Type", "")
    fields: dict[str, str] = {}
    files: dict[str, UploadedFile] = {}

    if content_type.startswith("application/x-www-form-urlencoded"):
        decoded = body.decode("utf-8", errors="replace")
        for name, values in parse_qs(decoded, keep_blank_values=True).items():
            fields[name] = values[0] if values else ""
        return ParsedForm(fields, files)

    if content_type.startswith("multipart/form-data"):
        message_bytes = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n"
            "\r\n"
        ).encode("utf-8") + body
        message = BytesParser(policy=policy.default).parsebytes(message_bytes)

        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue

            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if filename:
                files[name] = UploadedFile(filename=filename, content=payload)
            else:
                charset = part.get_content_charset() or "utf-8"
                fields[name] = payload.decode(charset, errors="replace")

    return ParsedForm(fields, files)


class AnalyzerHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        sys.stdout.write(f"[{timestamp}] {self.address_string()} {format % args}\n")

    def send_text(self, content: str, content_type: str = "text/html", status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def not_found(self) -> None:
        self.send_text(render_page(error="Page not found."), status=HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/":
            self.send_text(render_page())
            return

        if parsed.path == "/styles.css":
            self.send_text((BASE_DIR / "styles.css").read_text(encoding="utf-8"), "text/css")
            return

        if parsed.path == "/sample-marks.csv":
            self.send_text((BASE_DIR / "sample-marks.csv").read_text(encoding="utf-8"), "text/csv")
            return

        if parsed.path == "/view":
            token = params.get("token", [""])[0]
            stored = STORE.get(token)
            if not stored:
                self.send_text(render_page(error="That analysis session was not found. Please upload the file again."))
                return
            self.send_text(render_page(token, stored, params.get("search", [""])[0]))
            return

        if parsed.path == "/report":
            token = params.get("token", [""])[0]
            stored = STORE.get(token)
            if not stored:
                self.not_found()
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="student-performance-report.html"')
            payload = render_report(stored).encode("utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if parsed.path == "/download-csv":
            token = params.get("token", [""])[0]
            stored = STORE.get(token)
            if not stored:
                self.not_found()
                return
            payload = analyzed_csv(stored).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="analyzed-student-performance.csv"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.not_found()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            form = parse_post_form(self.headers, self.rfile.read(content_length))

            if parsed.path == "/demo":
                threshold = float(form.get("threshold", "50"))
                token = store_analysis("Demo marksheet", load_demo_dataframe(), threshold)
                self.redirect(f"/view?token={token}")
                return

            if parsed.path == "/recalculate":
                token = form.get("token", "")
                if token not in STORE:
                    raise ValueError("That analysis session was not found. Please upload the file again.")
                threshold = float(form.get("threshold", "50"))
                recalculate(token, threshold)
                self.redirect(f"/view?token={token}")
                return

            if parsed.path == "/analyze":
                threshold = float(form.get("threshold", "50"))
                file_item = form.files.get("file")
                if file_item is None or not file_item.filename:
                    raise ValueError("Please choose a CSV or XLSX file.")
                df = read_uploaded_dataframe(file_item.filename, file_item.content)
                token = store_analysis(file_item.filename, df, threshold)
                self.redirect(f"/view?token={token}")
                return

            self.not_found()
        except Exception as exc:
            self.send_text(render_page(error=str(exc)), status=HTTPStatus.BAD_REQUEST)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AnalyzerHandler)
    print(f"Student Performance Analyzer running at http://{HOST}:{PORT}/")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
