# Student Performance Analyzer

Team 6 project deliverable for an educational analytics dashboard.

## Features

- CSV and XLSX marksheet upload
- Python backend using pandas and NumPy
- Automatic subject detection from numeric marks columns
- Overall average, median, highest score, pass rate, and risk count
- Subject-wise average, strongest subject, weakest subject, and distribution metrics
- Student-level grade assignment and risk flagging
- Grade distribution and subject comparison charts rendered as HTML/SVG
- Downloadable executive HTML report
- Downloadable analyzed CSV

## How to Run

Run the Python app from this folder:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:8765/
```

Use `sample-marks.csv` if you want to test the dashboard quickly.

## Expected File Format

The first row should contain column names. Include one student name column and one or more numeric subject columns.

Example:

```csv
Student Name,Roll No,English,Mathematics,Science
Aarav Kumar,101,82,91,88
Diya Sharma,102,74,69,78
```

The analyzer ignores non-subject identifier columns such as roll number, ID, email, class, and section.
