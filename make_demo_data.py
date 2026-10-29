"""Generate fictional observations; the relationship is designed, not discovered."""
import csv
from pathlib import Path

def generate():
    rows = []
    for i in range(120):
        hours = round(0.5 + ((i * 37) % 95) / 10, 1)
        noise = ((i * 17) % 21) - 10
        score = round(35 + 5 * hours + noise, 1)
        rows.append([f"S{i + 1:03d}", "" if i in [12, 77] else hours, score])
    return rows

if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "data/students.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["student_id", "study_hours", "exam_score"])
        writer.writerows(generate())
    print(f"Wrote {path}")
