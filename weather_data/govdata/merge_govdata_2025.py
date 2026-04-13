import csv
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple


START = date(2025, 1, 1)
END = date(2025, 12, 31)

# Output column names mapped from source filenames.
FILE_TO_COLUMN = {
    "daily_HKO_RH_ALL.csv": "HKO_RH",
    "daily_KP_GSR_ALL.csv": "KP_GSR",
    "daily_KP_RF_ALL.csv": "KP_RF",
    "daily_KP_RH_ALL.csv": "KP_RH",
    "daily_KP_WSPD_ALL.csv": "KP_WSPD",
}


def load_one_file(path: Path) -> Dict[date, str]:
    """Load one govdata file and return mapping: date -> value."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("年/Year,月/Month,日/Day,數值/Value"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Cannot find data header in {path.name}")

    rows = list(csv.reader(lines[header_idx:]))
    if not rows:
        return {}

    data = {}
    for row in rows[1:]:
        if len(row) < 4:
            continue
        try:
            y = int(row[0])
            m = int(row[1])
            d = int(row[2])
            dt = date(y, m, d)
        except ValueError:
            continue

        if START <= dt <= END:
            data[dt] = row[3].strip()

    return data


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    files = [base_dir / name for name in FILE_TO_COLUMN.keys()]

    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Missing input file: {f}")

    series: List[Tuple[str, Dict[date, str]]] = []
    for f in files:
        col = FILE_TO_COLUMN[f.name]
        series.append((col, load_one_file(f)))

    rows = []
    day = START
    while day <= END:
        out_row = {"Date": day.isoformat()}
        for col, data in series:
            out_row[col] = data.get(day, "")
        rows.append(out_row)
        day = date.fromordinal(day.toordinal() + 1)

    out_path = base_dir / "govdata_merged_2025.csv"
    fieldnames = ["Date"] + [col for col, _ in series]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {out_path.name}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {', '.join(fieldnames)}")


if __name__ == "__main__":
    main()
