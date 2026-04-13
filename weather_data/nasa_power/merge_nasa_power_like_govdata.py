import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

KEY_COLS = ["LAT", "LON", "YEAR", "MO", "DY"]


def read_power_csv(path: Path) -> Tuple[List[Dict[str, str]], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        end_header_idx = lines.index("-END HEADER-")
    except ValueError as exc:
        raise ValueError(f"-END HEADER- not found in {path}") from exc

    data_lines = lines[end_header_idx + 1 :]
    reader = csv.DictReader(data_lines)
    rows = list(reader)

    fields = reader.fieldnames or []
    metric_cols = [c for c in fields if c not in KEY_COLS]
    if len(metric_cols) != 1:
        raise ValueError(f"Expected one metric column in {path.name}, got {metric_cols}")

    return rows, metric_cols[0]


def build_groups(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str, str], List[Dict[str, str]]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (row["LAT"], row["YEAR"], row["MO"], row["DY"])
        groups.setdefault(key, []).append(row)
    return groups


def to_date_str(row: Dict[str, str]) -> str:
    return f"{int(row['YEAR']):04d}-{int(row['MO']):02d}-{int(row['DY']):02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge NASA POWER CSV files by same LAT+date and nearest LON."
    )
    parser.add_argument("--year", type=int, default=2025, help="Target year, e.g. 2025")
    parser.add_argument(
        "--out",
        default="",
        help="Output file name (default: nasa_power_merged_<year>.csv)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    y = args.year
    pattern = f"POWER_Regional_Daily_{y}0101_{y}1231*.csv"
    files = sorted(base_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    parsed = []
    for file_path in files:
        rows, metric = read_power_csv(file_path)
        parsed.append((file_path, metric, rows))

    base_candidates = [x for x in parsed if x[1] == "ALLSKY_SFC_SW_DWN"]
    if len(base_candidates) != 1:
        names = ", ".join(p.name for p, _, _ in parsed)
        raise ValueError(f"Need exactly one ALLSKY base file. Found in: {names}")

    base_file, base_metric, base_rows = base_candidates[0]

    lookup_info = []
    for file_path, metric, rows in parsed:
        if file_path == base_file:
            continue
        lookup_info.append((metric, build_groups(rows)))

    out_name = args.out.strip() or f"nasa_power_merged_{y}.csv"
    out_path = base_dir / out_name

    output_cols = ["Date", "LAT", "LON", base_metric] + [m for m, _ in lookup_info]
    merged_rows: List[Dict[str, str]] = []
    match_count = {m: 0 for m, _ in lookup_info}

    for base in base_rows:
        out_row = {
            "Date": to_date_str(base),
            "LAT": base["LAT"],
            "LON": base["LON"],
            base_metric: base.get(base_metric, ""),
        }

        lat_date_key = (base["LAT"], base["YEAR"], base["MO"], base["DY"])
        target_lon = float(base["LON"])

        for metric, groups in lookup_info:
            candidates = groups.get(lat_date_key, [])
            value = ""
            if candidates:
                best = min(candidates, key=lambda x: abs(float(x["LON"]) - target_lon))
                value = best.get(metric, "")
                if value != "":
                    match_count[metric] += 1
            out_row[metric] = value

        merged_rows.append(out_row)

    merged_rows.sort(key=lambda r: (r["Date"], float(r["LAT"]), float(r["LON"])))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_cols)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Created: {out_path.name}")
    print(f"Rows(out): {len(merged_rows)}")
    print(f"Columns: {', '.join(output_cols)}")
    for metric, _ in lookup_info:
        print(f"Matched {metric}: {match_count[metric]}")


if __name__ == "__main__":
    main()
