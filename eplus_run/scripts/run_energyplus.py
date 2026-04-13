import subprocess
from pathlib import Path
import pandas as pd

ENERGYPLUS_EXE = r"D:\EnergyPlusV26-1-0\energyplus.exe" 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IDF_PATH = str(PROJECT_ROOT / "inputs" / "models" / "model.idf")
EPW_PATH = str(PROJECT_ROOT / "inputs" / "weather" / "hongkong.epw")
OUT_DIR = str(PROJECT_ROOT / "results" / "energyplus_raw")



def run_energyplus(energyplus_exe: str, idf_path: str, epw_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        energyplus_exe,
        "-w", epw_path,
        "-d", out_dir,
        "-r",
        idf_path
    ]

    print("Running EnergyPlus...")
    print("Command:", " ".join([f'"{c}"' if " " in c else c for c in cmd]))

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("\n=== EnergyPlus STDOUT ===")
    print(result.stdout[-2000:] if result.stdout else "(no stdout)")

    print("\n=== EnergyPlus STDERR ===")
    print(result.stderr[-2000:] if result.stderr else "(no stderr)")

    if result.returncode != 0:
        raise RuntimeError(
            f"EnergyPlus failed (return code={result.returncode}). "
            f"Check eplusout.err in: {out_dir}"
        )

    print("EnergyPlus finished successfully.")


def find_best_load_column(df: pd.DataFrame):
    cols = list(df.columns)

    # Priority: Power (W) > Energy (J)
    priority_patterns = [
        "Facility Total Electricity Demand Power",  # W
        "Electricity:Facility [J](Hourly)",         # J
        "Electricity:Facility",                     # alternative format
        "Facility Total Electric Demand Power",     # variant
        "Electricity:Building",                     # variant
    ]

    # Check priority patterns first
    for p in priority_patterns:
        for c in cols:
            if p.lower() in c.lower():
                return c

    # Fallback: find column with Electricity + Hourly
    for c in cols:
        lc = c.lower()
        if ("electricity" in lc) and ("hourly" in lc):
            return c

    # Return None if not found
    return None


def convert_to_load_kw(df: pd.DataFrame, col: str) -> pd.Series:
    lc = col.lower()

    # Power in W
    if "[w]" in lc or "power" in lc:
        return df[col] / 1000.0

    # Hourly energy in J
    if "[j]" in lc or "joule" in lc:
        # J -> kWh (1 hour per row)
        kwh = df[col] / 3_600_000.0
        # Average kW equals kWh when Δt=1h
        return kwh

    # Unit uncertain: assume W -> kW
    print(f"[WARN] Unit not clearly identified for column: {col}. Assume W -> kW.")
    return df[col] / 1000.0


def extract_load(out_dir: str):
    csv_path = Path(out_dir) / "eplusmtr.csv"
    err_path = Path(out_dir) / "eplusout.err"

    if not csv_path.exists():
        msg = f"Cannot find {csv_path}."
        if err_path.exists():
            msg += f" Please check error log: {err_path}"
        raise FileNotFoundError(msg)

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    # EnergyPlus CSV usually has Date/Time column
    # Auto-find time column
    time_col_candidates = [c for c in df.columns if c.lower() in ["date/time", "datetime", "date time", "time"]]
    time_col = time_col_candidates[0] if time_col_candidates else None

    load_col = find_best_load_column(df)
    if load_col is None:
        raise ValueError(
            "Could not find a suitable electricity/load column in eplusout.csv.\n"
            f"Available columns sample:\n{df.columns[:30].tolist()}"
        )

    print(f"Selected load column: {load_col}")
    df["load_kw"] = convert_to_load_kw(df, load_col)

    out_df = pd.DataFrame()
    if time_col:
        out_df["datetime"] = df[time_col]
    else:
        # Use row index if no datetime column
        out_df["datetime"] = range(1, len(df) + 1)
        print("[WARN] No datetime column found. Using row index as datetime.")

    out_df["load_kw"] = out_df["load_kw"] = df["load_kw"].clip(lower=0)

    out_file = Path(out_dir) / "load_hourly.csv"
    out_df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

    # Quick QA check
    print("\n=== Quick QA ===")
    print(f"Rows: {len(out_df)}")
    print(f"Min load_kw: {out_df['load_kw'].min():.4f}")
    print(f"Mean load_kw: {out_df['load_kw'].mean():.4f}")
    print(f"Max load_kw: {out_df['load_kw'].max():.4f}")


def main():
    # Run simulation
    run_energyplus(ENERGYPLUS_EXE, IDF_PATH, EPW_PATH, OUT_DIR)

    # Extract load profile
    extract_load(OUT_DIR)

    print("\nDone. You can now use load_hourly.csv for your simulation.")


if __name__ == "__main__":
    main()