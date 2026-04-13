import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1) Config (edit here)
# =========================
# Use paths relative to project root
WEATHER_CSV = "weather_data/nasa_power/nasa_power_merged_2025.csv"
LOAD_CSV    = "eplus_run/results/load_profiles/load_hourly_clean.csv"
OUT_DIR     = "outputs"

# PV parameters
PV_CAP_KW = 500.0           # installed PV capacity
GHI_REF = 1000.0            # W/m2
TEMP_COEFF = -0.004         # per degC
TEMP_REF = 25.0

# Battery parameters
ETA_CH = 0.95
ETA_DIS = 0.95
SOC_MIN = 0.10
SOC_MAX = 0.90
P_CH_MAX_KW = 250.0
P_DIS_MAX_KW = 250.0
SOC0 = 0.50                  # initial SOC fraction

# Sizing target
R_TARGET = 0.99
CAPACITY_GRID = np.arange(0, 3001, 100)  # kWh

# =========================


def load_inputs(weather_csv, load_csv):
    """
    Load weather and load data, merge them to hourly frequency.
    
    Weather data: daily from NASA POWER
      - Column: ALLSKY_SFC_SW_DWN (MJ/m2/day) -> convert to GHI (W/m2) hourly
      - Column: T2M (temperature in C)
    
    Load data: hourly from EnergyPlus results
      - Column: load_kw
    """
    
    # Load weather data
    w = pd.read_csv(weather_csv)
    w["Date"] = pd.to_datetime(w["Date"])
    
    # Normalize dates to start from 1900-01-01 to match load data virtual year
    # Get unique dates and assign virtual dates
    unique_dates = sorted(w["Date"].unique())
    date_map = {old_date: pd.Timestamp("1900-01-01") + pd.Timedelta(days=i) 
                for i, old_date in enumerate(unique_dates)}
    w["Date_virtual"] = w["Date"].map(date_map)
    
    # Expand daily data to hourly by repeating 24 times per day
    # and distribute GHI across 24 hours (average approach)
    # First, get unique locations (LAT, LON) - if multiple, we'll average or pick one
    
    w_hourly_list = []
    
    for (lat, lon), group in w.groupby(["LAT", "LON"]):
        group = group.sort_values("Date_virtual").reset_index(drop=True)
        
        hourly_records = []
        for _, row in group.iterrows():
            date = row["Date_virtual"]
            # Create 24 hourly records for this day
            # Distribute daily GHI across 24 hours using a simple triangular profile
            ghi_daily_mj = row["ALLSKY_SFC_SW_DWN"]  # MJ/m2/day
            # Convert MJ/m2/day to daily integral: MJ/m2 * 1e6 J/MJ / 3600 s/h = (MJ/m2) * 277.78 Wh/m2
            # Split into 24 hours with triangular profile: peak at noon, 0 at night
            ghi_daily_wh = ghi_daily_mj * 277.78  # Wh/m2 per day
            
            for h in range(24):
                hour_dt = date + pd.Timedelta(hours=h)
                # Triangular GHI profile: 0 before 6am, peak at noon, 0 after 6pm
                hour_of_day = h
                if 6 <= hour_of_day <= 18:
                    if hour_of_day < 12:
                        profile_factor = (hour_of_day - 6) / 6  # 0 to 1 from 6am to noon
                    else:
                        profile_factor = (18 - hour_of_day) / 6  # 1 to 0 from noon to 6pm
                    # Total daylight hours in profile: 12 hours (6am to 6pm)
                    # With triangular: integral is 0.5 * 12 = 6 hour-equivalents
                    ghi_this_hour = (ghi_daily_wh / 6) * profile_factor  # W/m2
                else:
                    ghi_this_hour = 0.0
                
                hourly_records.append({
                    "datetime": hour_dt,
                    "ghi": max(0, ghi_this_hour),
                    "temp_air": row["T2M"]
                })
        
        w_hourly_list.append(pd.DataFrame(hourly_records))
    
    if w_hourly_list:
        w_hourly = pd.concat(w_hourly_list, ignore_index=True)
        # If multiple locations, average by datetime
        w_hourly = w_hourly.groupby("datetime").agg({
            "ghi": "mean",
            "temp_air": "mean"
        }).reset_index()
    else:
        w_hourly = pd.DataFrame(columns=["datetime", "ghi", "temp_air"])
    
    # Load load data
    l = pd.read_csv(load_csv)
    l["datetime"] = pd.to_datetime(l["datetime"])
    
    # Merge on datetime
    df = pd.merge(w_hourly, l, on="datetime", how="inner").sort_values("datetime").reset_index(drop=True)
    df = df[["datetime", "ghi", "temp_air", "load_kw"]].copy()
    
    # Basic cleaning
    df["ghi"] = df["ghi"].fillna(0).clip(lower=0)
    df["load_kw"] = df["load_kw"].fillna(df["load_kw"].mean()).clip(lower=0)
    df["temp_air"] = df["temp_air"].interpolate(limit_direction="both").fillna(df["temp_air"].mean())
    
    print(f"Loaded data: {len(df)} hourly records from {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df


def pv_power_kw(ghi, temp_air, pv_cap_kw, ghi_ref=1000, temp_coeff=-0.004, temp_ref=25):
    # simple linear temperature correction
    p = pv_cap_kw * (ghi / ghi_ref) * (1 + temp_coeff * (temp_air - temp_ref))
    return np.clip(p, 0, pv_cap_kw)


def simulate_once(df, batt_kwh):
    n = len(df)
    dt = 1.0  # hour

    e_min = SOC_MIN * batt_kwh
    e_max = SOC_MAX * batt_kwh
    e = SOC0 * batt_kwh if batt_kwh > 0 else 0.0

    soc_series = np.zeros(n)
    unmet = np.zeros(n)

    pv = pv_power_kw(df["ghi"].values, df["temp_air"].values, PV_CAP_KW)

    for t in range(n):
        load = df.at[t, "load_kw"]
        net = pv[t] - load  # + surplus, - deficit

        if net >= 0:
            # charge
            ch_possible = min(net * dt, P_CH_MAX_KW * dt)
            headroom = max(0.0, e_max - e)
            ch = min(ch_possible * ETA_CH, headroom)
            e += ch
            unmet[t] = 0.0
        else:
            # discharge
            deficit = (-net) * dt
            dis_need_from_batt = min(deficit, P_DIS_MAX_KW * dt)
            available_to_grid = max(0.0, (e - e_min) * ETA_DIS)
            served = min(dis_need_from_batt, available_to_grid)

            # battery energy drop
            if ETA_DIS > 0:
                e -= served / ETA_DIS

            unmet_energy = deficit - served
            unmet[t] = max(0.0, unmet_energy)

        e = min(max(e, e_min if batt_kwh > 0 else 0.0), e_max if batt_kwh > 0 else 0.0)
        soc_series[t] = (e / batt_kwh) if batt_kwh > 0 else 0.0

    total_load = (df["load_kw"].values * dt).sum()
    ens = unmet.sum()
    lolp = (unmet > 1e-9).mean()
    reliability = 1 - ens / total_load if total_load > 0 else 1.0

    return {
        "reliability": reliability,
        "ens_kwh": ens,
        "lolp": lolp,
        "soc": soc_series,
        "pv_kw": pv,
        "unmet_kwh": unmet
    }


def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = load_inputs(WEATHER_CSV, LOAD_CSV)

    rows = []
    for b in CAPACITY_GRID:
        res = simulate_once(df, batt_kwh=float(b))
        rows.append({
            "battery_kwh": b,
            "reliability": res["reliability"],
            "ens_kwh": res["ens_kwh"],
            "lolp": res["lolp"]
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out / "baseline_capacity_scan.csv", index=False)

    # minimum feasible capacity
    feasible = result_df[result_df["reliability"] >= R_TARGET]
    b_star = float(feasible["battery_kwh"].min()) if not feasible.empty else np.nan

    # plot 1: capacity vs reliability
    plt.figure(figsize=(7, 4.5))
    plt.plot(result_df["battery_kwh"], result_df["reliability"], marker="o")
    plt.axhline(R_TARGET, linestyle="--", label=f"Target ({R_TARGET})")
    if not np.isnan(b_star):
        plt.axvline(b_star, linestyle=":", label=f"B* = {b_star:.0f} kWh")
    plt.xlabel("Battery capacity (kWh)")
    plt.ylabel("Reliability")
    plt.title("Capacity vs Reliability (Baseline)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "figures" / "capacity_vs_reliability.png", dpi=200)
    plt.close()

    # plot 2: capacity vs ENS
    plt.figure(figsize=(7, 4.5))
    plt.plot(result_df["battery_kwh"], result_df["ens_kwh"], marker="o", color="orange")
    plt.xlabel("Battery capacity (kWh)")
    plt.ylabel("ENS (kWh)")
    plt.title("Capacity vs ENS (Baseline)")
    plt.tight_layout()
    plt.savefig(out / "figures" / "capacity_vs_ens.png", dpi=200)
    plt.close()

    # run once at b_star for SOC example
    if not np.isnan(b_star):
        res_star = simulate_once(df, batt_kwh=b_star)
        tmp = df.copy()
        tmp["soc"] = res_star["soc"]
        tmp_week = tmp.iloc[:24 * 7]

        plt.figure(figsize=(8, 4))
        plt.plot(tmp_week["datetime"], tmp_week["soc"], linewidth=2)
        plt.xticks(rotation=30)
        plt.ylim(0, 1)
        plt.ylabel("SOC")
        plt.xlabel("Time")
        plt.title(f"SOC (First Week) at B*={b_star:.0f} kWh")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "figures" / "soc_first_week.png", dpi=200)
        plt.close()

    print("Done.")
    print(f"Saved: {out / 'baseline_capacity_scan.csv'}")
    print(f"Min battery capacity for target reliability ({R_TARGET}): {b_star} kWh")


if __name__ == "__main__":
    main()
