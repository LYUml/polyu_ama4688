import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

WEATHER_CSV = "weather_data/govdata/govdata_merged_2025.csv"
LOAD_CSV    = "eplus_run/results/load_profiles/load_hourly_clean.csv"
OUT_DIR     = "outputs"

PV_CAP_KW = 5000.0
GHI_REF = 1000.0
TEMP_COEFF = -0.004
TEMP_REF = 25.0

ETA_CH = 0.95
ETA_DIS = 0.95
SOC_MIN = 0.10
SOC_MAX = 0.90
P_CH_MAX_KW = 2500.0
P_DIS_MAX_KW = 2500.0
SOC0 = 0.50

R_TARGET = 0.99
CAPACITY_GRID = np.arange(0, 15001, 500)


def load_inputs(weather_csv, load_csv):
    """Load weather (govdata, 2025) and load data, merge to hourly."""
    w = pd.read_csv(weather_csv)
    w["Date"] = pd.to_datetime(w["Date"])
    
    l = pd.read_csv(load_csv)
    l["datetime"] = pd.to_datetime(l["datetime"])
    
    l["day_of_year"] = l["datetime"].dt.dayofyear
    l["hour_of_day"] = l["datetime"].dt.hour
    l["datetime_2025"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(l["day_of_year"] - 1, unit="D") + pd.to_timedelta(l["hour_of_day"], unit="h")
    
    w_hourly_list = []
    for idx, row in w.iterrows():
        date = row["Date"]
        gsi_mj_per_day = row["KP_GSR"]
        ghi_wh_per_day = (gsi_mj_per_day / 3.6) * 1000
        
        hourly_records = []
        for h in range(24):
            hour_dt = date + pd.Timedelta(hours=h)
            if 6 <= h < 18:
                hour_norm = (h - 6) / 12
                profile = 2 * hour_norm if hour_norm <= 0.5 else 2 * (1 - hour_norm)
                ghi_wh_this_hour = (ghi_wh_per_day / 6) * profile
            else:
                ghi_wh_this_hour = 0.0
            
            hourly_records.append({
                "datetime": hour_dt,
                "ghi": max(0, ghi_wh_this_hour),
                "temp_air": 23.0
            })
        
        w_hourly_list.append(pd.DataFrame(hourly_records))
    
    if w_hourly_list:
        w_hourly = pd.concat(w_hourly_list, ignore_index=True)
    else:
        w_hourly = pd.DataFrame(columns=["datetime", "ghi", "temp_air"])
    
    w_hourly.rename(columns={"datetime": "datetime_2025"}, inplace=True)
    df = pd.merge(w_hourly, l[["datetime_2025", "load_kw"]], on="datetime_2025", how="inner")
    df.rename(columns={"datetime_2025": "datetime"}, inplace=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df[["datetime", "ghi", "temp_air", "load_kw"]].copy()
    
    df["ghi"] = df["ghi"].fillna(0).clip(lower=0)
    df["load_kw"] = df["load_kw"].fillna(df["load_kw"].mean()).clip(lower=0)
    df["temp_air"] = df["temp_air"].interpolate(limit_direction="both").fillna(df["temp_air"].mean())
    
    return df


def pv_power_kw(ghi, temp_air, pv_cap_kw, ghi_ref=1000, temp_coeff=-0.004, temp_ref=25):
    p = pv_cap_kw * (ghi / ghi_ref) * (1 + temp_coeff * (temp_air - temp_ref))
    return np.clip(p, 0, pv_cap_kw)


def simulate_once(df, batt_kwh):
    n = len(df)
    dt = 1.0
    
    e_min = SOC_MIN * batt_kwh
    e_max = SOC_MAX * batt_kwh
    e = SOC0 * batt_kwh if batt_kwh > 0 else 0.0
    
    soc_series = np.zeros(n)
    unmet = np.zeros(n)
    pv = pv_power_kw(df["ghi"].values, df["temp_air"].values, PV_CAP_KW)
    
    for t in range(n):
        load = df.at[t, "load_kw"]
        net = pv[t] - load
        
        if net >= 0:
            ch_possible = min(net * dt, P_CH_MAX_KW * dt)
            headroom = max(0.0, e_max - e)
            ch = min(ch_possible * ETA_CH, headroom)
            e += ch
            unmet[t] = 0.0
        else:
            deficit = (-net) * dt
            dis_need = min(deficit, P_DIS_MAX_KW * dt)
            available = max(0.0, (e - e_min) * ETA_DIS) if batt_kwh > 0 else 0.0
            served = min(dis_need, available)
            
            if ETA_DIS > 0:
                e -= served / ETA_DIS
            
            unmet[t] = max(0.0, deficit - served)
        
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
    
    print("Loading data...")
    df = load_inputs(WEATHER_CSV, LOAD_CSV)
    print(f"Data: {len(df)} hours, {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    
    print("Running capacity scan...")
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
    
    feasible = result_df[result_df["reliability"] >= R_TARGET]
    b_star = float(feasible["battery_kwh"].min()) if not feasible.empty else np.nan
    
    plt.figure(figsize=(8, 5))
    plt.plot(result_df["battery_kwh"], result_df["reliability"], marker="o", linewidth=2)
    plt.axhline(R_TARGET, linestyle="--", color="red", label=f"Target ({R_TARGET:.0%})")
    if not np.isnan(b_star):
        plt.axvline(b_star, linestyle=":", color="green", label=f"B* = {b_star:,.0f} kWh")
    plt.xlabel("Battery Capacity (kWh)")
    plt.ylabel("Reliability")
    plt.title("System Reliability vs Battery Capacity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "figures" / "capacity_vs_reliability.png", dpi=200)
    plt.close()
    
    plt.figure(figsize=(8, 5))
    plt.plot(result_df["battery_kwh"], result_df["ens_kwh"], marker="o", color="orange", linewidth=2)
    plt.xlabel("Battery Capacity (kWh)")
    plt.ylabel("Energy Not Supplied (kWh/year)")
    plt.title("System ENS vs Battery Capacity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "figures" / "capacity_vs_ens.png", dpi=200)
    plt.close()
    
    if not np.isnan(b_star):
        res_star = simulate_once(df, batt_kwh=b_star)
        tmp = df.copy()
        tmp["soc"] = res_star["soc"]
        tmp_week = tmp.iloc[:24 * 7]
        
        plt.figure(figsize=(10, 4))
        plt.plot(tmp_week["datetime"], tmp_week["soc"], linewidth=2, color="darkblue")
        plt.fill_between(tmp_week["datetime"], 0, tmp_week["soc"], alpha=0.3, color="blue")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.ylabel("SOC")
        plt.xlabel("Date")
        plt.title(f"Battery State of Charge (First Week, B*={b_star:,.0f} kWh)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "figures" / "soc_first_week.png", dpi=200)
        plt.close()
    
    print(f"Results saved to {out / 'baseline_capacity_scan.csv'}")
    if not np.isnan(b_star):
        print(f"Minimum battery for {R_TARGET:.0%} reliability: {b_star:,.0f} kWh")
    else:
        print(f"Target reliability {R_TARGET:.0%} not achievable with given parameters")


if __name__ == "__main__":
    main()
