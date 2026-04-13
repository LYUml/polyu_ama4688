import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEATHER_CSV = PROJECT_ROOT / "weather_data" / "govdata" / "govdata_merged_2025.csv"
LOAD_CSV = PROJECT_ROOT / "eplus_run" / "results" / "load_profiles" / "load_hourly_clean.csv"
OUT_DIR = PROJECT_ROOT / "outputs"

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
CAPACITY_GRID = np.arange(0, 1001, 10)

# Monte Carlo parameters
N_MC_SAMPLES = 100
RANDOM_SEED = 42
LOAD_UNCERTAINTY_STD = 0.357  # from NASA Power CV analysis


def load_inputs(weather_csv, load_csv):
    w = pd.read_csv(weather_csv)
    w["Date"] = pd.to_datetime(w["Date"])
    
    l = pd.read_csv(load_csv)
    l["datetime"] = pd.to_datetime(l["datetime"])
    
    l["day_of_year"] = l["datetime"].dt.dayofyear
    l["hour_of_day"] = l["datetime"].dt.hour
    l["datetime_2025"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(l["day_of_year"] - 1, unit="D") + pd.to_timedelta(l["hour_of_day"], unit="h")
    
    w_hourly_list = []
    for _, row in w.iterrows():
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


def generate_mc_sample(df, seed):
    """
    Generate Monte Carlo sample by perturbing load with uncertainty factor.
    Preserves temporal structure while introducing realistic load variability.
    """
    np.random.seed(seed)
    df_sample = df.copy()
    # Apply multiplicative perturbation to load based on uncertainty from NASA data
    perturbation = np.random.normal(1.0, LOAD_UNCERTAINTY_STD, len(df))
    df_sample['load_kw'] = df_sample['load_kw'] * perturbation
    df_sample['load_kw'] = df_sample['load_kw'].clip(lower=0)
    return df_sample


def run_monte_carlo(df, batt_kwh, n_samples=N_MC_SAMPLES):
    """
    对单一电池容量运行蒙特卡洛模拟。
    返回可靠性的均值、标准差和分布。
    """
    reliability_samples = []
    
    for i in range(n_samples):
        df_sample = generate_mc_sample(df, seed=RANDOM_SEED + i)
        res = simulate_once(df_sample, batt_kwh)
        reliability_samples.append(res["reliability"])
    
    reliability_samples = np.array(reliability_samples)
    
    return {
        "mean": np.mean(reliability_samples),
        "std": np.std(reliability_samples),
        "samples": reliability_samples,
        "p05": np.percentile(reliability_samples, 5),
        "p95": np.percentile(reliability_samples, 95)
    }


def main():
    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_inputs(WEATHER_CSV, LOAD_CSV)
    print(f"Data: {len(df)} hours, {df['datetime'].min().date()} to {df['datetime'].max().date()}")

    # ===== 确定性基准情景 =====
    print("\n=== Running deterministic baseline scenario ===")
    print("Running capacity scan (deterministic)...")
    rows_det = []
    for b in CAPACITY_GRID:
        res = simulate_once(df, batt_kwh=float(b))
        rows_det.append({
            "battery_kwh": b,
            "reliability": res["reliability"],
            "ens_kwh": res["ens_kwh"],
            "lolp": res["lolp"]
        })

    result_df_det = pd.DataFrame(rows_det)
    result_df_det.to_csv(out / "deterministic_capacity_scan.csv", index=False)

    feasible_det = result_df_det[result_df_det["reliability"] >= R_TARGET]
    b_star_det = float(feasible_det["battery_kwh"].min()) if not feasible_det.empty else np.nan
    
    print(f"Deterministic result: B* = {b_star_det:.0f} kWh")

    # ===== 蒙特卡洛情景 =====
    print("\n=== Running Monte Carlo scenario ===")
    print(f"Running capacity scan with {N_MC_SAMPLES} MC samples...")
    rows_mc = []
    for b in CAPACITY_GRID:
        print(f"  Battery capacity: {b:.0f} kWh")
        mc_result = run_monte_carlo(df, batt_kwh=float(b), n_samples=N_MC_SAMPLES)
        rows_mc.append({
            "battery_kwh": b,
            "reliability_mean": mc_result["mean"],
            "reliability_std": mc_result["std"],
            "reliability_p05": mc_result["p05"],
            "reliability_p95": mc_result["p95"]
        })

    result_df_mc = pd.DataFrame(rows_mc)
    result_df_mc.to_csv(out / "monte_carlo_capacity_scan.csv", index=False)

    feasible_mc = result_df_mc[result_df_mc["reliability_mean"] >= R_TARGET]
    b_star_mc = float(feasible_mc["battery_kwh"].min()) if not feasible_mc.empty else np.nan
    
    print(f"Monte Carlo result: B* = {b_star_mc:.0f} kWh (mean reliability >= {R_TARGET:.0%})")

    # ===== 绘图：对比两种情景 =====
    plt.figure(figsize=(10, 6))
    plt.plot(result_df_det["battery_kwh"], result_df_det["reliability"], 
             marker="x", markersize=6, linewidth=2, label="Deterministic", color="blue")
    plt.plot(result_df_mc["battery_kwh"], result_df_mc["reliability_mean"], 
             marker="x", markersize=6, linewidth=2, label="MC Mean", color="red")
    plt.fill_between(result_df_mc["battery_kwh"], 
                     result_df_mc["reliability_p05"], 
                     result_df_mc["reliability_p95"],
                     alpha=0.2, color="red", label="MC 5-95% CI")
    plt.axhline(R_TARGET, linestyle="--", color="black", alpha=0.5, label=f"Target ({R_TARGET:.0%})")
    if not np.isnan(b_star_det):
        plt.axvline(b_star_det, linestyle=":", color="blue", alpha=0.5, label=f"B* Det = {b_star_det:.0f} kWh")
    if not np.isnan(b_star_mc):
        plt.axvline(b_star_mc, linestyle=":", color="red", alpha=0.5, label=f"B* MC = {b_star_mc:.0f} kWh")
    plt.xlabel("Battery Capacity (kWh)", fontsize=11)
    plt.ylabel("Reliability", fontsize=11)
    plt.title("Deterministic vs. Monte Carlo: System Reliability vs Battery Capacity", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "figures" / "deterministic_vs_mc.png", dpi=200)
    plt.close()

    # ===== 确定性情景的旧图 =====
    plt.figure(figsize=(8, 5))
    plt.plot(result_df_det["battery_kwh"], result_df_det["reliability"], marker="x", markersize=6, linewidth=2)
    plt.axhline(R_TARGET, linestyle="--", color="red", label=f"Target ({R_TARGET:.0%})")
    if not np.isnan(b_star_det):
        plt.axvline(b_star_det, linestyle=":", color="green", label=f"B* = {b_star_det:,.0f} kWh")
    plt.xlabel("Battery Capacity (kWh)")
    plt.ylabel("Reliability")
    plt.title("System Reliability vs Battery Capacity (Deterministic)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "figures" / "capacity_vs_reliability_det.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(result_df_det["battery_kwh"], result_df_det["ens_kwh"], marker="x", markersize=6, color="orange", linewidth=2)
    plt.xlabel("Battery Capacity (kWh)")
    plt.ylabel("Energy Not Supplied (kWh/year)")
    plt.title("System ENS vs Battery Capacity (Deterministic)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "figures" / "capacity_vs_ens.png", dpi=200)
    plt.close()

    if not np.isnan(b_star_det):
        res_star = simulate_once(df, batt_kwh=b_star_det)
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
        plt.title(f"Battery State of Charge (First Week, B*={b_star_det:,.0f} kWh)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "figures" / "soc_first_week.png", dpi=200)
        plt.close()

    print(f"\nResults saved to {out}")
    print(f"Deterministic: B* = {b_star_det:.0f} kWh")
    print(f"Monte Carlo:   B* = {b_star_mc:.0f} kWh")


if __name__ == "__main__":
    main()