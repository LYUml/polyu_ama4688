# PV + Battery Sizing Optimization with Uncertainty Analysis

A Monte Carlo-based energy simulation framework for optimizing solar-battery microgrid sizing under load uncertainty.

## Overview

This project performs optimal battery capacity sizing for a **5 MW photovoltaic (PV) system** serving a **~484 MWh annual load**, targeting **99% system reliability**. The analysis includes:

- **Deterministic optimization**: Sizing based on average/median conditions
- **Stochastic analysis**: Monte Carlo sampling with load uncertainty (σ=0.357)
- **Comparison**: Quantifying the gap between deterministic and MC-based designs

### Key Results

| Method | Battery Capacity | Reliability | Notes |
|--------|------------------|-------------|-------|
| Deterministic | **570 kWh** | 99.04% | Conservative baseline |
| Monte Carlo | **600 kWh** | 99.08% | Accounts for demand variability |
| **Difference** | **+30 kWh (5.3%)** | — | MC requires larger buffer |

---

## Project Structure

```
ama4688project/
├── pv_battery_sizing.py           # Main sizing & optimization script
├── analyze_uncertainty.py          # Compute MC perturbation parameters (CV from NASA data)
├── requirements.txt                # Python 3.11 dependencies
├── weather_data/
│   └── govdata/
│       ├── daily_*.csv             # Source data (5 HKO stations)
│       └── govdata_merged_2025.csv # Processed: daily GHI [MJ/m²/day] for 365 days
├── eplus_run/
│   ├── inputs/
│   │   ├── model.idf               # Building model (reference only)
│   │   └── hongkong.epw            # Weather file (reference only)
│   └── results/
│       └── load_profiles/
│           ├── load_hourly_clean.csv      # Main input: 8760 hourly load [kW]
│           ├── load_profile_scenarios.csv # Auxiliary load data
│           └── load_shape_normalized.csv  # Normalized profiles (reference)
└── outputs/
    ├── deterministic_capacity_scan.csv    # 101 capacity points (0 → 1000 kWh, 10 kWh steps)
    ├── monte_carlo_capacity_scan.csv      # MC results: mean, std, P5, P95 per capacity
    └── figures/
        ├── deterministic_vs_mc_comparison.png  # Main comparison plot
        └── capacity_vs_ens.png            # Energy Not Supplied vs capacity
```

---

## Installation & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.11
- numpy, pandas, matplotlib

### 2. Run Main Optimization

```bash
python pv_battery_sizing.py
```

**Execution time:** ~5–10 minutes (101 capacity × 100 MC samples = 10,100 simulations)

**Outputs generated:**
- `outputs/deterministic_capacity_scan.csv` — Deterministic sizing results
- `outputs/monte_carlo_capacity_scan.csv` — MC results with confidence intervals
- `outputs/figures/deterministic_vs_mc_comparison.png` — Visualization

### 3. (Optional) Recompute Uncertainty Parameter

```bash
python analyze_uncertainty.py
```

This script extracts load variability statistics from NASA POWER data to calibrate the MC perturbation factor (currently σ=0.357, representing a 35.7% coefficient of variation).

---

## Method

### System Model

**PV:**
- Capacity: 5 MW
- Model: Temperature-adjusted power output from daily GHI
- Data: Hong Kong government (GovData) merged weather data

**Battery:**
- Discharge/charge power: 2,500 kW (bidirectional)
- Efficiency: 95% round-trip
- State of charge (SOC) limits: 10%–90%
- Dispatch: Greedy (charge excess PV, discharge to meet unmet demand)

**Load:**
- Time series: 8,760 hourly values for 2025 (365 days)
- Annual total: 484.3 MWh
- Average: 55.3 kW

### Reliability Metrics

- **Reliability:** Fraction of timesteps where demand is fully met
- **ENS (Energy Not Supplied):** Total unmet energy [kWh]
- **LOLP (Loss of Load Probability):** Fraction of hours with unmet demand

### Monte Carlo Approach

1. **Baseline:** Run deterministic simulation with median load
2. **Perturbation:** For each Monte Carlo sample, multiply hourly load by $L \sim \mathcal{N}(1.0, 0.357)$
3. **Sampling:** 100 replications per battery capacity point
4. **Results:** Aggregate reliability distribution (mean, σ, P5, P95)

**Uncertainty parameter calibration:**
- Derived from NASA POWER vs. GovData daily GHI comparison
- Coefficient of variation (CV) = 0.357 ≈ realistic load forecast error

---

## Data Sources

| Source | File | Usage |
|--------|------|-------|
| Hong Kong Observatory (GovData) | `govdata_merged_2025.csv` | Primary weather data (5 stations, daily GHI) |
| Load profile (EnergyPlus) | `load_hourly_clean.csv` | Demand time series (8,760 hours) |
| NASA POWER (archive) | — | Validation & uncertainty calibration only |

---

## Key Findings

✓ **MC-based sizing is ~5% larger** than deterministic baseline  
✓ **Confidence interval narrow** (P5–P95 < 0.3% spread) → 100 samples sufficient  
✓ **Deterministic underestimates risk** by ignoring load variability  
✓ **GovData superior to NASA** for local predictions (0.2% → realistic coverage)

---

## Author & Attribution

**Project:** AMA 4688 Microgrid Design (PolyU)  
**Repository:** [github.com/LYUml/polyu_ama4688](https://github.com/LYUml/polyu_ama4688)  
**Last updated:** 2026-04-14
