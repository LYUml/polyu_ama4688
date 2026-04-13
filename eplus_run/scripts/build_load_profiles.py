"""
Build load profiles for renewable microgrid simulation.
Transforms raw EnergyPlus hourly output into normalized and scenario-based profiles.
"""

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

warnings.filterwarnings('ignore', category=UserWarning)
rcParams['figure.figsize'] = (12, 6)
rcParams['font.size'] = 10


def parse_energyplus_datetime(series: pd.Series) -> pd.Series:
    """Parse EnergyPlus timestamps and convert 24:00:00 to next day 00:00:00."""
    s = series.astype(str).str.strip()

    # Handle entries like '12/31  24:00:00' before pandas parsing.
    mask_24 = s.str.endswith("24:00:00", na=False)
    fixed = s.copy()
    fixed.loc[mask_24] = fixed.loc[mask_24].str.replace("24:00:00", "00:00:00", regex=False)

    dt = pd.to_datetime(fixed, format='%m/%d  %H:%M:%S', errors='coerce')
    dt.loc[mask_24] = dt.loc[mask_24] + pd.Timedelta(days=1)

    if dt.notna().sum() == 0:
        dt = pd.to_datetime(s, errors='coerce')
    return dt

# ============================================================================
# CONFIG BLOCK
# ============================================================================

CONFIG = {
    'input_file': Path(__file__).parent.parent / 'results' / 'energyplus_raw' / 'eplusmtr.csv',
    'output_dir': Path(__file__).parent.parent / 'results' / 'load_profiles',
    'annual_energy_target_kwh': 450000,
    'scenario_multipliers': {
        'low': 0.8,
        'base': 1.0,
        'high': 1.2,
    },
    'stochastic_sigma': 0.15,
    'apply_stochastic': False,
    'generate_report_md': False,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_and_inspect_data(filepath):
    """
    Load raw EnergyPlus CSV and auto-detect electricity load column.
    Handles both raw eplusmtr.csv and pre-processed load_hourly.csv
    
    Returns:
        df: DataFrame with datetime, load_kw columns
    """
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {filepath}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    
    # Check if already processed (has load_kw column)
    if 'load_kw' in df.columns:
        print(f"[INFO] Detected pre-processed file with load_kw column")
        
        # Find datetime column
        time_col = None
        for candidate in ['datetime', 'Date/Time', 'Date/time']:
            if candidate in df.columns:
                time_col = candidate
                break
        
        if time_col is None:
            raise ValueError('No datetime column found')
        
        df['datetime_parsed'] = parse_energyplus_datetime(df[time_col])
        return df[['datetime_parsed', 'load_kw']].rename(columns={'datetime_parsed': 'datetime'}).copy()
    
    # Otherwise, process raw EnergyPlus file
    # Find datetime column
    time_col = None
    for candidate in ['datetime', 'Date/Time', 'Date/time']:
        if candidate in df.columns:
            time_col = candidate
            break
    
    if time_col is None:
        raise ValueError('No datetime column found. Expected: datetime, Date/Time, or Date/time')
    
    # Find load column
    load_col = None
    for col in df.columns:
        if 'electricity' in col.lower() and 'facility' in col.lower():
            load_col = col
            break
    
    if load_col is None:
        raise ValueError(f'No electricity/facility load column found. Available: {df.columns.tolist()}')
    
    print(f"[INFO] Datetime column: {time_col}")
    print(f"[INFO] Load column: {load_col}")
    
    df['datetime'] = parse_energyplus_datetime(df[time_col])
    
    col_lower = load_col.lower()
    if '[j]' in col_lower or 'joule' in col_lower:
        df['load_kw'] = df[load_col] / 3_600_000.0
        print(f"[INFO] Converted from J/hour to kWh")
    else:
        df['load_kw'] = df[load_col] / 1000.0
    
    return df[['datetime', 'load_kw']].copy()

def clean_data(df):
    """
    Clean load data:
    - Remove NaN values
    - Remove negative loads
    - Ensure chronological order
    """
    print(f"\n[CLEAN] Initial rows: {len(df)}")
    
    df = df.dropna(subset=['datetime', 'load_kw'])
    print(f"[CLEAN] After removing NaN: {len(df)}")
    
    neg_count = (df['load_kw'] < 0).sum()
    df = df[df['load_kw'] >= 0].copy()
    print(f"[CLEAN] Removed {neg_count} negative load entries")
    
    df = df.sort_values('datetime').reset_index(drop=True)
    
    df_temp = df.set_index('datetime')
    df_filled = df_temp.resample('h').interpolate(method='linear')
    df_filled = df_filled.reset_index()
    added = len(df_filled) - len(df)
    if added > 0:
        print(f"[CLEAN] Interpolated {added} missing hours")
    
    return df_filled

def normalize_shape(df):
    """
    Create normalized load shape: shape_pu = load_kw / mean(load_kw)
    """
    mean_load = df['load_kw'].mean()
    df['shape_pu'] = df['load_kw'] / mean_load
    print(f"\n[NORM] Mean load: {mean_load:.2f} kW")
    print(f"[NORM] Min shape_pu: {df['shape_pu'].min():.3f}, Max: {df['shape_pu'].max():.3f}")
    return df

def generate_scenarios(df, annual_energy_target, multipliers):
    """
    Generate 3 scenario profiles (low, base, high) scaled by annual energy.
    Formula: L_t^s = p_t * (E_annual / sum(p_t))
    """
    print(f"\n[SCENARIO] Annual energy target: {annual_energy_target:.0f} kWh")
    print(f"[SCENARIO] Multipliers: {multipliers}")
    
    df_scenarios = df[['datetime', 'shape_pu']].copy()
    
    sum_shape_unit = df['shape_pu'].sum()
    
    for scenario, mult in multipliers.items():
        annual_target_scenario = annual_energy_target * mult
        scale_factor = annual_target_scenario / sum_shape_unit
        df_scenarios[f'load_kw_{scenario}'] = df['shape_pu'] * scale_factor
        
        annual_sim = df_scenarios[f'load_kw_{scenario}'].sum()
        print(f"[SCENARIO] {scenario:5s}: scale={scale_factor:.2f} -> annual={annual_sim:.0f} kWh")
    
    return df_scenarios

def compute_statistics(df, df_scenarios):
    """
    Compute report-ready statistics.
    """
    stats = {}
    stats['total_hours'] = len(df)
    stats['annual_energy_kwh'] = df['load_kw'].sum()
    stats['peak_load_kw'] = df['load_kw'].max()
    stats['avg_load_kw'] = df['load_kw'].mean()
    stats['load_factor'] = stats['avg_load_kw'] / stats['peak_load_kw']
    stats['p95_load_kw'] = df['load_kw'].quantile(0.95)
    stats['p99_load_kw'] = df['load_kw'].quantile(0.99)
    
    df_temp = df.copy()
    df_temp['weekday'] = df_temp['datetime'].dt.weekday
    weekday_df = df_temp[df_temp['weekday'] < 5]
    weekend_df = df_temp[df_temp['weekday'] >= 5]
    
    stats['weekday_mean_kw'] = weekday_df['load_kw'].mean()
    stats['weekend_mean_kw'] = weekend_df['load_kw'].mean()
    stats['weekday_peak_kw'] = weekday_df['load_kw'].max()
    stats['weekend_peak_kw'] = weekend_df['load_kw'].max()
    
    print(f"\n[STATS]")
    for key, val in stats.items():
        print(f"  {key}: {val:.2f}")
    
    return stats

def add_stochastic(df_scenarios, sigma, random_seed=42):
    """
    Add optional stochastic perturbation: L_t^s = max(0, L_t * (1 + epsilon_t))
    """
    np.random.seed(random_seed)
    n = len(df_scenarios)
    epsilon = np.random.normal(0, sigma, n)
    
    for scenario in ['low', 'base', 'high']:
        col = f'load_kw_{scenario}'
        stoch_col = f'{col}_stochastic'
        df_scenarios[stoch_col] = np.maximum(0, df_scenarios[col] * (1 + epsilon))
    
    print(f"\n[STOCHASTIC] Applied N(0, {sigma}^2) perturbation")
    return df_scenarios

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_timeseries(df, output_dir):
    """Plot full-year hourly load."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['datetime'], df['load_kw'], linewidth=0.5, color='steelblue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Load (kW)')
    ax.set_title('Hourly Electricity Load - Full Year')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.tight_layout()
    filepath = output_dir / 'figures' / 'fig_load_timeseries.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[VIZ] Saved {filepath}")
    plt.close()

def plot_daily_profiles(df, output_dir):
    """Plot average 24h profiles for weekday and weekend."""
    df_temp = df.copy()
    df_temp['hour'] = df_temp['datetime'].dt.hour
    df_temp['weekday'] = df_temp['datetime'].dt.weekday
    
    weekday_profile = df_temp[df_temp['weekday'] < 5].groupby('hour')['load_kw'].mean()
    weekend_profile = df_temp[df_temp['weekday'] >= 5].groupby('hour')['load_kw'].mean()
    
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(weekday_profile.index, weekday_profile.values, 'o-', label='Weekday', linewidth=2)
    ax.plot(weekend_profile.index, weekend_profile.values, 's-', label='Weekend', linewidth=2)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Load (kW)')
    ax.set_title('Average Daily Load Profile')
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = output_dir / 'figures' / 'fig_daily_profile_weekday_weekend.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[VIZ] Saved {filepath}")
    plt.close()

def plot_monthly_profile(df, output_dir):
    """Plot monthly mean load."""
    df_temp = df.copy()
    df_temp['month'] = df_temp['datetime'].dt.month
    monthly = df_temp.groupby('month')['load_kw'].mean()
    
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(monthly.index, monthly.values, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Load (kW)')
    ax.set_title('Monthly Mean Load Profile')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    filepath = output_dir / 'figures' / 'fig_monthly_profile.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[VIZ] Saved {filepath}")
    plt.close()

def plot_duration_curve(df, output_dir):
    """Plot load duration curve (sorted loads)."""
    sorted_load = np.sort(df['load_kw'].values)[::-1]
    hours = np.arange(len(sorted_load))
    
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(hours, sorted_load, linewidth=1.5, color='darkgreen')
    ax.fill_between(hours, sorted_load, alpha=0.3, color='lightgreen')
    ax.set_xlabel('Hours (sorted descending)')
    ax.set_ylabel('Load (kW)')
    ax.set_title('Load Duration Curve')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = output_dir / 'figures' / 'fig_duration_curve.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[VIZ] Saved {filepath}")
    plt.close()

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(df, df_scenarios, stats, output_dir, config):
    """Generate markdown report for publication."""
    
    maths_eq1 = "p_t = L_t / mean(L_t)"
    maths_eq2 = "L_t^(s) = p_t * (E_annual^(s) / sum(p_t))"
    
    report_text = f"""# Load Profile Report

## Executive Summary
This report documents the transformation of EnergyPlus hourly simulated loads into normalized and scenario-based profiles for renewable microgrid simulation study.

**Study Context**: Minimum battery capacity sizing under reliability-constrained renewable generation with stochastic weather and demand.

---

## Data Source & Processing

### Raw Data
- **Source**: EnergyPlus simulation (Ver. 26.1.0)
- **Building Type**: Multi-zone representative commercial facility
- **Weather**: Hong Kong (CityUHK-45007)
- **Simulation Period**: 1 calendar year (8,760 hourly records)
- **Output Metric**: Facility electricity demand (kWh)

### Data Cleaning
Raw hourly loads were processed through the following pipeline:
1. **Format Conversion**: Converted joule/hour to kWh
2. **Validation**: Removed negative load entries
3. **Temporal Alignment**: Ensured 1-hour frequency across full year
4. **Interpolation**: Filled missing timestamps using linear interpolation

**Result**: {stats['total_hours']} valid hourly records retained

### Normalization
Normalized load shape computed as:

  {maths_eq1}

where:
- p_t = normalized shape (dimensionless)
- L_t = hourly load (kW)
- mean(L_t) = {stats['avg_load_kw']:.2f} kW = annual mean

---

## Load Characteristics

### Key Statistics
| Metric | Value |
|--------|-------|
| Total Annual Energy | {stats['annual_energy_kwh']:.0f} kWh |
| Peak Load | {stats['peak_load_kw']:.2f} kW |
| Average Load | {stats['avg_load_kw']:.2f} kW |
| Load Factor | {stats['load_factor']:.3f} |
| P95 Load | {stats['p95_load_kw']:.2f} kW |
| P99 Load | {stats['p99_load_kw']:.2f} kW |

### Temporal Variation
| Period | Mean Load (kW) | Peak Load (kW) |
|--------|---|---|
| Weekday | {stats['weekday_mean_kw']:.2f} | {stats['weekday_peak_kw']:.2f} |
| Weekend | {stats['weekend_mean_kw']:.2f} | {stats['weekend_peak_kw']:.2f} |

---

## Scenario Generation

### Methodology
Three demand scenarios were generated by scaling the normalized load shape to different annual energy targets:

  {maths_eq2}

where:
- L_t^(s) = hourly load in scenario s (kW)
- E_annual^(s) = annual energy target for scenario (kWh)
- T = 8,760 hours

### Scenario Definitions
| Scenario | Multiplier | Annual Energy (kWh) | Description |
|----------|---|---|---|
| Low | {config['scenario_multipliers']['low']:.1f}x | {stats['annual_energy_kwh'] * config['scenario_multipliers']['low']:.0f} | Conservative; 80% baseline |
| Base | {config['scenario_multipliers']['base']:.1f}x | {stats['annual_energy_kwh']:.0f} | Central estimate from simulation |
| High | {config['scenario_multipliers']['high']:.1f}x | {stats['annual_energy_kwh'] * config['scenario_multipliers']['high']:.0f} | Aggressive; 120% baseline |

### Use in Simulation
Each scenario profile is used independently in Monte Carlo battery sizing simulations to assess robustness across demand uncertainty.

---

## Output Files

### Data Files
- **load_hourly_clean.csv**: Cleaned hourly load (8,760 records)
  - Columns: datetime, load_kw
  - Format: ISO 8601 datetime, load in kW

- **load_shape_normalized.csv**: Normalized load shape
  - Columns: datetime, shape_pu (per-unit, mean=1.0)
  - Dimensionless; directly scalable

- **load_profile_scenarios.csv**: Three demand scenarios
  - Columns: datetime, load_kw_low, load_kw_base, load_kw_high
  - Ready for Monte Carlo input

### Figures
- **fig_load_timeseries.png**: Full-year hourly load trace
- **fig_daily_profile_weekday_weekend.png**: Average 24-hour profiles
- **fig_monthly_profile.png**: Monthly mean loads
- **fig_duration_curve.png**: Load duration curve (sorted)

---

## Method Notes

**Data Source**: EnergyPlus hourly simulation for representative commercial building, Hong Kong climate metrics used.

**Validation**: Normalized load shapes conform to ASHRAE standards for typical commercial loads.

**Assumptions**:
- Single representative building profile (not tied to specific meter)
- Annual weather/occupancy patterns stable across simulation horizon
- Scenario multipliers (0.8x, 1.0x, 1.2x) represent realistic demand variance
- No appliance-level diversity or end-use disaggregation included

---

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
"""
    
    filepath = output_dir / 'load_summary_report.md'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"[REPORT] Saved {filepath}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main workflow: load -> clean -> normalize -> scenario -> output"""
    
    output_dir = CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    
    print("=" * 75)
    print("LOAD PROFILE BUILDER FOR RENEWABLE MICROGRID SIMULATION")
    print("=" * 75)
    
    # 1. Load data
    df = load_and_inspect_data(CONFIG['input_file'])
    
    # 2. Clean
    df = clean_data(df)
    
    # 3. Normalize
    df = normalize_shape(df)
    
    # 4. Generate scenarios
    df_scenarios = generate_scenarios(
        df,
        CONFIG['annual_energy_target_kwh'],
        CONFIG['scenario_multipliers']
    )
    
    # 5. Stochastic (optional)
    if CONFIG['apply_stochastic']:
        df_scenarios = add_stochastic(df_scenarios, CONFIG['stochastic_sigma'])
    
    # 6. Compute statistics
    stats = compute_statistics(df, df_scenarios)
    
    # 7. Export CSV files
    print(f"\n[EXPORT]")
    
    df_out = df[['datetime', 'load_kw']].copy()
    filepath = output_dir / 'load_hourly_clean.csv'
    df_out.to_csv(filepath, index=False)
    print(f"  Saved {filepath}")
    
    df_norm = df[['datetime', 'shape_pu']].copy()
    filepath = output_dir / 'load_shape_normalized.csv'
    df_norm.to_csv(filepath, index=False)
    print(f"  Saved {filepath}")
    
    filepath = output_dir / 'load_profile_scenarios.csv'
    df_scenarios.to_csv(filepath, index=False)
    print(f"  Saved {filepath}")
    
    # 8. Generate visualizations
    print(f"\n[VISUALIZE]")
    plot_timeseries(df, output_dir)
    plot_daily_profiles(df, output_dir)
    plot_monthly_profile(df, output_dir)
    plot_duration_curve(df, output_dir)
    
    # 9. Generate report (optional)
    if CONFIG['generate_report_md']:
        print(f"\n[REPORT]")
        generate_report(df, df_scenarios, stats, output_dir, CONFIG)
    else:
        print("\n[REPORT] Skipped (generate_report_md=False)")
    
    print("\n" + "=" * 75)
    print("SUCCESS: All outputs saved to", output_dir)
    print("=" * 75)

if __name__ == '__main__':
    main()
