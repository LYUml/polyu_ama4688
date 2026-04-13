# Project Overview

This repository contains an EnergyPlus-based load workflow and a PV + battery sizing simulation.

## Structure

- `pv_battery_sizing.py` - main sizing script
- `requirements.txt` - Python dependencies for the root workflow
- `eplus_run/` - EnergyPlus model, weather input, and load profile generation assets
- `weather_data/` - merged weather datasets and merge scripts
- `outputs/` - generated CSV files and figures

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main simulation:

```bash
python pv_battery_sizing.py
```

The script reads:

- `weather_data/govdata/govdata_merged_2025.csv`
- `eplus_run/results/load_profiles/load_hourly_clean.csv`

and writes results to `outputs/`.
