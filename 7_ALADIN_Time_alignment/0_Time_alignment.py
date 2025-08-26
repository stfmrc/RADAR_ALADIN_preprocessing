"""
Script: 0_Time_alignment.py
Purpose: Align ALADIN model dataset with RADAR dataset in time dimension without interpolation,
         ensuring consistent temporal coverage for further analysis.

Input:
  - ALADIN dataset: NetCDF file containing ALADIN model variables across time
  - RADAR dataset: NetCDF file with cleaned RADAR reflectivity (DBZH) and time dimension
  - File paths are configurable via base_dir and file variables.

Processing Steps:
  1. Load ALADIN and RADAR datasets using xarray.
  2. Ensure RADAR time dimension is 1D; fix if needed.
  3. Reindex ALADIN dataset to RADAR time steps (no interpolation).
  4. Identify timesteps in RADAR where any DBZH values are zero for diagnostics.
  5. Save the time-aligned ALADIN dataset to a NetCDF output file.

Output:
  - NetCDF file containing ALADIN variables reindexed to RADAR time steps,
    e.g., ALADIN_2019_2023_radar_time_aligned.nc

Usage:
  - Run in background using:
      nohup python 0_Time_alignment.py > 0_output.log 2> 0_error.log &

Notes:
  - Ensures datasets are temporally compatible for joint analysis.
  - Diagnostic prints provide information on time steps and zero-value timesteps in RADAR.

AUTHOR: Marco Stefanelli
"""

import xarray as xr
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------
# File paths (adjust as needed)
# -------------------------------------------------------------------
base_dir = Path("/scratch_ssd2/stefanelli/3DHNN_18_06_2025")

aladin_file = base_dir / "2_DATASETS_PREPROCESSING_FOR_3DHNN/5_RADAR_ALADIN_Concatenate_datasets/ALADIN_2019_2023.nc"
radar_file  = base_dir / "2_DATASETS_PREPROCESSING_FOR_3DHNN/6_RADAR_Clean_zeros_timesteps_and_outliers/RADAR_2019_2023_cleaned_zeros_time_and_outliers.nc"
output_file = "ALADIN_2019_2023_radar_time_aligned.nc"

# -------------------------------------------------------------------
# Load datasets
# -------------------------------------------------------------------
aladin = xr.open_dataset(aladin_file)
radar  = xr.open_dataset(radar_file)

print(f"ALADIN time steps: {aladin.time.size}")
print(f"RADAR time steps : {radar.time.size}")

# -------------------------------------------------------------------
# Ensure radar time dimension is 1D
# -------------------------------------------------------------------
if radar.time.ndim > 1:
    radar = radar.assign_coords(time=("time", radar.time[:, 0].values))
    print("Fixed radar.time to 1D.")

# -------------------------------------------------------------------
# Align ALADIN time with RADAR time (no interpolation)
# -------------------------------------------------------------------
aladin_aligned = aladin.reindex(time=radar.time, method=None)
print(f"Aligned ALADIN dataset:\n{aladin_aligned}")

# -------------------------------------------------------------------
# Identify timesteps where DBZH contains zeros
# -------------------------------------------------------------------
dbzh = radar.DBZH.values
zero_timesteps = np.where(np.any(dbzh == 0, axis=(1, 2)))[0]
print(f"\nNumber of timesteps with zero values in DBZH: {zero_timesteps.shape[0]}")

# -------------------------------------------------------------------
# Save aligned dataset
# -------------------------------------------------------------------
aladin_aligned.to_netcdf(output_file)
print(f"Aligned ALADIN dataset saved to: {output_file}")
print("DONE!")