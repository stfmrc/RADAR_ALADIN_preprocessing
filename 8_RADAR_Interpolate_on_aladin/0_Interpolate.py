"""
Script: 0_Interpolate.py

Purpose:
---------
Interpolate RADAR reflectivity data (DBZH) onto the ALADIN model grid.
This ensures both datasets share the same spatial resolution for joint analysis.

Input:
---------
- ALADIN dataset (time-aligned): NetCDF file with ALADIN model variables
- RADAR dataset (cleaned & filtered): NetCDF file with DBZH reflectivity

Processing Steps:
-----------------
1. Load ALADIN and RADAR datasets using xarray.
2. Optionally mask the outermost radar range bin to avoid spurious values.
3. Extract spatial coordinates (longitude, latitude) for both datasets.
4. Loop over each timestep:
   - Interpolate RADAR DBZH values onto ALADIN grid using nearest-neighbor interpolation.
5. Build a new xarray Dataset containing interpolated RADAR values while preserving metadata.
6. Save the interpolated dataset to a NetCDF file.

Output:
--------
- NetCDF file with RADAR DBZH interpolated onto ALADIN grid, e.g.,
  RADAR_2019_2023_INTERPOLATED_160km.nc

Usage:
-------
- Submit in background using:
    nohup python 0_Interpolate.py > 0_output.log 2> 0_error.log &

Notes:
------
- The interpolation preserves original dataset attributes and coordinates.
- Useful for further analysis where ALADIN and RADAR datasets must be spatially aligned.

AUTHOR: Marco Stefanelli
"""


import xarray as xr
import numpy as np
from scipy.interpolate import griddata

# ==========================================================
# Load datasets: ALADIN model and RADAR observations
# ==========================================================
aladin = xr.open_dataset(
    '/scratch_ssd2/stefanelli/3DHNN_18_06_2025/2_DATASETS_PREPROCESSING_FOR_3DHNN/7_ALADIN_Time_alignment/ALADIN_2019_2023_radar_time_aligned.nc'
)

radar = xr.open_dataset(
    "/scratch_ssd2/stefanelli/3DHNN_18_06_2025/2_DATASETS_PREPROCESSING_FOR_3DHNN/6_RADAR_Clean_zeros_timesteps_and_outliers/RADAR_2019_2023_cleaned_zeros_time_and_outliers_filtered_160Km.nc", chunks={'concat_dim': 100}
)


# === Mask the outermost range bin with NaN ===
# If not range filtering is applied.
# This keeps the dataset dimensions unchanged, but ensures the last bin is ignored
# during interpolation (no artificial values outside the radar circle).
# radar["DBZH"] = radar["DBZH"].where(radar["range"] != radar["range"].max())

print(radar)

# ==========================================================
# Extract spatial coordinates from both datasets
# ==========================================================
lonR = radar.longitude.values  # RADAR longitudes
latR = radar.latitude.values   # RADAR latitudes
lonM = aladin.longitude.values # ALADIN model longitudes
latM = aladin.latitude.values  # ALADIN model latitudes

# ==========================================================
# Extract reflectivity (DBZH) and number of timesteps
# ==========================================================
DBZH = radar.DBZH.values
n_timesteps = radar.time.shape[0]

# ==========================================================
# Prepare empty array to store interpolated values
# (dimensions: time × model_lat × model_lon)
# ==========================================================
interpolated_radar_data = np.zeros((n_timesteps, latM.shape[0], lonM.shape[1]))

# ==========================================================
# Interpolation loop
# Interpolates radar DBZH onto the ALADIN grid, timestep by timestep
# ==========================================================
for t in range(n_timesteps):
    
    # Progress print every 10000 timesteps
    if (t + 1) % 100 == 0:
        print(f"Interpolating timestep {t+1}/{n_timesteps}", flush=True)
    
    # Interpolation using nearest-neighbor method
    interpolated_values = griddata(
        (lonR[t, :, :].flatten(), latR[t, :, :].flatten()),  # input points
        DBZH[t, :, :].flatten(),                             # input values
        (lonM, latM),                                        # target grid
        method='nearest'
    )
    
    # Store results
    interpolated_radar_data[t] = interpolated_values

# ==========================================================
# Build new xarray Dataset with interpolated radar data
# Preserve original attributes and metadata
# ==========================================================
ds = xr.Dataset(
    {
        "DBZH": (
            ["time", "x", "y"],
            interpolated_radar_data,
            radar["DBZH"].attrs  # copy DBZH attributes
        ),
    },
    coords={
        "time": radar.time.values[:, 0],
        "lat": (["x", "y"], latM, radar["latitude"].attrs),
        "lon": (["x", "y"], lonM, radar["longitude"].attrs),
    },
    attrs=radar.attrs  # copy global attributes
)

print(ds)

# ==========================================================
# Save interpolated dataset to NetCDF
# ==========================================================
ds.to_netcdf("RADAR_2019_2023_INTERPOLATED_160km.nc")

print("DONE!!!")
