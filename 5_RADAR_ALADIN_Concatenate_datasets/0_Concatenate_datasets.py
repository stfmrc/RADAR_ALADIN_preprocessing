"""
Script: 0_Concatenate_datasets.py
Purpose: Incrementally concatenate RADAR and ALADIN NetCDF datasets into single files.

Input:
  - RADAR datasets:
      /scratch_ssd2/stefanelli/3DHNN_18_06_2025/2_DATASETS_PREPROCESSING_FOR_3DHNN/3_hourly_average/hourly_cumulated_dbzh/PAZZ41/sweep_0/*
  - ALADIN datasets:
      /scratch_ssd2/stefanelli/3DHNN_18_06_2025/2_DATASETS_PREPROCESSING_FOR_3DHNN/4_ALADIN_GRIB_to_NetCDF/NetCDF/*

Processing:
  - RADAR:
      • Opens multiple files with xarray (open_mfdataset)
      • Concatenates along a temporary dimension
      • Writes to a single NetCDF file (h5netcdf engine)
  - ALADIN:
      • Opens each NetCDF file individually
      • Keeps only selected variables and first 4 pressure levels
      • Drops time steps according to a mask (every 7th step retained)
      • Concatenates all processed datasets along the time dimension
      • Writes to a single NetCDF file (h5netcdf engine)

Output:
  - RADAR concatenated dataset: RADAR_2019_2023.nc
  - ALADIN concatenated dataset: ALADIN_2019_2023.nc

Usage:
  - Run in background to detach from terminal:
      nohup python 0_Concatenate_datasets.py > 0_output.log 2> 0_error.log &

Notes:
  - Adjust chunk size for memory efficiency (currently chunks={'concat_dim': 700})
  - Uses incremental loading to avoid memory overflow
  - Memory usage printed after RADAR and ALADIN concatenations

AUTHOR: Marco Stefanelli
"""
import xarray as xr
import numpy as np
import glob
import psutil
import os
import gc

# ===============================
# Paths and other params
# ===============================
model_path = sorted(glob.glob(
    "/scratch_ssd2/stefanelli/3DHNN_18_06_2025/2_DATASETS_PREPROCESSING_FOR_3DHNN/4_ALADIN_GRIB_to_NetCDF/NetCDF/*"
))
radar_path = sorted(glob.glob(
    "/scratch_ssd2/stefanelli/3DHNN_18_06_2025/2_DATASETS_PREPROCESSING_FOR_3DHNN/3_hourly_average/hourly_cumulated_dbzh/PAZZ41/sweep_0/*"
))

aladin_variables_to_keep = [
      "t", "v", "u", "r", "r2m", "t2m", "msl",
      "z_pressure", "longitude", "latitude"
]

radar_out_name  = "RADAR_2019_2023.nc"
aladin_out_name = "ALADIN_2019_2023.nc"

# ===============================
# RADAR PROCESSING (incremental)
# ===============================
print('RADAR concatenation ...', flush=True)

ds = xr.open_mfdataset(
    radar_path,
    combine='nested',
    concat_dim='concat_dim',
    parallel=True,
    chunks={'concat_dim': 700}  # adjust chunk size to memory
)

ds.to_netcdf(radar_out_name, engine='h5netcdf')  # write once

del ds
gc.collect()

# Memory check
process = psutil.Process(os.getpid())
print(f"Memory usage after radar: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# ===============================
# ALADIN PROCESSING (incremental)
# ===============================
print('ALADIN concatenation ...', flush=True)

aladin_datasets = []
for file in model_path:
    print(f"Processing: {file}")
    with xr.open_dataset(file, chunks={}) as ds:
        # keep only selected variables
        ds = ds[aladin_variables_to_keep]

        # keep first 4 pressure levels
        ds = ds.isel(isobaricInhPa=slice(0, 4))

        # drop timesteps by mask
        time_indices = np.arange(ds.sizes["time"])
        mask = (time_indices + 1) % 7 != 0
        ds = ds.isel(time=mask)

        aladin_datasets.append(ds.load())  # force load then free

# Final concat
aladin_concat = xr.concat(aladin_datasets, dim="time")
aladin_concat = aladin_concat.transpose("time", ..., "y", "x")

aladin_concat.to_netcdf(
    aladin_out_name,
    engine="h5netcdf"
)

# Memory check
print(f"Memory usage after aladin: {process.memory_info().rss / 1024 ** 2:.2f} MB")
print("DONE!!!")