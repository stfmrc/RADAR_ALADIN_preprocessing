"""
This script converts ARSO ALADIN GRIB forecast datasets into a single,
merged NetCDF file per year and month, harmonizing variables across
different levels and stacking time/step dimensions.

- Input:
  Reads GRIB files located in:
      /scratch_ssd2/stefanelli/3DHNN_18_06_2025/0_DATASETS_FROM_ARSO_AND_UNZIPPED/
          ALADIN_unzipped/{year}/{month}/ashz_{year}{month}*.grb
  Arguments required:
      --year   : Year of the dataset (e.g., 2019)
      --month  : Month of the dataset (e.g., 01)

- Processing:
  • Loads multiple GRIB files using xarray with the cfgrib engine.
  • Extracts variables at specific levels:
      - 2m height: temperature, humidity, wind, etc.
      - 10m height: temperature, wind speed
      - Pressure levels and mean sea level
      - Surface variables (sp, tcc, z, tp, etc.)
  • Renames overlapping variables for clarity (e.g., z → z2m, z10m, z_pressure).
  • Merges all datasets into a single xarray.Dataset.
  • Stacks 'time' and 'step' dimensions into a single 'time' dimension.
  • Assigns a new time coordinate combining forecast times and steps.
  • Updates global metadata with author info and dataset comments.

- Output:
  Saves the merged dataset as a NetCDF file in:
      ./NetCDF/ALADIN_{year}{month}.nc

- Purpose:
  Provides a consistent, analysis-ready NetCDF version of ALADIN forecast
  data for subsequent modeling, visualization, or use in ML pipelines
  (e.g., 3DHNN SMASH project).

AUTHOR: Marco Stefanelli
"""


import warnings
import datetime
import argparse
import xarray as xr
from pathlib import Path
import cfgrib
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

start = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year to be converted in NetCDF", type=str, required=True, default='2019')
parser.add_argument("--month", help="Month to be converted in NetCDF", type=str, required=True, default='01')
args = parser.parse_args()


files_dir = '/scratch_ssd2/stefanelli/3DHNN_18_06_2025/0_DATASETS_FROM_ARSO_AND_UNZIPPED'
unzip_dir  = f'{files_dir}/ALADIN_unzipped/{args.year}/{args.month}/'
filename = f'ashz_{args.year}{args.month}*.grb'

directory_datasets = Path(f'NetCDF/')
# Check if the directory exists    
if not directory_datasets.exists():
    directory_datasets.mkdir(parents=True, exist_ok=True)

print(f'\n\n YEAR IS {args.year}', flush=True)
print(f' MONTH IS {args.month}', flush=True)

print('\n-------------------------------------------------------------------\n', flush=True)


# --------------------------------------------------------------------------#
#                             GRIB TO NETCDF
# --------------------------------------------------------------------------#

# Function to load and filter dataset by heightAboveGround, if present rename r in r2m 
# and rename heightAboveGround in heightAboveGround{height_level}
def load_filtered_dataset(filename, height_level):
    ds = xr.open_mfdataset(f'{unzip_dir}/{filename}', 
                           combine='nested',
                           concat_dim='time',
                           engine='cfgrib', 
                           backend_kwargs={'indexpath': '', 
                                           'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': height_level}})
    ds = ds.rename({'heightAboveGround' : f'heightAboveGround{height_level}'})
    return ds


# OPEN DATSETS

dsHeight2m = load_filtered_dataset(filename, 2)

dsHeight10m = load_filtered_dataset(filename, 10)

dsPressure=xr.open_mfdataset(f'{unzip_dir}/{filename}', 
                   combine='nested',
                   concat_dim='time',
                   engine='cfgrib', 
                   backend_kwargs={'indexpath':'', 'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}} )

dsMeanSea=xr.open_mfdataset(f'{unzip_dir}/{filename}', 
                   combine='nested',
                   concat_dim='time',
                   engine='cfgrib', 
                   backend_kwargs={'indexpath':'', 'filter_by_keys': {'typeOfLevel': 'meanSea'}} )

# List of short names to load from the GRIB file at surface level
variables = ['unknow','sp', 'tcc', 'z', 'tp']

# Load each variable and store datasets in a list
datasets = []
for var in variables:    
    ds=xr.open_mfdataset(f'{unzip_dir}/{filename}', 
                   combine='nested',
                   concat_dim='time',
                   engine='cfgrib', 
                   backend_kwargs={'indexpath':'', 'filter_by_keys': {'typeOfLevel': 'surface', 'shortName': var}})
    datasets.append(ds)
    
# Merge all datasets into a single dataset
dsSurface = xr.merge(datasets, compat='override')

if 'z' in dsHeight2m.variables:
    dsHeight2m = dsHeight2m.rename({'z': 'z2m'})
    print('\nz is present in dsHeight2m and it is renamed as z2m\n', flush=True)
    print(dsHeight2m, flush=True)
    
if 'q' in dsHeight2m.variables:
    dsHeight2m = dsHeight2m.rename({'q': 'q2m'})
    print('\nq is present in dsHeight2m and it is renamed as q2m\n', flush=True)
    print(dsHeight2m, flush=True)
    
if 'r' in dsHeight2m.variables:
    dsHeight2m = dsHeight2m.rename({'r': 'r2m'})
    print('\nr is present in dsHeight2m and it is renamed as r2m\n', flush=True)
    print(dsHeight2m, flush=True)

if 'z' in dsHeight10m.variables:
    dsHeight10m = dsHeight10m.rename({'z': 'z10m'})
    print('\nz is present in dsHeight10m and it is renamed as z10m\n', flush=True)
    print(dsHeight10m, flush=True)

if 'ws' in dsHeight10m.variables:
    dsHeight10m = dsHeight10m.rename({'ws': 'ws10m'})
    print('\nws is present in dsHeight10m and it is renamed as ws10m\n', flush=True)
    print(dsHeight10m, flush=True)    
    
    
if 'z' in dsPressure.variables:
    dsPressure = dsPressure.rename({'z': 'z_pressure'})
    print('\nz is present in dsPressure\n', flush=True)
    print(dsPressure, flush=True)

if 'q' in dsPressure.variables:
    print(f'\n has q on the different pressure levels', flush=True)    
    
if 'z' in dsMeanSea.variables:
    dsMeanSea = dsMeanSea.rename({'z': 'z_mean_sea'})
    print('\nz is present in dsMeanSea and it is renamed as z_mean_sea\n', flush=True)
    print(dsMeanSea, flush=True)

if 'z' in dsSurface.variables:
    dsSurface = dsSurface.rename({'z': 'z_surf'})
    print('\nz is present in dsSurface\n', flush=True)
    print(dsSurface, flush=True)

merged_data = xr.merge([dsHeight2m, dsHeight10m, dsMeanSea, dsSurface, dsPressure])

    
#----------------------------------------------------
#          REFINE THE DATASET 
#----------------------------------------------------

# Define new attributes or update existing ones
new_attrs = {
    'AUTHOR': 'Marco Stefanelli, Neva Pristov, Jure Cedilnik',
    'MAIL': 'marco.stefanelli@fmf.uni-lj.si',
    'COMMENT': f'Concatenated original ARSO ALADIN dataset for {args.year} {args.month} to be used in the 3DHNN SMASH project'
}

# Update the dataset attributes
merged_data.attrs.update(new_attrs)

print('\n-------------------------------------------------------------------\n', flush=True)

print('\nThe merged datasets are:', flush=True)

print('\n dsHeight2m \n', flush=True)
print(dsHeight2m, flush=True)

print('\n dsHeight10m \n', flush=True)
print(dsHeight10m, flush=True)

print('\n dsPressure \n', flush=True)
print(dsPressure, flush=True)

print('\n dsMeanSea \n', flush=True)
print(dsMeanSea, flush=True)

print('\n dsSurface \n', flush=True)
print(dsSurface, flush=True)

print('\n-------------------------------------------------------------------\n')

# Assign time coordinate instead of time and step
time2 = np.zeros(len(merged_data["time"])*len(merged_data["step"]), dtype="datetime64[ns]")

# Populate the new array for time coordinates
k=0
for i in range(len(merged_data["time"])):
            for j in range(len(merged_data["step"])):
                           time2[k]=merged_data["time"].values[i] + merged_data["step"].values[j]
                           k=k+1
                           
# Stack the time and step dimensions into a single dimension
ds_stacked = merged_data.stack(time_step=("time", "step"))
# Assign the new time2 coordinate to replace the stacked dimension
ds_stacked = ds_stacked.assign_coords(time_step=("time_step", time2))
# Rename the stacked dimension for clarity
ds_stacked = ds_stacked.rename(time_step="time")

'''
# remove duplicate time indexes (The last hour is present in 2 consecutive run)
_, index = np.unique(ds_stacked['time'], return_index=True)
ds_stacked = ds_stacked.isel(time=index)
# Remove the last timestep of the model
ds_stacked = ds_stacked.isel(time=slice(None, -1))
# Time on first position
ds_stacked = ds_stacked.transpose("time", ...)
'''
print('\nThe final dataset saved in NetCDF is:\n', flush=True)
print(ds_stacked)




ds_stacked.to_netcdf(f'{directory_datasets}/ALADIN_{args.year}{args.month}.nc')


print(f'\n{args.year}{args.month} TO NetCDF DONE!!!\n', flush=True)
end = datetime.datetime.now()
print(f'START --> {start}')
print(f'END   --> {end}')


    


