"""
This script computes hourly cumulated radar reflectivity (DBZH) datasets
from previously concatenated monthly NetCDF files and converts radar
altitude levels to atmospheric pressure levels.

- Input:
  Reads a monthly concatenated NetCDF file located at:
      /scratch_ssd2/stefanelli/3DHNN_18_06_2025/
          2_monthy_concatenated_dataset/monthly_concatenated_datasets/
          {radar}/{sweep}/{YYYYMM}.nc
  Arguments required:
      --year    : Year of the dataset (e.g., 2019)
      --month   : Month of the dataset (e.g., 01)
      --radar   : Radar identifier ("PAZZ41" or "PAZZ42")
      --sweep   : Radar sweep name (e.g., "sweep_0")

- Processing:
  • Converts radar altitude (z) levels to pressure levels (hPa) using
    the barometric formula.
  • Filters DBZH values above a minimum threshold (13.5 dBZ) and
    computes hourly sums in linear units, converting back to dBZ.
  • Calculates hourly averaged longitude, latitude, and z grids.
  • Generates a new dataset with hourly cumulated DBZH, corresponding
    coordinates, pressure, and time dimensions.

- Output:
  Saves the hourly cumulated dataset as a NetCDF file under:
      hourly_cumulated_dbzh/{radar}/{sweep}/{YYYYMM}.nc
  Attributes include metadata such as radar info and comments on processing.

- Purpose:
  Facilitates hourly radar precipitation analysis by transforming raw
  DBZH measurements into cumulative hourly values and pressure levels,
  ready for further hydrological or meteorological modeling.

AUTHOR: Marco Stefanelli
"""



#------------------------------------------------------------------------
#      IMPORT MODULES AND DEFINE FUNCTIONS
#------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyproj.network")


import os
import pyproj

# Set PROJ path explicitly
os.environ['PROJ_LIB'] = "/scratch_ssd2/stefanelli/anaconda3/envs/3DHNN_preproc/share/proj"
pyproj.datadir.set_data_dir(os.environ['PROJ_LIB'])

import xarray as xr
import glob
from pathlib import Path
import gc
import os
import argparse
import numpy as np
import time
import datetime

# TRANSFORM RADAR Z LEVELS TO PRESSURE LEVELS
def height_to_pressure(height_m):
    """
    Convert height in meters to pressure in hPa using the barometric formula.
    """
    # Constants for the standard atmosphere
    P0 = 1013.25  # Sea level standard atmospheric pressure in hPa
    T0 = 288.15   # Sea level standard temperature in Kelvin
    L = 0.0065    # Temperature lapse rate in K/m
    R = 8.31447   # Ideal gas constant in J/(mol*K)
    M = 0.0289644 # Molar mass of Earth's air in kg/mol
    g = 9.80665   # Standard gravity in m/s^2

    # Calculate the pressure using the barometric formula
    pressure_hPa = P0 * (1 - (L * height_m) / T0) ** (g * M / (R * L))
    
    return pressure_hPa


# COMPUTE RAIN RATE
def compute_R(DBZ):
    Z = 10**(DBZ/10)
    R = (Z / 250) ** (2/3)
    return R

# COMPUTE HOURLY CUMULATED DBZH DATASET
def compute_hourly_cumulated_dbzh(dataset):
    print('Computing hourly cumulated dbzh ...')
    # Filter the dataset with lower cutoff decibel value
    DBZH_sel = dataset['DBZH'].where(dataset["DBZH"] > 13.5, drop=False)
    # Transform times such that the final format is YYYYMMDDHH
    time     = dataset.time.values.astype('datetime64[h]')
    time_min = np.min(time).astype('datetime64[h]')
    time_max = np.max(time).astype('datetime64[h]')
    
    # Compute time bins
    time_bins = (time - time_min).astype(int)


    # Find the number of unique hourly bins
    num_hours = np.max(time_bins).astype(int) + 1
    print('num hours', num_hours)

    # Initialize array to hold hourly cumulated dbzh
    hourly_sum_DBZH = np.zeros((num_hours, dataset.DBZH.shape[1], dataset.DBZH.shape[2]))
    hourly_avg_lons = np.zeros((num_hours, dataset.DBZH.shape[1], dataset.DBZH.shape[2]))
    hourly_avg_lats = np.zeros((num_hours, dataset.DBZH.shape[1], dataset.DBZH.shape[2]))
    hourly_avg_z    = np.zeros((num_hours, dataset.DBZH.shape[1], dataset.DBZH.shape[2]))


    # Compute hourly cumulated dbzh
    print('TIME:', time_min, time_max, time_bins )
    for hour in range(num_hours):
        # Create a mask for the current hour
        mask = (time_bins == hour)
        
        
        if np.any(mask):
            array_DBZH = np.reshape(DBZH_sel.values[mask, :],(-1,360,249))
            array_lons = np.reshape(dataset.longitude.values[mask, :],(-1,360,249))
            array_lats = np.reshape(dataset.latitude.values[mask, :],(-1,360,249))
            array_z = np.reshape(dataset.z.values[mask, :],(-1,360,249))
            
            # Convert from dBZ to Z (linear units)
            Z_lin = 10 ** (array_DBZH / 10.0)
#            print("Z_lin:", np.nanmin(Z_lin),np.nanmax(Z_lin))

            # Sum in linear units
            Z_lin_sum = np.nansum(Z_lin, axis=0)
#            print("Z_lin_sum:", np.nanmin(Z_lin_sum),np.nanmax(Z_lin_sum))

            # Mask or replace zeros to avoid log10(0)
            Z_lin_sum[Z_lin_sum <= 0] = np.nan

            # Convert back to dBZ 
            hourly_sum_DBZH[hour, :, :] = 10 * np.log10(Z_lin_sum)
#            print("hourly_sum_DBZH:", np.nanmin(hourly_sum_DBZH[hour, :, :]),np.nanmax(hourly_sum_DBZH[hour, :, :]))

            #hourly_sum_DBZH[hour, :, :] = np.nansum(array_DBZH, axis=0) # BETTER nanmean or mean?
            hourly_avg_lons[hour, :, :] = np.nanmean(array_lons, axis=0) #(array_lons[round(array_lons.shape[0]/2),:,:]) # BETTER nanmean or mean?
            hourly_avg_lats[hour, :, :] = np.nanmean(array_lats, axis=0) #(array_lats[round(array_lats.shape[0]/2),:,:]) # BETTER nanmean or mean?
            hourly_avg_z[hour, :, :]    = np.nanmean(array_z,    axis=0) #(array_lats[round(array_lats.shape[0]/2),:,:]) # BETTER nanmean or mean?
    
    del array_DBZH
    del array_lons
    del array_lats
    del array_z
    gc.collect()
    
    # Generate an array of hourly timestamps between time_min and time_max (1D array)
    hourly_times = np.arange(time_min, time_max + np.timedelta64(1, 'h'), np.timedelta64(1, 'h'))

    # Tile the 1D array across the azimuth dimension to get the shape (num_hours, azimuth_dim)
    hourly_times2d = np.tile(hourly_times[:, np.newaxis], (1, time.shape[1]))
    
    # Define a dataset to hold the cumulated DBZH
    sum_dataset = dataset

    # Drop the existing DBZH and time variables to avoid conflicts
    try:
        sum_dataset = sum_dataset.drop_vars(['DBZH', 'time','longitude','latitude','x','y','z'])
    except:
        sum_dataset = sum_dataset.drop_vars(['DBZH', 'time','longitude','latitude','z'])

    # Adjust the concat_dim coordinate to match the new size
    sum_dataset = sum_dataset.assign_coords(concat_dim=np.arange(num_hours))

    # Replace DBZH and time in the dataset with new data
    sum_dataset = sum_dataset.assign(
        DBZH=(('concat_dim', 'azimuth', 'range'), hourly_sum_DBZH),
        longitude=(('concat_dim', 'azimuth', 'range'), hourly_avg_lons),
        latitude=(('concat_dim', 'azimuth', 'range'), hourly_avg_lats),
        z=(('concat_dim', 'azimuth', 'range'), hourly_avg_z),
        time=(('concat_dim', 'azimuth'), hourly_times2d)
    )
    
    return sum_dataset


start = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument("--year",  help="Year to be converted in NetCDF", type=str, required=True, default='2019')
parser.add_argument("--month", help="Month to be converted in NetCDF", type=str, required=True, default='01')
parser.add_argument("--radar", help="RADAR folder. Options are PAZZ41 or PAZZ42", type=str, required=True, default='PAZZ41')
parser.add_argument("--sweep", help="Sweep of the h5 dataset. Options are from sweep_0 to sweep_11", type=str, required=True, default='sweep_0')
args = parser.parse_args()


#------------------------------------------------------------------------
#      COMPUTE CUMULATED PRECIPITATION DATASET AND PRESSURE LEVELS
#------------------------------------------------------------------------

files_dir = '/scratch_ssd2/stefanelli/3DHNN_18_06_2025/2_DATASETS_PREPROCESSING_FOR_3DHNN/2_monthy_concatenated_dataset/'

output_path = f'hourly_cumulated_dbzh/{args.radar}/{args.sweep}'    

directory_datasets = Path(output_path)
# Check if the output directory exists    
if not directory_datasets.exists():
    directory_datasets.mkdir(parents=True, exist_ok=True)

#------------------------------------------------------------------------
#      COMPUTE CUMULATED PRECIPITATION DATASET AND PRESSURE LEVELS
#------------------------------------------------------------------------
start_time = time.time()

file = f'{files_dir}/monthly_concatenated_datasets/{args.radar}/{args.sweep}/{args.year}{args.month}.nc'

print(f'\nRADAR IS {args.radar}, YEAR IS {args.year}, MONTH IS {args.month}, SWEEP IS {args.sweep}', flush=True)
print('\n-------------------------------------------------------------------\n', flush=True)

print(file, flush=True)
RADAR = xr.open_dataset(file)
SUM_RADAR = compute_hourly_cumulated_dbzh(RADAR)
z = SUM_RADAR.z
p = height_to_pressure(z)
SUM_RADAR["p"] = (('concat_dim','azimuth','range'), p.values)

print('\n\n Final dataset is \n\n',SUM_RADAR, flush=True)

#----------------------------------------------------
#                  SAVE THE DATASET 
#----------------------------------------------------
filename_with_ext = os.path.basename(file)
print(filename_with_ext,flush=True)
SUM_RADAR.attrs.update({
    'COMMENT': f'Concatenated ARSO RADAR dataset for {args.year} {args.month} at {args.sweep}. Hourly cumulated DBZ'
})
SUM_RADAR.to_netcdf(f'{output_path}/{filename_with_ext}')
del(RADAR)
del(SUM_RADAR)
gc.collect()

end = datetime.datetime.now()

print(f'START --> {start}')
print(f'END   --> {end}')
print(f'\nRADAR IS {args.radar}, YEAR IS {args.year}, MONTH IS {args.month}, SWEEP IS {args.sweep}', flush=True)
print(f'\n\nDataset saved in\n{output_path}')
print('DONE!!!')
