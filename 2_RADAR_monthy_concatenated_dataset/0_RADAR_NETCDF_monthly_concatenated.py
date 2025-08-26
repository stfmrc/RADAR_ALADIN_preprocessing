"""
This script concatenates monthly radar sweep NetCDF files into a single 
dataset, while harmonizing coordinates and attaching metadata.

- Input:
  Reads multiple NetCDF files produced from earlier preprocessing steps, 
  located under:
      /scratch_ssd2/stefanelli/3DHNN_18_06_2025/
          2_DATASETS_PREPROCESSING_FOR_3DHNN/
          1_sweep_to_netcdf/NetCDF/single_nc_files_separated_by_sweeps/{radar}/{sweep}/
  Files are matched by pattern:
      T_{radar}_C_LJLM_{YYYYMM}*.nc

- Processing:
  • Opens one file as a reference to extract CRS, coordinates, and metadata.
  • Uses pyproj to transform Cartesian x/y grids into geographic lon/lat.
  • Iterates through all monthly files, dropping redundant variables and 
    reattaching consistent longitude/latitude grids.
  • Concatenates all datasets along a new dimension (`concat_dim`).
  • Updates global attributes with radar metadata and provenance.

- Output:
  Saves a single concatenated NetCDF file per month to:
      ./monthly_concatenated_datasets/{radar}/{sweep}/{YYYYMM}.nc

- Usage:
  Run via command line with required arguments:
      --year <YYYY>
      --month <MM>
      --radar <PAZZ41|PAZZ42>
      --sweep <sweep_name>

- Purpose:
  Streamlines monthly radar data handling by consolidating many 
  per-scan NetCDF files into one consistent dataset, ready for 
  higher-level analysis or ML pipelines (e.g., 3DHNN).
"""


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyproj.network")


import os
import pyproj

# Set PROJ path explicitly
os.environ['PROJ_LIB'] = "/scratch_ssd2/stefanelli/anaconda3/envs/3DHNN_preproc/share/proj"
pyproj.datadir.set_data_dir(os.environ['PROJ_LIB'])

import xarray as xr
import glob
import argparse
import datetime
from pyproj import CRS, Transformer
import numpy as np
from pathlib import Path

# -------------------------------
# Parse arguments
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True)
parser.add_argument("--month", required=True)
parser.add_argument("--radar", required=True)
parser.add_argument("--sweep", required=True)
args = parser.parse_args()

# -------------------------------
# File paths
# -------------------------------
files_dir = '/scratch_ssd2/stefanelli/3DHNN_18_06_2025/2_DATASETS_PREPROCESSING_FOR_3DHNN/1_sweep_to_netcdf/NetCDF'
file_pattern = f"{files_dir}/single_nc_files_separated_by_sweeps/{args.radar}/{args.sweep}/T_{args.radar}_C_LJLM_{args.year}{args.month}*.nc"
path = sorted(glob.glob(file_pattern))

print(f"\nRADAR={args.radar}, YEAR={args.year}, MONTH={args.month}, SWEEP={args.sweep}")
print(f"Found {len(path)} files")

# -------------------------------
# Output directory
# -------------------------------
output_path = Path("./monthly_concatenated_datasets") / args.radar / args.sweep
output_path.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Reference dataset
# -------------------------------
with xr.open_dataset(path[0]) as ref_ds:
    crs_wkt = ref_ds.crs_wkt.attrs.get('crs_wkt')
    longitude = ref_ds.longitude.values
    latitude = ref_ds.latitude.values
    altitude = ref_ds.altitude.values
    elevation = np.unique(ref_ds.elevation.values)[0]
    cart_crs = CRS.from_wkt(crs_wkt)
    x_grid = ref_ds['x'].values
    y_grid = ref_ds['y'].values

# Precompute lon/lat grid once
transformer = Transformer.from_crs(cart_crs, "EPSG:4326", always_xy=True)
lon_grid, lat_grid = transformer.transform(x_grid, y_grid)

# -------------------------------
# Process and combine datasets
# -------------------------------
datasets = []
for i, file in enumerate(path, start=1):
    print(f"{i}/{len(path)}")
    with xr.open_dataset(file) as ds:
        ds = ds.reset_coords(drop=False).drop_vars(
            ['x', 'y', 'elevation', 'longitude', 'latitude', 'altitude', 'crs_wkt']
        )
        ds["longitude"] = (('azimuth', 'range'), lon_grid)
        ds["latitude"] = (('azimuth', 'range'), lat_grid)
        datasets.append(ds)

combined_dataset = xr.concat(datasets, dim='concat_dim')

# -------------------------------
# Add metadata
# -------------------------------
combined_dataset.attrs.update({
    'RADAR': args.radar,
    'RADAR lon': longitude,
    'RADAR lat': latitude,
    'RADAR height': f'{altitude} m',
    'RADAR elev_angle': f'{elevation} deg',
    'crs_wkt': crs_wkt,
    'AUTHOR': 'Marco Stefanelli',
    'MAIL': 'marco.stefanelli@fmf.uni-lj.si',
    'COMMENT': f'Concatenated ARSO RADAR dataset for {args.year} {args.month} at {args.sweep}'
})

# -------------------------------
# Save
# -------------------------------
outfile = output_path / f"{args.year}{args.month}.nc"
combined_dataset.to_netcdf(outfile)
print(f"Saved to {outfile}")
print('DONE!!!')
