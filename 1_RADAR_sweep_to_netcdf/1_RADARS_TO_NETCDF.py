"""
This script preprocesses radar data files by converting them from ODIM HDF5 
format to NetCDF, separated by radar sweep.

- Input:
  Reads a text file (--file) containing a list of radar data file paths 
  (one per line). Each file is expected to be in ODIM HDF5 format.

- Processing:
  • Opens each file with xradar and georeferences it.
  • Extracts the specified sweep (--sweep) and the "DBZH" variable.
  • Converts the data to NetCDF and saves it in a structured directory:
        <output_base>/<radar>/<sweep>/<filename>.nc
  • Monitors memory usage and cleans up after each file.
  • Dynamically adjusts the open file descriptor limit to handle large datasets.

- Output:
  NetCDF files organized by radar and sweep, written to:
      /scratch_ssd2/stefanelli/3DHNN_18_06_2025/
          2_DATASETS_PREPROCESSING_FOR_3DHNN/
          1_sweep_to_netcdf/NetCDF/single_nc_files_separated_by_sweeps/{radar}/{sweep}/

- Usage:
  Run via command line with required arguments:
      --year <YYYY> 
      --month <MM>
      --radar <PAZZ41|PAZZ42>
      --sweep <sweep_name>
      --file <path_to_filelist.txt>

- Purpose:
  Enables efficient batch preprocessing of large radar datasets, 
  preparing them in a uniform NetCDF format for later analysis or 
  use in machine learning pipelines (e.g., 3DHNN).

AUTHOR: Marco Stefanelli
"""






import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyproj.network")


import os
import pyproj

# Set PROJ path explicitly
os.environ['PROJ_LIB'] = "/scratch_ssd2/stefanelli/anaconda3/envs/3DHNN_preproc/share/proj"
pyproj.datadir.set_data_dir(os.environ['PROJ_LIB'])

import xradar as xd
import argparse
import datetime
import gc
import resource
import time
import psutil
from pathlib import Path



def read_file_paths(file_path):
    """Read file paths from a text file and return a stripped list."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def print_memory_usage():
    """Print current memory usage in MB."""
    mem_info = psutil.Process().memory_info()
    print(f'Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB', flush=True)

def ensure_dir(path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def process_file(file_path, sweep, output_base):
    """Open, process, and save radar data."""
    start_time = time.time()

    try:
        radar = xd.io.open_odim_datatree(file_path)
    except OSError as e:
        print(f"Skipping {file_path}: {e}")
        return

    radar = radar.xradar.georeference()[sweep]["DBZH"]

    output_dir = output_base / sweep
    ensure_dir(output_dir)

    out_file = output_dir / f"{Path(file_path).stem}.nc"
    radar.to_netcdf(out_file)

    del radar
    gc.collect()

    print(f"Processed: {file_path}")
    print(f"Saved to: {out_file}")
    print(f"Elapsed: {time.time() - start_time:.2f} sec", flush=True)
    print_memory_usage()
    print()

def main():
    start = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--year",  type=str, required=True)
    parser.add_argument("--month", type=str, required=True)
    parser.add_argument("--radar", type=str, required=True, choices=["PAZZ41", "PAZZ42"])
    parser.add_argument("--sweep", type=str, required=True)
    parser.add_argument("--file",  type=str, required=True)
    args = parser.parse_args()

    print(f"\nProcessing {args.file}")
    print(f"Radar: {args.radar} | Year: {args.year} | Month: {args.month} | Sweep: {args.sweep}")
    print('\n' + '-' * 70 + '\n', flush=True)

    # Read dataset list
    filenames = read_file_paths(args.file)
    print(f"Found {len(filenames)} files", flush=True)

    # Adjust open file limit
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft_limit = min(len(filenames) + 1000, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
    # Prepare output base path
    base_path=Path('/scratch_ssd2/stefanelli/3DHNN_18_06_2025/2_DATASETS_PREPROCESSING_FOR_3DHNN/1_sweep_to_netcdf/NetCDF/single_nc_files_separated_by_sweeps') / args.radar


    # Process each file
    for idx, file_path in enumerate(filenames, start=1):
        print(f"{idx}/{len(filenames)}")
        process_file(file_path, args.sweep, base_path)

    print(f"\nStarted: {start}")
    print(f"Finished: {datetime.datetime.now()}")
    print("DONE!!!")

if __name__ == "__main__":
    main()
