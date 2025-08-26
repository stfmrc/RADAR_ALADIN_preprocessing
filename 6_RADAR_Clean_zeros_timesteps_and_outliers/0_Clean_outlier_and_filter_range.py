"""
Script: 0_Clean_outlier_and_filter_range.py
Purpose: Clean RADAR reflectivity dataset (DBZH) by removing empty timesteps
         and filtering outliers using a global Mahalanobis distance approach.

Input:
  - INPUT_FILE: Path to the concatenated RADAR dataset (NetCDF)
                e.g., RADAR_2019_2023.nc
  - Parameters:
      • DBZ_VAR: Reflectivity variable name in dataset
      • TIME_DIM: Time dimension name
      • MAX_RANGE: Maximum radar range to retain (meters)
      • MIN_DBZ_KEEP: Minimum reflectivity to retain (dBZ)
      • SAMPLE_FRAC: Fraction of time steps used for histogram estimation
      • N_BINS / BIN_WIDTH: Histogram binning parameters
      • PERCENTILE: Percentile threshold for outlier detection

Processing Steps:
  1. Load the RADAR dataset using xarray.
  2. Identify and remove timesteps where all DBZH values are zero.
  3. Build a global outlier mask using:
      • Histogram-based estimation of reflectivity per gate
      • Mahalanobis distance computation across all gates
      • Classification of the top percentile of distances as outliers
  4. Apply the mask to filter out outlier values and retain only data within MAX_RANGE.
  5. Store the outlier mask in the dataset as DBZH_mask (1=outlier, 0=valid).

Output:
  - OUTPUT_FILE: Cleaned RADAR dataset with masked outliers saved in NetCDF format
                 e.g., RADAR_2019_2023_cleaned_zeros_time_and_outliers_filtered_160Km.nc

Usage:
  - Run in background to detach from terminal:
      nohup python 0_Clean_outlier_and_filter_range.py > 0_output.log 2> 0_error.log &

Notes:
  - Uses incremental memory handling and garbage collection for large datasets.
  - Random seed ensures reproducibility of sampled histogram-based outlier detection.
  - Prints diagnostic information including number of removed timesteps and outlier fraction.

AUTHOR: Marco Stefanelli
"""

# ==============================
# Parameters (User-defined)
# ==============================
INPUT_FILE   = "/scratch_ssd2/stefanelli/3DHNN_18_06_2025/2_DATASETS_PREPROCESSING_FOR_3DHNN/5_RADAR_ALADIN_Concatenate_datasets/RADAR_2019_2023.nc"
OUTPUT_FILE  = "RADAR_2019_2023_cleaned_zeros_time_and_outliers_filtered_160Km.nc"

DBZ_VAR      = "DBZH"       # Reflectivity variable name in dataset
TIME_DIM     = "concat_dim" # Time dimension name
MAX_RANGE    = 160000.0    # Maximum radar range to consider [m]. IN the RADAR dataset the maximum is 248500
MIN_DBZ_KEEP = 13.5         # Minimum reflectivity to keep [dBZ]

SAMPLE_FRAC  = 0.2          # Fraction of time samples used for histogram estimation
N_BINS       = 526          # Number of histogram bins
BIN_WIDTH    = 0.1          # Histogram bin width [dBZ]
PERCENTILE   = 0.90         # Percentile for threshold

RANDOM_SEED  = 42           # RNG seed for reproducibility
VERBOSE      = True         # Print progress information
# ==============================


# --- Imports ---
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gc


def build_global_outlier_mask(ds, dbz_var=DBZ_VAR, time_dim=TIME_DIM,
                              max_range=MAX_RANGE, min_dbz_keep=MIN_DBZ_KEEP,
                              sample_frac=SAMPLE_FRAC, n_bins=N_BINS, bin_width=BIN_WIDTH, percentile=PERCENTILE,
                              random_seed=RANDOM_SEED, verbose=VERBOSE):
    """
    Compute a global outlier mask for radar reflectivity data.

    Parameters
    ----------
    ds : xarray.Dataset
        Input radar dataset.
    dbz_var : str
        Variable name for reflectivity (dBZ).
    time_dim : str
        Name of time dimension.
    max_range : float
        Maximum radar range to consider (m).
    min_dbz_keep : float
        Minimum reflectivity to retain (dBZ).
    sample_frac : float
        Fraction of time samples used for building histograms.
    n_bins : int
        Number of histogram bins.
    bin_width : float
        Histogram bin width (dBZ).
    random_seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress info.

    Returns
    -------
    mask : xarray.DataArray
        Boolean mask (azimuth, range) marking outlier gates.
    distances : np.ndarray
        Mahalanobis distances for each gate.
    threshold : float
        Distance threshold for classifying outliers.
    """

    # 1. Apply reflectivity and range filters
    dbz = ds[dbz_var].where(ds[dbz_var] >= min_dbz_keep)
    dbz = dbz.where(ds.range <= max_range)

    # 2. Collapse azimuth & range into a single "gate" dimension
    stacked = dbz.stack(gate=("azimuth", "range"))

    # 3. Randomly sample time steps for efficiency
    rng = np.random.default_rng(random_seed)
    n_time = stacked.sizes[time_dim]
    n_sample = max(1, int(np.ceil(sample_frac * n_time)))
    sample_idx = np.sort(rng.choice(n_time, size=n_sample, replace=False))
    stacked_sample = stacked.isel({time_dim: sample_idx})

    # 4. Convert to NumPy array (time × gates)
    A = stacked_sample.to_numpy()
    n_sample, n_gates = A.shape

    # 5. Build histograms per gate
    qmin = np.nanquantile(A, 0.01)  # robust min
    hist_min_dbz = np.floor(qmin * 10.0) / 10.0
    hist_max_dbz = hist_min_dbz + n_bins * bin_width
    bin_edges = np.linspace(hist_min_dbz, hist_max_dbz, n_bins + 1, dtype=np.float32)

    H = np.zeros((n_gates, n_bins), dtype=np.float32)
    for j in range(n_gates):
        col = A[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        hist, _ = np.histogram(col, bins=bin_edges)
        H[j, :] = hist

    # 6. Normalize histograms (probability distributions per gate)
    row_sums = H.sum(axis=1, keepdims=True)
    nonzero = row_sums.squeeze() > 0
    H[nonzero, :] /= row_sums[nonzero]

    # 7. Compute Mahalanobis distances
    mu = H.mean(axis=0, dtype=np.float64)
    X = H.astype(np.float64) - mu[None, :]
    Sigma = np.cov(X, rowvar=False, bias=False)
    eps = 1e-6  # regularization
    Sigma_reg = Sigma + eps * np.eye(Sigma.shape[0])
    Sigma_pinv = np.linalg.pinv(Sigma_reg)
    distances = np.sqrt(np.einsum("ij,jk,ik->i", X, Sigma_pinv, X))

    #Classify the top (percentile - 1)*100% farthest-away gates (in Mahalanobis distance space) as outliers.
    threshold = np.quantile(distances, percentile)  # top 5% flagged

    # 8. Build mask in azimuth–range space
    outlier = distances > threshold
    mask = xr.DataArray(outlier.reshape((ds.dims["azimuth"], ds.dims["range"])),
                        coords={"azimuth": ds.azimuth, "range": ds.range},
                        dims=("azimuth", "range"))

    return mask, distances, threshold


# ==============================
# Main Script
# ==============================

# Load the radar dataset from the specified input file
ds1 = xr.open_dataset(INPUT_FILE)
print("Dataset loaded:", ds1)

# Extract the DBZH (radar reflectivity) values as a NumPy array
dbzh = ds1.DBZH.values

# Identify timesteps where all DBZH values are zero (completely empty scans)
# np.all(..., axis=(1, 2)) checks each timestep (first dimension) across all spatial points
# This issue cames from step 3 in preprocessing when initializing arrays as zeros
idx = np.where(np.all(dbzh == 0, axis=(1, 2)))[0]
print(f'\nNumber of timesteps with all zero values --> {idx.shape[0]}')

# Create a boolean mask to keep only timesteps with non-zero values
mask = np.ones(dbzh.shape[0], dtype=bool)
mask[idx] = False

# Apply the mask to the dataset to filter out empty timesteps
ds = ds1.isel(concat_dim=mask)
print(ds)

# Clean up memory by deleting large temporary arrays
del ds1, dbzh, mask, idx
gc.collect()

# Compute outlier mask
mask, distances, threshold = build_global_outlier_mask(ds)

# Number of discarded (masked) values
n_discarded = int(mask.sum().item())
n_total = mask.size
frac_discarded = n_discarded / n_total * 100

print(f"Threshold distance is {threshold:.3f}")
print(f"Discarded values: {n_discarded} / {n_total} "
      f"({frac_discarded:.2f}% of all values)")

# Store the mask in the dataset
ds["DBZH_mask"] = mask.astype("int8")  # 1 = outlier, 0 = valid
ds["DBZH_mask"].attrs["description"] = "Outlier mask for DBZH (1=outlier, 0=valid)"
ds["DBZH_mask"].attrs["method"] = f"Global Mahalanobis distance outlier detection using {int(PERCENTILE*100)}th percentile"
ds["DBZH_mask"].attrs["threshold"] = float(threshold)

# Expand mask along time dimension
mask_expanded = mask.broadcast_like(ds[DBZ_VAR])

# Apply mask to clean reflectivity data
DBZH_clean = ds[DBZ_VAR].where((~mask_expanded) & (ds['range'] <= MAX_RANGE))

# Diagnostics
print("Mask True count:", mask.sum().item())
print("Mask shape:", mask.shape)
print("Any outliers?", bool(mask.sum() > 0))

# Replace variable in dataset and save
ds[DBZ_VAR] = DBZH_clean
ds.to_netcdf(OUTPUT_FILE, engine="netcdf4", format="NETCDF4")
print("Cleaned dataset saved to:", OUTPUT_FILE)
print('DONE!!')