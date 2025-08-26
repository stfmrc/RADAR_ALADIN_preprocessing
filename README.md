# RADAR–ALADIN Preprocessing Pipeline
The directory contains the codes developed during the SMASH Postdoc in the period April 2024-April 2026

This repository contains a complete preprocessing pipeline for preparing **RADAR reflectivity (DBZH)** and **ALADIN model outputs** for machine learning applications (e.g., observation operator training, precipitation nowcasting with neural networks).

The pipeline converts raw data into aligned and interpolated NetCDF datasets suitable for training and analysis.

---

## Repository Structure

The workflow is organized into sequential steps, each stored in its own directory:

### 1. `1_RADAR_sweep_to_netcdf`
- Scripts to convert **raw RADAR sweeps** into NetCDF format.
- Handles multiple sweeps and ensures compatibility with downstream processing.

### 2. `2_RADAR_monthy_concatenated_dataset`
- Concatenates monthly RADAR NetCDF files.
- Runs in parallel to accelerate processing across years/months.

### 3. `3_RADAR_cumulated_dbzh`
- Produces **hourly cumulated reflectivity (DBZH)** datasets from sweep-based data.
- Includes job submission scripts for parallel runs.

### 4. `4_ALADIN_GRIB_to_NetCDF`
- Converts **ALADIN GRIB files** into NetCDF.
- Keeps selected meteorological variables (temperature, wind, pressure, etc.).

### 5. `5_RADAR_ALADIN_Concatenate_datasets`
- Concatenates **RADAR and ALADIN** datasets over the full study period (2019–2023).
- Performs filtering and reduction of unnecessary variables.

### 6. `6_RADAR_Clean_zeros_timesteps_and_outliers`
- Removes time steps with corrupted/zero DBZH values.
- Filters outliers in the RADAR reflectivity fields.

### 7. `7_ALADIN_Time_alignment`
- Ensures **temporal alignment** between ALADIN and RADAR datasets.
- Reindexes ALADIN to RADAR timestamps without interpolation.

### 8. `8_RADAR_Interpolate_on_aladin`
- Interpolates **RADAR DBZH** onto the **ALADIN spatial grid**.
- Produces the final dataset with matched spatial/temporal dimensions for analysis.

---

## Installation

The repository provides a conda environment specification:

```bash
conda env create -f environment.yml
conda activate 3DHNN_preproc
