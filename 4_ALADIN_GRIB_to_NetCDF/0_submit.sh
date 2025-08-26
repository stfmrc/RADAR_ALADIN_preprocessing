#!/bin/bash
# -------------------------------------------------------------------
# Script: parallel_submit_grib_to_netcdf.sh
# Purpose: Automates the conversion of ALADIN GRIB files to NetCDF 
#          format in parallel, with throttling to limit CPU usage.
#
# Input:
#   - ALADIN GRIB files stored under:
#       /scratch_ssd2/stefanelli/3DHNN_18_06_2025/0_DATASETS_FROM_ARSO_AND_UNZIPPED/
#   - Python script performing the conversion:
#       0_GRIB_TO_NETCDF.py
#   - Configurable years and months (YEARS, MONTHS arrays)
#
# Processing:
#   • Iterates over specified YEARS and MONTHS.
#   • Waits for available CPU slots before submitting new jobs 
#     to respect MAX_JOBS.
#   • Launches each Python conversion job in the background using nohup.
#   • Redirects stdout and stderr to log files for monitoring.
#
# Output:
#   - NetCDF files produced by 0_GRIB_TO_NETCDF.py
#   - Logs saved in:
#       0_GRIB_TO_NETCDF_LOGS/output_YYYYMM.log
#       0_GRIB_TO_NETCDF_LOGS/error_YYYYMM.log
#
# Usage:
#   Run with nohup to detach and capture logs:
#       nohup ./0_submit.sh > 0_submit.sh_output.log 2> 0_submit.sh_error.log &
#
# Notes:
#   - MAX_JOBS controls parallelization degree (CPU usage).
#   - SLEEP_BETWEEN_CHECKS defines the interval between checking job slots.
#   - Each job processes one year-month combination at a time.
# -------------------------------------------------------------------



# -------------------------------
# CONFIG
# -------------------------------
YEARS=(2019 2020 2021 2022 2023)
MONTHS=(01 02 03 04 05 06 07 08 09 10 11 12)

MAX_JOBS=2        # use all CPU cores, or set manually (e.g. 8)
SLEEP_BETWEEN_CHECKS=5   # seconds between slot checks

LOG_DIR="0_GRIB_TO_NETCDF_LOGS"
SCRIPT="0_GRIB_TO_NETCDF.py"

mkdir -p "$LOG_DIR"

# -------------------------------
# FUNCTION: wait until job slot is free
# -------------------------------
wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep "$SLEEP_BETWEEN_CHECKS"
    done
}

# -------------------------------
# MAIN LOOP
# -------------------------------
for year in "${YEARS[@]}"; do
    for month in "${MONTHS[@]}"; do
        
        # Wait for an available CPU slot
        wait_for_slot

        OUT_LOG="${LOG_DIR}/output_${year}${month}.log"
        ERR_LOG="${LOG_DIR}/error_${year}${month}.log"

        # Launch in background
        nohup stdbuf -oL -eL python "$SCRIPT" \
            --year="$year" --month="$month" \
            > "$OUT_LOG" 2> "$ERR_LOG" &

        PID=$!
        echo "Submitted: ${year}${month} -> PID: $PID"

    done
done

# -------------------------------
# Wait for all background jobs
# -------------------------------
wait
echo "All submissions finished at $(date)"
