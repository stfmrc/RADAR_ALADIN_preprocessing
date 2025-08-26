#!/bin/bash
# -------------------------------------------------------------------
# Script: parallel_submit_hourly_cumulated_dbz.sh
# Purpose: Automates the submission of hourly cumulated radar DBZH 
#          processing jobs in parallel, with throttling to limit the 
#          number of concurrent Python processes.
#
# Input:
#   - Preprocessed monthly concatenated NetCDF files for each radar 
#     sweep, located under:
#       hourly_cumulated_dbzh/{radar}/{sweep}/{YYYYMM}.nc
#   - Python script performing the hourly cumulation:
#       0_RADAR_hourly_cumulated_dbz.py
#
# Processing:
#   • Iterates over configured RADAR, SWEEPS, YEARS, and MONTHS.
#   • Submits one Python job per (sweep, year, month) combination.
#   • Limits parallel execution to MAX_JOBS (default: 3), checking 
#     every SLEEP_BETWEEN_CHECKS seconds.
#   • Redirects stdout and stderr of each job to timestamped log files.
#
# Output:
#   - Hourly cumulated DBZH NetCDF files (produced by the Python script).
#   - Logs saved in:
#       0_RADAR_NETCDF_hourly_cumulated_dbz_LOGS/{sweep}/
#       (separate output_*.log and error_*.log per job)
#
# Usage:
#   Run with nohup to detach and capture logs:
#       nohup ./0_submit.sh > 0_submit.sh_output.log 2> 0_submit.sh_error.log &
#
# Notes:
#   - Adjust RADAR, SWEEPS, YEARS, MONTHS to select datasets.
#   - MAX_JOBS controls parallelization degree (CPU usage).
#   - A unique timestamp (TS) is appended to log filenames for traceability.
# -------------------------------------------------------------------

# -------------------------------
# CONFIGURATION
# -------------------------------
RADAR="PAZZ41"                   # or "PAZZ42"
SWEEPS=(sweep_0)         # adjust sweeps to run
YEARS=(2023) #(2019 2020 2021 2022 2023)
MONTHS=(08 09 10 11 12) #(01 02 03 04 05 06 07 08 09 10 11 12)

MAX_JOBS=3                # Number of parallel jobs (CPU cores)
SLEEP_BETWEEN_CHECKS=5           # Seconds between job slot checks

LOG_BASE_DIR="0_RADAR_NETCDF_hourly_cumulated_dbz_LOGS"
SCRIPT="0_RADAR_hourly_cumulated_dbz.py"

# Timestamp for uniqueness
TS=$(date +%Y%m%d_%H%M%S)

# -------------------------------
# FUNCTION: Wait until a job slot is free
# -------------------------------
wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep "$SLEEP_BETWEEN_CHECKS"
    done
}

# -------------------------------
# MAIN LOOP
# -------------------------------
mkdir -p "$LOG_BASE_DIR"

for sweep in "${SWEEPS[@]}"; do
    mkdir -p "$LOG_BASE_DIR/$sweep"

    for year in "${YEARS[@]}"; do
        for month in "${MONTHS[@]}"; do
            # Wait until a CPU slot is free
            wait_for_slot

            OUT_LOG="$LOG_BASE_DIR/$sweep/output_${RADAR}_${year}${month}_${TS}.log"
            ERR_LOG="$LOG_BASE_DIR/$sweep/error_${RADAR}_${year}${month}_${TS}.log"

            # Launch job in background
            nohup stdbuf -oL -eL python "$SCRIPT" \
                --radar="$RADAR" \
                --year="$year" \
                --month="$month" \
                --sweep="$sweep" \
                > "$OUT_LOG" 2> "$ERR_LOG" &

            PID=$!
            echo "Submitted: $RADAR $sweep $year-$month -> PID: $PID"
        done
    done
done

# Wait for all jobs to finish
wait
echo "All submissions finished at $(date)"
