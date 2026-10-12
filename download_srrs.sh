#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Parallel SRA downloader
#
# Each accession is processed independently:
#   prefetch -> fasterq-dump -> pigz -> remove .sra
#
# Requirements:
#   sra-tools >= 3.x recommended
#   GNU parallel
#   pigz
#
# Install:
#   conda install -c conda-forge -c bioconda \
#       sra-tools parallel pigz
# ============================================================


# -----------------------------
# SRA accessions
# -----------------------------
ACCESSIONS=(
    SRR3458591
    SRR3458592
    SRR3458593
    SRR10315320
    SRR10315321
    SRR10315322
    SRR14510018
    SRR14510019
)


# -----------------------------
# Parallelization settings
# -----------------------------
#
# Number of datasets processed simultaneously
JOBS=4
#
# Threads used by fasterq-dump / pigz PER DATASET
THREADS=4


# -----------------------------
# Directories
# -----------------------------
BASE_DIR="$PWD/SRA_downloads"

SRA_DIR="$BASE_DIR/sra"
FASTQ_DIR="$BASE_DIR/fastq"
TMP_DIR="$BASE_DIR/tmp"

mkdir -p "$SRA_DIR"
mkdir -p "$FASTQ_DIR"
mkdir -p "$TMP_DIR"


# -----------------------------
# Check programs
# -----------------------------
for PROGRAM in prefetch fasterq-dump pigz parallel; do

    if ! command -v "$PROGRAM" >/dev/null 2>&1; then
        echo "ERROR: $PROGRAM not found."
        exit 1
    fi

done


echo
echo "============================================================"
echo "Parallel SRA downloader"
echo "============================================================"
echo
echo "Datasets:           ${#ACCESSIONS[@]}"
echo "Parallel jobs:      $JOBS"
echo "Threads per job:    $THREADS"
echo "Maximum CPU usage:  approximately $((JOBS * THREADS)) threads"
echo
echo "FASTQ output:"
echo "  $FASTQ_DIR"
echo
echo "Temporary files:"
echo "  $TMP_DIR"
echo
echo "============================================================"
echo


# ============================================================
# Process one accession
# ============================================================

process_one() {

    ACC="$1"

    echo "[$(date '+%H:%M:%S')] STARTING $ACC"


    # --------------------------------------------------------
    # Check whether output already exists
    # --------------------------------------------------------

    if [[ -f "$FASTQ_DIR/${ACC}.fastq.gz" ]]; then

        echo "[$(date '+%H:%M:%S')] $ACC already complete -- skipping"
        return 0

    fi


    if [[ -f "$FASTQ_DIR/${ACC}_1.fastq.gz" &&
          -f "$FASTQ_DIR/${ACC}_2.fastq.gz" ]]; then

        echo "[$(date '+%H:%M:%S')] $ACC already complete -- skipping"
        return 0

    fi


    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    echo "[$(date '+%H:%M:%S')] $ACC downloading..."


    # Try up to 3 times
    SUCCESS=0

    for TRY in 1 2 3; do

        echo "[$(date '+%H:%M:%S')] $ACC download attempt $TRY/3"

        if prefetch \
            --progress \
            --max-size u \
            -O "$SRA_DIR" \
            "$ACC"
        then
            SUCCESS=1
            break
        fi

        echo "[$(date '+%H:%M:%S')] WARNING: $ACC download failed"

        sleep 10

    done


    if [[ "$SUCCESS" -ne 1 ]]; then
        echo "[$(date '+%H:%M:%S')] ERROR: $ACC failed after 3 attempts"
        return 1
    fi


    SRA_FILE="$SRA_DIR/$ACC/$ACC.sra"


    if [[ ! -f "$SRA_FILE" ]]; then

        echo "[$(date '+%H:%M:%S')] ERROR: SRA file not found:"
        echo "$SRA_FILE"

        return 1

    fi


    echo "[$(date '+%H:%M:%S')] $ACC download complete"

    ls -lh "$SRA_FILE"


    # --------------------------------------------------------
    # Create accession-specific temporary directory
    # --------------------------------------------------------

    ACC_TMP="$TMP_DIR/$ACC"

    rm -rf "$ACC_TMP"
    mkdir -p "$ACC_TMP"


    # --------------------------------------------------------
    # SRA -> FASTQ
    # --------------------------------------------------------

    echo "[$(date '+%H:%M:%S')] $ACC converting to FASTQ..."


    if ! fasterq-dump \
        "$SRA_FILE" \
        --split-files \
        --threads "$THREADS" \
        --outdir "$FASTQ_DIR" \
        --temp "$ACC_TMP" \
        --progress
    then

        echo "[$(date '+%H:%M:%S')] ERROR: fasterq-dump failed for $ACC"

        rm -rf "$ACC_TMP"

        return 1

    fi


    rm -rf "$ACC_TMP"


    echo "[$(date '+%H:%M:%S')] $ACC FASTQ conversion complete"


    # --------------------------------------------------------
    # Find FASTQ files
    # --------------------------------------------------------

    shopt -s nullglob

    FASTQ_FILES=(
        "$FASTQ_DIR"/"${ACC}"*.fastq
    )

    shopt -u nullglob


    if [[ ${#FASTQ_FILES[@]} -eq 0 ]]; then

        echo "[$(date '+%H:%M:%S')] ERROR: No FASTQ files for $ACC"

        return 1

    fi


    # --------------------------------------------------------
    # Compress FASTQ
    # --------------------------------------------------------

    echo "[$(date '+%H:%M:%S')] $ACC compressing FASTQ..."


    for FILE in "${FASTQ_FILES[@]}"; do

        echo "[$(date '+%H:%M:%S')] $ACC compressing $(basename "$FILE")"

        pigz \
            -p "$THREADS" \
            -v \
            "$FILE"

    done


    # --------------------------------------------------------
    # Remove downloaded SRA
    # --------------------------------------------------------

    echo "[$(date '+%H:%M:%S')] $ACC removing SRA copy..."

    rm -rf "$SRA_DIR/$ACC"


    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    echo
    echo "[$(date '+%H:%M:%S')] COMPLETED $ACC"

    find "$FASTQ_DIR" \
        -maxdepth 1 \
        -type f \
        -name "${ACC}*.fastq.gz" \
        -exec ls -lh {} \;

    echo

}


# Make function and variables available to GNU parallel
export -f process_one

export SRA_DIR
export FASTQ_DIR
export TMP_DIR
export THREADS


# ============================================================
# RUN DATASETS IN PARALLEL
# ============================================================

parallel \
    --jobs "$JOBS" \
    --line-buffer \
    --tag \
    process_one {} \
    ::: "${ACCESSIONS[@]}"


# ============================================================
# Summary
# ============================================================

echo
echo
echo "============================================================"
echo "ALL JOBS FINISHED"
echo "============================================================"
echo

echo "FASTQ directory:"
echo "$FASTQ_DIR"

echo
echo "Files:"
echo

ls -lh "$FASTQ_DIR"/*.fastq.gz 2>/dev/null || true

echo
echo "Total compressed size:"
du -sh "$FASTQ_DIR"

echo
echo "============================================================"