#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Download everything the pipeline needs:
#
#   1. the 8 SRA runs, in parallel:
#        prefetch -> fasterq-dump -> pigz -> remove .sra
#   2. the reference genomes, annotations, bowtie2 indexes and rRNA
#      sequences (scripts/remote/fetch_data.py --skip-fastq)
#
# Re-runnable: everything already present is skipped.
#   - a run is complete when <ACC>.fastq.gz or <ACC>_1.fastq.gz exists, is
#     non-empty, and no uncompressed <ACC>*.fastq is left beside it
#   - an interrupted compression (.fastq still there) is finished without
#     re-downloading
#   - an already-downloaded .sra is reused by prefetch
#   - each reference file is skipped when present
#
# Requirements:
#   sra-tools >= 3.x, GNU parallel, pigz   (FASTQs)
#   python3, bowtie2 (bowtie2-build-s)     (references; the utrscan env has both)
#
# Install:
#   conda install -c conda-forge -c bioconda \
#       sra-tools parallel pigz bowtie2 python
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
# Pinned to the repository (where this script lives), not the current
# directory, so the pipeline always finds the FASTQs in <repo>/fastq.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$REPO"

SRA_DIR="$BASE_DIR/sra"
FASTQ_DIR="$BASE_DIR/fastq"
TMP_DIR="$BASE_DIR/tmp"

mkdir -p "$SRA_DIR"
mkdir -p "$FASTQ_DIR"
mkdir -p "$TMP_DIR"


# ============================================================
# Is a run already complete?
# ============================================================

is_complete() {

    local ACC="$1"
    local GZ=""

    for CAND in "$FASTQ_DIR/${ACC}.fastq.gz" "$FASTQ_DIR/${ACC}_1.fastq.gz"; do
        if [[ -s "$CAND" ]]; then
            GZ="$CAND"
            break
        fi
    done

    [[ -n "$GZ" ]] || return 1

    # pigz deletes the .fastq only after a successful compression; one still
    # present means the .gz beside it is incomplete.
    shopt -s nullglob
    local LEFTOVER=( "$FASTQ_DIR"/"${ACC}"*.fastq )
    shopt -u nullglob
    [[ ${#LEFTOVER[@]} -eq 0 ]]
}


# -----------------------------
# Which runs still need work?
# -----------------------------
TODO=()

for ACC in "${ACCESSIONS[@]}"; do
    if is_complete "$ACC"; then
        echo "present:  $ACC  -- skipping"
    else
        TODO+=( "$ACC" )
    fi
done


# ============================================================
# Process one accession
# ============================================================

process_one() {

    ACC="$1"

    echo "[$(date '+%H:%M:%S')] STARTING $ACC"


    # --------------------------------------------------------
    # Interrupted compression: finish it, no re-download
    # --------------------------------------------------------

    shopt -s nullglob
    FASTQ_FILES=( "$FASTQ_DIR"/"${ACC}"*.fastq )
    shopt -u nullglob

    if [[ ${#FASTQ_FILES[@]} -gt 0 ]]; then

        echo "[$(date '+%H:%M:%S')] $ACC found uncompressed FASTQ -- finishing compression"

    else

        # ----------------------------------------------------
        # Download (prefetch skips a complete .sra already on disk)
        # ----------------------------------------------------

        echo "[$(date '+%H:%M:%S')] $ACC downloading..."

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


        # ----------------------------------------------------
        # Remove partial outputs from an earlier failed attempt
        # ----------------------------------------------------

        rm -f "$FASTQ_DIR/${ACC}.fastq.gz" "$FASTQ_DIR/${ACC}"_*.fastq.gz


        # ----------------------------------------------------
        # Create accession-specific temporary directory
        # ----------------------------------------------------

        ACC_TMP="$TMP_DIR/$ACC"

        rm -rf "$ACC_TMP"
        mkdir -p "$ACC_TMP"


        # ----------------------------------------------------
        # SRA -> FASTQ
        # ----------------------------------------------------

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


        shopt -s nullglob
        FASTQ_FILES=( "$FASTQ_DIR"/"${ACC}"*.fastq )
        shopt -u nullglob


        if [[ ${#FASTQ_FILES[@]} -eq 0 ]]; then

            echo "[$(date '+%H:%M:%S')] ERROR: No FASTQ files for $ACC"

            return 1

        fi

    fi


    # --------------------------------------------------------
    # Compress FASTQ (-f: overwrite a partial .gz from an interrupted run)
    # --------------------------------------------------------

    echo "[$(date '+%H:%M:%S')] $ACC compressing FASTQ..."


    for FILE in "${FASTQ_FILES[@]}"; do

        echo "[$(date '+%H:%M:%S')] $ACC compressing $(basename "$FILE")"

        pigz \
            -f \
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


# ============================================================
# 1. FASTQs
# ============================================================

FASTQ_FAILED=0

if [[ ${#TODO[@]} -eq 0 ]]; then

    echo
    echo "All ${#ACCESSIONS[@]} FASTQ runs already present."

else

    for PROGRAM in prefetch fasterq-dump pigz parallel; do

        if ! command -v "$PROGRAM" >/dev/null 2>&1; then
            echo "ERROR: $PROGRAM not found (needed for ${TODO[*]})."
            exit 1
        fi

    done


    echo
    echo "============================================================"
    echo "Parallel SRA downloader"
    echo "============================================================"
    echo
    echo "To download:        ${#TODO[@]} of ${#ACCESSIONS[@]}  (${TODO[*]})"
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


    # Make function and variables available to GNU parallel
    export -f process_one

    export SRA_DIR
    export FASTQ_DIR
    export TMP_DIR
    export THREADS


    # A failed run must not stop the references from being fetched; the
    # failure is reported at the end and the script exits non-zero.
    parallel \
        --jobs "$JOBS" \
        --line-buffer \
        --tag \
        process_one {} \
        ::: "${TODO[@]}" || FASTQ_FAILED=1

fi


# ============================================================
# 2. References (genomes, annotations, bowtie2 indexes, rRNA)
# ============================================================

echo
echo "============================================================"
echo "References"
echo "============================================================"
echo

REF_FAILED=0

if ! command -v python3 >/dev/null 2>&1; then

    echo "ERROR: python3 not found -- cannot fetch references."
    echo "       Activate the utrscan environment (conda activate utrscan) and re-run."
    REF_FAILED=1

elif ! command -v bowtie2-build-s >/dev/null 2>&1 && ! command -v bowtie2-build >/dev/null 2>&1; then

    echo "ERROR: bowtie2-build not found -- needed to build the yeast index."
    echo "       Activate the utrscan environment (conda activate utrscan) and re-run."
    REF_FAILED=1

else

    ( cd "$REPO" && python3 scripts/remote/fetch_data.py --skip-fastq ) || REF_FAILED=1

fi


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

MISSING=()
for ACC in "${ACCESSIONS[@]}"; do
    is_complete "$ACC" || MISSING+=( "$ACC" )
done

if [[ ${#MISSING[@]} -gt 0 || "$FASTQ_FAILED" -ne 0 || "$REF_FAILED" -ne 0 ]]; then
    [[ ${#MISSING[@]} -gt 0 ]] && echo "INCOMPLETE FASTQs: ${MISSING[*]}"
    [[ "$REF_FAILED" -ne 0 ]] && echo "References: FAILED (see messages above)"
    echo "Re-run this script to retry; everything already present is skipped."
    echo "============================================================"
    exit 1
fi

echo "All FASTQs and references present."
echo "============================================================"
