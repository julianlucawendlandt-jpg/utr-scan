#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# SRA download -> FASTQ -> gzip
# Requires:
#   sra-tools
# Optional but recommended:
#   pigz
#
# Install compression tool with:
#   conda install -c conda-forge pigz
# ============================================================

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

# Number of CPU threads
THREADS=8

# Output directories
BASE_DIR="$PWD/SRA_downloads"
SRA_DIR="$BASE_DIR/sra"
FASTQ_DIR="$BASE_DIR/fastq"
TMP_DIR="$BASE_DIR/tmp"

mkdir -p "$SRA_DIR" "$FASTQ_DIR" "$TMP_DIR"

# ------------------------------------------------------------
# Pretty status messages
# ------------------------------------------------------------
status() {
    echo
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "============================================================"
}

# ------------------------------------------------------------
# Check required programs
# ------------------------------------------------------------
command -v prefetch >/dev/null 2>&1 || {
    echo "ERROR: prefetch not found."
    echo "Make sure sra-tools is installed/activated."
    exit 1
}

command -v fasterq-dump >/dev/null 2>&1 || {
    echo "ERROR: fasterq-dump not found."
    echo "Make sure sra-tools is installed/activated."
    exit 1
}

# Prefer pigz because it uses multiple CPU cores.
if command -v pigz >/dev/null 2>&1; then
    COMPRESSOR="pigz"
    status "Using pigz with $THREADS threads for compression"
else
    COMPRESSOR="gzip"
    status "pigz not found; using standard gzip"
    echo "For faster compression:"
    echo "  conda install -c conda-forge pigz"
fi

TOTAL=${#ACCESSIONS[@]}
COUNT=0

# ------------------------------------------------------------
# Process each accession
# ------------------------------------------------------------
for ACC in "${ACCESSIONS[@]}"; do

    COUNT=$((COUNT + 1))

    status "[$COUNT/$TOTAL] Processing $ACC"

    # --------------------------------------------------------
    # Skip if compressed output already exists
    # --------------------------------------------------------
    if [[ -f "$FASTQ_DIR/${ACC}.fastq.gz" ]]; then
        echo "$ACC already exists as single-end FASTQ."
        echo "Skipping."
        continue
    fi

    if [[ -f "$FASTQ_DIR/${ACC}_1.fastq.gz" &&
          -f "$FASTQ_DIR/${ACC}_2.fastq.gz" ]]; then
        echo "$ACC already exists as paired-end FASTQ."
        echo "Skipping."
        continue
    fi

    # --------------------------------------------------------
    # 1. Download SRA
    # --------------------------------------------------------
    status "[$COUNT/$TOTAL] Downloading $ACC"

    prefetch \
        --progress \
        --max-size u \
        -O "$SRA_DIR" \
        "$ACC"

    SRA_FILE="$SRA_DIR/$ACC/$ACC.sra"

    if [[ ! -f "$SRA_FILE" ]]; then
        echo "ERROR: Could not find:"
        echo "  $SRA_FILE"
        exit 1
    fi

    echo
    echo "Downloaded:"
    ls -lh "$SRA_FILE"

    # --------------------------------------------------------
    # 2. Convert SRA -> FASTQ
    # --------------------------------------------------------
    status "[$COUNT/$TOTAL] Converting $ACC to FASTQ"

    mkdir -p "$TMP_DIR/$ACC"

    fasterq-dump \
        "$SRA_FILE" \
        --split-files \
        -e "$THREADS" \
        -O "$FASTQ_DIR" \
        -t "$TMP_DIR/$ACC" \
        -p

    # Temporary fasterq files are no longer needed
    rm -rf "$TMP_DIR/$ACC"

    echo
    echo "FASTQ files created:"
    find "$FASTQ_DIR" \
        -maxdepth 1 \
        -type f \
        -name "${ACC}*.fastq" \
        -exec ls -lh {} \;

    # --------------------------------------------------------
    # 3. Compress FASTQ files
    # --------------------------------------------------------
    status "[$COUNT/$TOTAL] Compressing $ACC"

    shopt -s nullglob
    FASTQ_FILES=("$FASTQ_DIR"/"${ACC}"*.fastq)
    shopt -u nullglob

    if [[ ${#FASTQ_FILES[@]} -eq 0 ]]; then
        echo "ERROR: No FASTQ files found for $ACC"
        exit 1
    fi

    for FILE in "${FASTQ_FILES[@]}"; do
        echo
        echo "Compressing: $(basename "$FILE")"

        if [[ "$COMPRESSOR" == "pigz" ]]; then
            pigz -p "$THREADS" -v "$FILE"
        else
            gzip -v "$FILE"
        fi
    done

    # --------------------------------------------------------
    # 4. Delete original SRA to save disk space
    # --------------------------------------------------------
    status "Removing SRA copy for $ACC"

    rm -rf "$SRA_DIR/$ACC"

    # --------------------------------------------------------
    # 5. Show final files
    # --------------------------------------------------------
    echo "Compressed output:"
    find "$FASTQ_DIR" \
        -maxdepth 1 \
        -type f \
        -name "${ACC}*.fastq.gz" \
        -exec ls -lh {} \;

    status "[$COUNT/$TOTAL] Finished $ACC"

done

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------
status "ALL DOWNLOADS COMPLETE"

echo "FASTQ files are in:"
echo "  $FASTQ_DIR"
echo

echo "Files:"
ls -lh "$FASTQ_DIR"/*.fastq.gz

echo
echo "Total compressed size:"
du -sh "$FASTQ_DIR"