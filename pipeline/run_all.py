"""Run the whole FASTQ -> cached-examples pipeline.

    python pipeline/run_all.py [--max-reads 2000000] [--offset 0]

Each step checks for its own output first, so re-running after a failure
picks up where it left off instead of redoing the expensive parts.  Start
with --max-reads to confirm the wiring before committing to the full 23 GB.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

STEPS = ["01_trim.py", "02_align_count.py", "03_build_examples.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq-dir")
    ap.add_argument("--max-reads", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--min-mapq", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--utr-length", type=int, default=50)
    ap.add_argument("--utr-table")
    ap.add_argument("--min-ssu-counts", type=int, default=5)
    ap.add_argument("--cds-length", type=int, default=100)
    ap.add_argument("--context-flank", type=int, default=10)
    ap.add_argument("--cds-states", type=int, default=0)
    args = ap.parse_args()

    per_step = {
        "01_trim.py": (["--fastq-dir", args.fastq_dir] if args.fastq_dir else [])
                      + (["--max-reads", str(args.max_reads)] if args.max_reads else []),
        "02_align_count.py": ["--offset", str(args.offset),
                              "--min-mapq", str(args.min_mapq),
                              "--threads", str(args.threads)],
        "03_build_examples.py": ["--utr-length", str(args.utr_length),
                                 "--min-ssu-counts", str(args.min_ssu_counts),
                                 "--cds-length", str(args.cds_length),
                                 "--context-flank", str(args.context_flank),
                                 "--cds-states", str(args.cds_states)]
                                + (["--utr-table", args.utr_table] if args.utr_table else []),
    }

    for step in STEPS:
        cmd = [sys.executable, os.path.join(HERE, step)] + per_step[step]
        print("\n=== {} ===".format(step))
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            sys.exit("\n{} failed (exit {}).  Fix it and re-run run_all.py; "
                     "completed steps will be skipped.\n".format(step, r.returncode))

    print("\nPipeline done.  Train on it with:\n  python train.py --data archer\n")


if __name__ == "__main__":
    main()
