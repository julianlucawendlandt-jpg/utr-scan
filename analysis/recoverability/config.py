"""Study configuration: datasets, accessions, paths, analysis grids.

Everything that names a dataset, a replicate or a result location lives here,
so the numbered scripts never hard-code an accession.  Values that are
MEASURED rather than chosen (P-site offsets, adapters confirmed on raw reads)
are not stored here; they are written to results/ by the QC scripts and read
back, so a measurement can never silently disagree with a constant.
"""

import csv
import datetime
import hashlib
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VERSION = "v1"
RESULTS = os.path.join(ROOT, "results", "recoverability", VERSION)
PANEL_DIR = os.path.join(ROOT, "cache", "recoverability")
FASTQ_DIR = os.path.join(ROOT, "fastq")

SUBDIRS = ("qc", "resolution", "effects", "nulls", "depth", "synthetic",
           "chain", "downstream", "figures")

# ---------------------------------------------------------------------------
# The dataset panel.
#
# `preset` is the key in pipeline/_common.py DATASETS, which owns the adapter,
# genome and fraction -> run mapping used by the alignment step.  `replicates`
# maps the study's replicate label to that preset's fraction name.  Replicates
# are NEVER pooled anywhere in this package; a dataset with one replicate
# simply cannot enter the cross-replicate analyses.
# ---------------------------------------------------------------------------
DATASETS = {
    "gse174329_eif3b": {
        "preset": "human-eif3b-replicated",
        "species": "human",
        "study": "GSE174329",
        "assay": "eIF3B Sel-TCP-seq",
        "selected": True,
        "role": "primary independent-replicate recoverability dataset",
        "replicates": {
            "rep1": {"fraction": "ssu_rep1", "gsm": "GSM5291862", "srr": "SRR14510018"},
            "rep2": {"fraction": "ssu_rep2", "gsm": "GSM5291863", "srr": "SRR14510019"},
        },
    },
    "gse139131_unsel": {
        "preset": "human-40s",
        "species": "human",
        "study": "GSE139131",
        "assay": "unselected 40S TCP-seq",
        "selected": False,
        "role": "assay noise ceiling; unselected vs selected comparison",
        "replicates": {
            "rep1": {"fraction": "ssu", "gsm": "GSM4131410", "srr": "SRR10315320"},
            "rep2": {"fraction": "ssu_rep2", "gsm": "GSM4131411", "srr": "SRR10315321"},
        },
    },
    "gse139131_eif3b": {
        "preset": "human-eif3b",
        "species": "human",
        "study": "GSE139131",
        "assay": "eIF3B Sel-TCP-seq",
        "selected": True,
        "role": "deep scanning-enriched discovery dataset (single replicate)",
        "replicates": {
            "rep1": {"fraction": "ssu", "gsm": "GSM4131412", "srr": "SRR10315322"},
        },
    },
    "archer_yeast": {
        "preset": "archer-yeast",
        "species": "yeast",
        "study": "SRP074093",
        "assay": "TCP-seq SSU",
        "selected": False,
        "role": "cross-species validation",
        "replicates": {
            "rep1": {"fraction": "ssu", "gsm": "SRX1734358", "srr": "SRR3458591"},
        },
        # Auxiliary fractions: never occupancy targets.
        "auxiliary": {
            "rs": {"srx": "SRX1734359", "srr": "SRR3458592"},
            "input": {"srx": "SRX1734360", "srr": "SRR3458593"},
        },
    },
}

# ENA md5 of the archive copy, where the local file IS the ENA file.  Files
# that were produced locally (fasterq-dump + gzip) have a different
# compressed byte stream, so for those the manifest records the local md5 and
# the SRA spot count is the cross-check instead.
ENA_MD5 = {
    "SRR14510018": "c776b156e3bdf078c37515fbbc76d7e5",
    "SRR14510019": "f45de6867b0139e70e7fbfd74b7cda7e",
    "SRR3458591": "3706a2c1439a1059584df86748bdadf4",
    "SRR3458592": "a1133a4e940a0590ca3fb98eb052a95a",
    "SRR3458593": "5101dbb596859aa6a80369b8da752de0",
}

# SRA spot counts (runinfo), the reference for "did we read the whole run".
SRA_SPOTS = {
    "SRR14510018": 17135762, "SRR14510019": 15448720,
    "SRR10315320": 164227985, "SRR10315321": 65084610, "SRR10315322": 180969785,
    "SRR3458591": 51381807, "SRR3458592": 30319437, "SRR3458593": 17036717,
}

ANNOTATION = {
    "human": {"genome": "GRCh38 primary assembly (GENCODE release 49)",
              "gtf": "gencode.v49.annotation.gtf",
              "index": "bowtie2 GRCh38_noalt_as (genome-idx.s3.amazonaws.com)",
              "transcripts": "MANE Select, last 5'UTR exon"},
    "yeast": {"genome": "R64-1-1 (Ensembl 115)",
              "gtf": "Saccharomyces_cerevisiae.R64-1-1.115.gtf",
              "index": "bowtie2 built locally from the same FASTA",
              "transcripts": "start_codon features; leader from Input coverage"},
}

# ---------------------------------------------------------------------------
# Analysis grids
# ---------------------------------------------------------------------------
RESOLUTION_BINS = (1, 3, 5, 10, 20, 50)
MODEL_BINS = (1, 5, 10, 20)
DEPTH_STRATA = ((20, 49), (50, 99), (100, 199), (200, None))
QC_THRESHOLDS = (5, 20, 50, 100, 200)
SEEDS = (0, 1, 2)
N_BOOT = 1000
N_NULL = 100
MIN_LEADER_LEN = 20
CDS_EXTENSION = 30          # nt past the AUG stored for sensitivity analyses
FOLD_WINDOW = 40            # local-dG window, never crossing an intron
N_FOLDS = {"human": 5, "yeast": 4}

# Columns every result table carries, in this order, before any extras.
STANDARD_COLUMNS = ("dataset", "replicate", "split", "seed", "bin_width",
                    "model", "n_transcripts", "n_reads", "metric", "value")


# ---------------------------------------------------------------------------
# Hardware.  Detected, never assumed, so one code base runs on a laptop and on
# a GPU server.  Environment overrides: UTR_BACKEND, UTR_WORKERS, UTR_THREADS.
# ---------------------------------------------------------------------------

def cuda_available():
    """True when a CUDA GPU is usable.  Checked with nvidia-smi first so the
    parent process never initialises CUDA before forking workers."""
    if os.environ.get("UTR_BACKEND", "auto") == "scipy":
        return False
    import shutil
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        import torch
        # is_available() also catches a driver too old for the CUDA build.
        # Worker pools use spawn, so querying CUDA in the parent is safe.
        return bool(torch.version.cuda) and torch.cuda.is_available()
    except Exception:
        return False


def backend():
    """'torch' (GPU) or 'scipy' (CPU) for likelihood.fit."""
    choice = os.environ.get("UTR_BACKEND", "auto").lower()
    if choice in ("scipy", "torch"):
        return choice
    return "torch" if cuda_available() else "scipy"


def gpu_memory_gb():
    """Total memory of the first GPU from nvidia-smi (no CUDA initialisation)."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return float(out.strip().splitlines()[0]) / 1024
    except Exception:
        return 0.0


def resources():
    """{'cpus', 'ram_gb', 'gpu'} for sizing workers and threads."""
    cpus = os.cpu_count() or 1
    ram = None
    try:
        import psutil
        ram = psutil.virtual_memory().total / 2 ** 30
    except Exception:
        try:
            with open("/proc/meminfo") as fh:
                ram = int(fh.readline().split()[1]) / 2 ** 20
        except Exception:
            ram = 8.0
    gpu = cuda_available()
    return {"cpus": cpus, "ram_gb": float(ram), "gpu": gpu,
            "gpu_mem_gb": gpu_memory_gb() if gpu else 0.0}


def default_workers(gb_per_worker=1.6, gpu_gb_per_worker=2.0):
    """Parallel analysis workers: one per core, capped by RAM (each human
    worker holds the panel, ~1-1.5 GB) and, when the GPU backend is in use,
    by GPU memory (each worker opens its own CUDA context plus chunk
    buffers, ~1.5-2 GB).  UTR_WORKERS overrides."""
    if os.environ.get("UTR_WORKERS"):
        return max(1, int(os.environ["UTR_WORKERS"]))
    r = resources()
    n = min(r["cpus"], int(r["ram_gb"] // gb_per_worker) - 1)
    if r["gpu"] and backend() == "torch" and r["gpu_mem_gb"] > 0:
        n = min(n, int(r["gpu_mem_gb"] // gpu_gb_per_worker))
    return max(1, n)


def results_path(*parts):
    p = os.path.join(RESULTS, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def ensure_layout():
    for d in SUBDIRS:
        os.makedirs(os.path.join(RESULTS, d), exist_ok=True)
    os.makedirs(PANEL_DIR, exist_ok=True)


def write_tsv(rows, path, extra_first=()):
    """Write dict rows with the standard columns first, then any others.

    Missing standard fields are written empty rather than dropped, so every
    table can be concatenated with every other.
    """
    rows = list(rows)
    keys = list(extra_first) + [c for c in STANDARD_COLUMNS if c not in extra_first]
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in keys})
    return path


def read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _fmt(v):
    if isinstance(v, float):
        return "{:.6g}".format(v)
    return "" if v is None else v


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest: one JSON, updated section by section by whichever script owns the
# section.  Each update stamps the commit it was produced from.
# ---------------------------------------------------------------------------
MANIFEST = os.path.join(RESULTS, "manifest.json")


def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as fh:
            return json.load(fh)
    return {}


class file_lock(object):
    """Exclusive lock file next to `path`, for read-modify-write updates by
    concurrent processes.  A lock older than 10 minutes is treated as stale."""

    def __init__(self, path):
        self.lock = path + ".lock"

    def __enter__(self):
        import time
        os.makedirs(os.path.dirname(self.lock) or ".", exist_ok=True)
        for _ in range(36000):
            try:
                os.close(os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.lock) > 600:
                        os.remove(self.lock)
                        continue
                except OSError:
                    pass
                time.sleep(0.1)
        raise TimeoutError("could not acquire " + self.lock)

    def __exit__(self, *exc):
        try:
            os.remove(self.lock)
        except OSError:
            pass


def merge_tsv(path, rows, keep_old):
    """Replace this run's rows in a shared table, atomically w.r.t. other
    processes: under the lock, keep existing rows for which keep_old(row) is
    True, append `rows`, write.  Concurrent steps (e.g. the four effect grids)
    can then share one table without losing each other's results."""
    with file_lock(path):
        old = read_tsv(path) if os.path.exists(path) else []
        write_tsv([r for r in old if keep_old(r)] + list(rows), path)
    return path


def update_manifest(section, value, merge=False):
    """Read-modify-write one section under a lock file.

    Several scripts run concurrently (alignment QC, panels, effect fits);
    without the lock, two of them updating different sections at once would
    silently drop one.  A lock older than 10 minutes is treated as stale.
    """
    import time
    os.makedirs(RESULTS, exist_ok=True)
    lock = MANIFEST + ".lock"
    for _ in range(6000):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > 600:
                    os.remove(lock)
                    continue
            except OSError:
                pass
            time.sleep(0.1)
    try:
        m = load_manifest()
        m.setdefault("version", VERSION)
        if merge and isinstance(m.get(section), dict) and isinstance(value, dict):
            m[section].update(value)      # nested update done inside the lock
        else:
            m[section] = value
        m.setdefault("history", []).append({
            "section": section, "git_commit": git_commit(),
            "time": datetime.datetime.now().isoformat(timespec="seconds")})
        tmp = MANIFEST + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(m, fh, indent=2, sort_keys=False, default=str)
        os.replace(tmp, MANIFEST)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass
    return m


def measured(name, default=None):
    """A value written by a QC script, e.g. measured('psite_offsets')."""
    return load_manifest().get(name, default)
