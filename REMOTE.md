# Running the full pipeline on a Linux server

From a fresh machine to the final tables and figures in four commands.
Uses an NVIDIA GPU automatically when one is present; runs on CPU otherwise.

## What the server needs

- **OS:** Linux x86_64. Ubuntu 22.04/24.04 or any distro with `bash`, `curl`,
  `git` and `tar`. No root needed.
- **Disk:** at least **80 GB** free in the directory you clone into. Raw reads
  take ~22 GB, references ~12 GB, and intermediates and results the rest.
- **RAM:** 16 GB minimum; 32 GB+ lets steps run in parallel.
- **CPU:** more cores make alignment and the parallel analysis steps faster.
  16–32 is a good match for a single GPU.
- **GPU (optional):** any NVIDIA card with a driver supporting CUDA 12.x, for
  example an RTX 6000 Ada. Check with `nvidia-smi`.

## Steps

**a. Get the code onto the server**

```bash
git clone <your repo URL> utr-scan      # or: rsync -a utr-scan/ server:utr-scan/
cd utr-scan
```

**b. One-time setup** of the environment and tools, plus self-tests (about
10–20 minutes). This step downloads no data.

```bash
bash scripts/remote/bootstrap.sh
```

It does all of this, and is safe to re-run if interrupted:

- installs Miniforge (conda) if there is none, and creates the `utrscan`
  environment from `environment.yml`, which includes bowtie2, cutadapt,
  ViennaRNA, sra-tools, GNU parallel and pigz;
- installs PyTorch, the CUDA build if a GPU is visible and the CPU build if not;
- checks every tool;
- runs the unit tests, including the GPU-vs-CPU agreement test when a GPU is
  present.

**c. Download the data.** This fetches the 8 FASTQs from SRA and all
references: yeast and human genomes, annotation (human GTF filtered to MANE
Select, as in v1), bowtie2 indexes and rRNA sequences. Anything already
present is skipped, so re-run it after an interruption.

```bash
conda activate utrscan
bash download_srrs.sh
```

On a machine with fewer than 16 cores, set `THREADS=2` near the top of the
script.

**d. Run everything**

```bash
bash scripts/remote/run_pipeline.sh
```

This runs alignment, QC, panels, effect models, nulls, synthetic power, depth
curves, chain ablation, summary, figures and the MPRA step, in that order. It
is detached, so you can log out.

```bash
bash scripts/remote/run_pipeline.sh status    # where it is
tail -f results/recoverability/v1/logs/run_all.log
bash scripts/remote/run_pipeline.sh stop      # stop; run again to resume
```

If a step fails, the progress log names it and shows the last lines of its log
(`results/recoverability/v1/logs/<step>.log`). Fix the cause and start again.
Finished steps are skipped.

## Results

```
results/recoverability/v1/
  summary/recoverability_summary.tsv   one verdict per effect x resolution
  figures/                             fig1 ... fig6 (PNG + SVG)
  manifest.json                        commit, accessions, checksums, P-site rules, seeds
  qc/ resolution/ effects/ nulls/ synthetic/ depth/ chain/ downstream/
```

Copy them back with, for example,
`rsync -a server:utr-scan/results/ results/`.

## Useful options

| | |
|---|---|
| `python -m analysis.recoverability.run_all --list` | the plan, and which steps are done |
| `bash scripts/remote/run_pipeline.sh -- --from nulls_P_b1` | resume from a step |
| `bash scripts/remote/run_pipeline.sh -- --only figures,summary --force` | redo selected steps |
| `UTR_BACKEND=scipy` | force CPU fitting even with a GPU |
| `UTR_WORKERS=8` | override the automatic worker count |
| `UTR_CHUNK_MB=1024` | larger GPU chunks (more GPU memory, slightly faster) |

## What uses the GPU

| Step | Device |
|---|---|
| Effect models, nulls, synthetic power, depth curves (`likelihood.fit`) | GPU: PyTorch backend, float64 objective, float32 information matrix |
| Chain ablation (RateNet + chain) | GPU |
| Alignment (bowtie2), RNA folding (ViennaRNA), panel building | CPU, all cores |

Both fitting backends run the same algorithm. On every run, bootstrap's
`test_backends` confirms they agree to below 10⁻⁶ log-likelihood per read on
the machine you are using.
