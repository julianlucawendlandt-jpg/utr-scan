# Recoverability analysis

Which aspects of 5′UTR scanning are statistically recoverable from 40S
footprinting, at what spatial resolution, and at what depth. The study
design is in the project brief; this file covers how to run it and every
decision the code makes that the brief left open.

Results go to `results/recoverability/v1/`, never only into comments.
`manifest.json` there records the commit, accessions, checksums, P-site
rules, structure method, thresholds and seeds of every section.

## Running

```bash
# 0. data and references (once)
python scripts/download_reference.py                       # yeast
#    human: GRCh38 bowtie2 index (genome-idx GRCh38_noalt_as) + GENCODE v49
#    FASTA/GTF into ref/GRCh38.*; GTF filtered to MANE_Select records
python pipeline/02_align_count.py --dataset human-eif3b-replicated --stream --threads 7
python pipeline/02_align_count.py --dataset archer-yeast --stream --threads 7
python pipeline/02_align_count.py --dataset human-40s --stream --threads 7
python pipeline/02_align_count.py --dataset human-eif3b --stream --threads 7

# 1. everything else, in the study's order (resumable with --from)
python -m analysis.recoverability.run_all
```

Unit tests (plain asserts):

```bash
python -m analysis.recoverability.tests.test_likelihood   # multinomial model
python -m analysis.recoverability.tests.test_nulls        # matched permutations
python -m analysis.recoverability.tests.test_chainfast    # closed-form chain
```

## Layout

| file | step | what |
|---|---|---|
| `config.py` | – | datasets, accessions, grids, manifest/TSV helpers |
| `00_inspect_reads.py` | 1 | 100k raw reads per run: adapters, inserts, rRNA, unique rate |
| `psite.py`, `01_qc.py --stage offsets` | 1 | read-length-resolved P-site rule from start-codon metagenes |
| `panel.py` | 2 | annotation layer per species + counts layer per dataset |
| `01_qc.py --stage tables` | 3 | sparsity / depth / region table |
| `02_replicate_resolution.py` | 4 | replicate agreement vs bin width, by depth |
| `splits.py` | 5 | frozen depth-balanced chromosome folds |
| `likelihood.py`, `features.py` | 6 | conditional multinomial, Fisher scoring, feature blocks |
| `03_nested_effects.py` | 6–7 | nested models, chromosome holdout, 4 transfer directions |
| `04_shuffle_nulls.py` | 8 | position-matched permutation nulls |
| `06_synthetic_power.py` | 9 | planted truths, power vs depth and effect size |
| `05_depth_curve.py` | 10 | binomial thinning; matched-depth enriched vs unselected |
| `06b_identifiability.py` | 11 | Fisher information of chain parameters |
| `chainfast.py`, `07_chain_ablation.py` | 12 | position vs constrained chain vs RateNet+chain |
| `08_downstream_utility.py` | 13 | MPRA relevance (after the above are frozen) |
| `summary.py` | 14 | one status per effect × resolution |
| `make_figures.py` | – | figures 1–6 |

## Decisions the brief left open

**Data**
- GSE174329 runs were resolved through SRA runinfo, not by order:
  GSM5291862 = SRR14510018, GSM5291863 = SRR14510019. Both match their ENA
  MD5s. The deposited reads are already adapter-trimmed (no adapter in 100k
  reads); the preset applies quality and length filtering only.
- The GSE139131 mapping was verified the same way: SRR10315320 = GSM4131410,
  SRR10315321 = GSM4131411, SRR10315322 = GSM4131412.
- All human datasets use min length 25, so selected and unselected libraries
  are filtered identically.
- Human alignment uses the prebuilt GRCh38 no-alt analysis-set bowtie2 index.
  Building one locally needs hours and more than the 8 GB RAM this machine
  has. Its coordinates are identical to the GENCODE primary assembly on
  chr1–22/X/Y/M.

**P-site assignment.** A single 5′-end offset does not fit selected 40S. The
start-codon metagene of raw 5′ ends in GSE174329 is a ~20 nt plateau, while
the footprint 3′ end sits at +24 nt in every length bin. `psite.py` therefore
picks, per footprint-length bin, whichever end gives the sharper metagene. It
keeps that rule only if it beats the best single 5′ offset on peak sharpness
at the AUG. The rule is measured on rep1 and applied to both replicates, so
they share identical processing. Per-read 5′ positions and spans are stored
(`work/*.reads.npz`), so any other rule can be applied without re-aligning.

**Leaders and structure**
- Human leaders are the last 5′UTR exon of the MANE Select transcript, plus
  the AUG and 30 nt of exon-aware CDS.
- Yeast leaders are called from Input coverage. Genes without a call are
  excluded, not given a fixed window.
- Structure is folded on exactly that stored string, with no genomic flank.
  Local-ΔG windows are clipped to it, so no window can contain intron.

**Model**
- Positional baseline: bins of distance to AUG and to the 5′ end, at the
  same width as the evaluation resolution, capped at 300 nt.
- Bins are anchored so the main-AUG bin covers the codon. At 5 nt it is
  exactly the start nucleotide ±2.
- Gaussian prior sd 1 on every coefficient.
- Start features occupy the start nucleotide ±2 and are silent elsewhere.
- Structure features are pairing and ΔG at the P site and ahead of the 40S
  (+5..+20 nt).

**Evaluation**
- Every transcript is tested once, in its held-out fold.
- The test set requires ≥10 leader reads in every replicate, so all four
  transfer directions use the same transcripts. The training set requires
  ≥10 reads in the training replicate.
- Bootstraps resample transcripts.

**Added beyond the brief**
- `pooled_posadj_r` in Step 4. Raw replicate correlations are inflated by
  the shared AUG peak and the cap-to-AUG ramp, so this metric subtracts the
  dataset-wide positional profile first.

**Chain models.** B and C run without reinitiation. That makes the expected
visits closed-form (`chainfast.py`, checked against `model/chain.py`), so all
transcripts train in one batch. Leaving at an AUG counts as committing,
because commit vs. drop is unidentifiable from occupancy (Step 11). B and C
also get the same small 5′-proximal observation term the position baseline
has.

**Overdispersion.** The closed-form Dirichlet-multinomial estimator is biased
at ~1 read per position, so α is calibrated by simulation to reproduce the
real replicate dispersion ratio (`data/synthetic.py`).
