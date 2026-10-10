# utr-scan

A sequence-conditioned absorbing Markov chain for 40S scanning of yeast
5′UTRs, fit to Archer TCP-seq.

A single 40S particle loads at the cap and takes discrete steps along the UTR.
Every nucleotide is one transient state; the particle either scans on, commits
at an AUG, pauses there, drops off, or reads straight through the start codon. A small shared network turns a window of local
sequence and structure into per-position step, commit and dwell tendencies, and the
chain is solved in closed form with the fundamental matrix
`N = (I − Q)⁻¹`. The expected-visits row from the cap state *is* the predicted
occupancy profile, and it is fit to observed SSU footprint 5′ ends with a
Poisson likelihood.

Research prototype. No evaluation, metrics or figures — correct logic, working
data connectors, and a training loop that runs.

## Why the chain stays memoryless

Context enters through the **rates**, not the state. `f_θ` sees ±10 nt around
each position, so it can read a 3-nt codon and its Kozak context, and it bakes
that into `w_j`, `p_j` and `d_j` once, up front. The walk itself never looks back.
That is what keeps `Q` a plain matrix and the solve exact.

`start_mask` is a **separate boolean track**, never a substitution into the
sequence: the real nucleotides stay intact and the rate net still sees them.
The mask only tells the chain where a commit transition, and a pause, are
allowed to open.

## Why there is a pause

A walker that only moves forward visits each position at most once, so its
expected-visits profile can only *fall* from cap to AUG. Real SSU footprints
do the opposite: they pile up at the start codon (in the Archer data, a sharp
peak 12 nt upstream of the AUG in raw 5′-end coordinates — the classic P-site
offset). A self-loop `Q[j,j] > 0` at start positions is the minimal thing that
lets the chain draw that peak: expected visits at *j* become
`P(reach j) / (1 − P(stay at j))`. Dwell changes occupancy but leaves the
fates untouched, since staying only defers the choice among the other moves.
Both facts are checked numerically in the sanity asserts below.

## Why the rate net gets flanking sequence

The chain's states are the UTR window. The rate net's *input* is that window
plus `--context-flank` nt of **real genomic sequence** on each side, and its
output is cropped back to the states.

This is not cosmetic. The window ends 2 nt after the main AUG, so
`main_start_index` is the same in every example, and a zero-padded
convolution can read "how far am I from the padding". Trained that way, the
net learned *"3 slots from the zeros = the real start codon"* and never looked
at a codon: the same AUG with the same neighbours got commit logit 1.94 at the
window edge and 6.94 moved inland, with the dwell logit flipping from +6.2 to
−8.5. With real flanks the same comparison gives **0.000e+00** — a convolution
cannot see array indices, only the values in its receptive field, so once
those are all real DNA there is no positional cue left to find.

`model/scanner.py` asserts the flanks are at least as wide as the net's
window; widening `--window` without widening `--context-flank` is refused
rather than silently reopening the hole.

## Why read_through is separate from drop_off

Both mean "no protein from this gene", but they are different events. Nothing
downstream of the main AUG is ever a start codon — the window ends 2 nt later
— so a walker can only get there by leaking past the AUG, and every exit from
downstream is a read-through. Folding those into `drop_off` (as an earlier
version did) hid leaky scanning inside the drop-off number. The split is
exact, not an approximation.

## Why there is a second likelihood term

Occupancy records where particles *are*. Committing and dropping off both
remove a particle from the chain, so the SSU profile alone cannot tell them
apart — fit to occupancy only, the commit/drop split at an AUG is a free
parameter, and on the Archer data gradient descent drove commit to zero and
called everything drop-off.

The 80S fraction (SRX1734359) sees the *consequence* of committing: a 40S
that commits becomes an 80S that elongates through the coding region; one
that drops off leaves nothing. So each transcript carries a second
observation — 80S reads over the first 100 nt of its CDS — and the loss asks
that `rho_load × P(commit_main) × (Input reads in that window) ≈ 80S reads`.
Input divides out mRNA abundance, and `rho_load`, the single global scalar,
converts "probability of committing per scan" into "80S per mRNA copy".

That term is what pins the commit/drop split. On synthetic data with planted
`rho_load = 2.0` the model recovers **2.01** and the mean fates to within
0.005 of truth; with `--commit-weight 0` it does neither. The CDS window is
not part of the chain's state space — it is a gauge, not a model, and its
length (`--cds-length`) only sets how much the readout is averaged.

## Layout

```
model/chain.py        the absorbing chain: build Q,R -> solve -> (profile, fates)
model/rate_net.py     f_theta: windowed sequence+structure -> (w_j, p_j, d_j)
model/scanner.py      glue: features -> rate net -> chain
data/synthetic.py     toy UTRs with planted uORFs and hairpins
data/features.py      one-hot, start masks, ViennaRNA structure tracks
data/coverage.py      5'-end assignment + projection onto UTRs (pure functions)
data/archer.py        loads the cached pipeline output
pipeline/             idempotent FASTQ -> counts -> examples (subprocess orchestration)
scripts/download_reference.py
scripts/call_utr_boundaries.py   per-gene 5'UTR lengths from Input coverage
train.py
```

`model/chain.py` is the piece to read first; it is written to stand alone for
anyone who knows the absorbing-chain math.

## Quick start — synthetic

No external data, no binaries, nothing to download:

```bash
pip install -r requirements.txt
python train.py
```

`data/synthetic.py` plants strong- and weak-Kozak uORFs and GC-rich hairpin
stems in toy UTRs, then generates occupancy by running **the same forward
chain** with hand-set ground-truth rates plus Poisson noise. So the target is
genuinely inside the model's hypothesis class, which is what makes this a test
of the learning path rather than of the model's expressiveness.

It converges to the oracle on occupancy, and recovers the planted commitment
scale and fates through the 80S term:

```
[train] grad check through the solve: ok
epoch   0  loss/nt  -55.8391  commit/gene  -9.64  rho_load 1.047  commit_main 0.031  commit_uORF 0.107  drop_off 0.863
epoch  10  loss/nt  -57.9216  commit/gene -20.57  rho_load 1.436  commit_main 0.087  commit_uORF 0.237  drop_off 0.676
epoch  39  loss/nt  -57.9557  commit/gene -21.07  rho_load 2.012  commit_main 0.076  commit_uORF 0.232  drop_off 0.692
                                                  true:     2.0               0.073              0.229           0.698
```

Useful flags: `--commit-weight` (weight of the 80S term; 0 = occupancy only),
`--window` (rate-net context; synthetic flanks follow it automatically),
`--structure` (fold with ViennaRNA), `--near-cognate` (let near-cognate codons
open commit edges), `--normalize-input` (divide the target by the abundance
track), `--epochs`, `--lr`.

## Archer TCP-seq

```bash
python scripts/download_reference.py     # genome + annotation + bowtie2 index
python pipeline/run_all.py --max-reads 2000000   # sanity run on a subsample
python pipeline/run_all.py                       # the real thing
python train.py --data archer
```

Every step checks for its output before recomputing, so a re-run is cheap and
a failed run resumes where it stopped. Start with `--max-reads`; the SSU FASTQ
is 23 GB.

`--offset 12` on the alignment step shifts the footprint 5′-end assignment to
the SSU P site, so the occupancy peak lands on the AUG state where the pause
and commit transitions live. Default is 0, the raw 5′ end. Changing the
offset invalidates the cached counts; the trimmed FASTQs are reused.

Three fractions go through the pipeline: SSU (occupancy target), RS / 80S
(commitment readout) and Input (abundance). `--cds-length` sets the coding
window the 80S and Input reads are summed over (default 100 nt, exon-aware),
and `--context-flank` the real sequence given to the rate net (default 10 nt,
must be ≥ `--window`).

### Data-driven 5′UTR boundaries

The default fixed 50 nt window is a guess. To call each gene's UTR from the
data instead:

```bash
python scripts/call_utr_boundaries.py            # writes cache/utr_lengths.tsv
python pipeline/03_build_examples.py --utr-table cache/utr_lengths.tsv
```

Step 2 writes a full read-coverage track for the Input fraction alongside the
5′-end counts (same single pass over the SAM stream). The Input fraction is
total RNA, so its coverage stops where the transcript does; walking upstream
from the start codon until a sustained gap gives the boundary, with a hard
clip at the nearest same-strand upstream CDS so a window can never reach into
a neighbouring gene. Genes with no coverage at their own AUG get no call and
fall back to the fixed window.

See **DATA.md** for the FASTQ paths, the adapter, the aligner and multimapper
policy, the 5′UTR annotation decision, and the transcript counts.

### Required binaries

Python packages come from `requirements.txt`. Two external binaries are needed
for the Archer path only — the synthetic path needs neither:

| Binary | Needed by | Install |
|---|---|---|
| `cutadapt` | `pipeline/01_trim.py` | `pip install cutadapt` (pure Python, works on Windows) |
| `bowtie2-align-s`, `bowtie2-build-s` | `pipeline/02_align_count.py`, `scripts/download_reference.py` | **Windows:** unzip `bowtie2-<ver>-mingw-x86_64.zip` from [bowtie2 releases](https://github.com/BenLangmead/bowtie2/releases) anywhere under `utr-scan/bin/` — bioconda has no Windows packages. **Linux/macOS/WSL:** `conda install -c bioconda bowtie2` |

Anything under `bin/` is found before `PATH`. The pipeline calls the bare
`-s` executables directly rather than the `bowtie2` Perl wrapper, which on
Windows resolves to a `.bat` that mangles the index path; the wrapper's only
jobs (large-index dispatch, gz decompression) are not needed here.

`samtools` is **not** required — see DATA.md §5.

ViennaRNA (`import RNA`) is optional. Without it the structure tracks degrade
to zeros with a warning rather than hard-failing.

Each script exits with an actionable message naming the missing binary and its
install command, so a missing dependency is never a traceback.

## Sanity checks

Inline asserts, not a test suite. They run as part of normal execution:

- transient rows of `[Q | R]` sum to 1, and no self-loop has probability 1
  (`model/chain.py:build_QR`)
- the rate net's flanks are at least as wide as its window, so no output
  position ever sees padding (`model/scanner.py`)
- fates sum to 1, profile non-negative (`model/chain.py:solve_chain`)
- gradients through `torch.linalg.solve` are finite and nonzero (`train.py`,
  checked once per run)
- CIGAR arithmetic, 1-based→0-based conversion, strand-dependent offset sign,
  UTR interval geometry and exon-aware CDS windows (`data/coverage.py`, at
  import time)
- `context_seq[context_offset:][:N] == seq` (`data/features.py`)

## Numerics

The chain builds `Q`, `R` and solves in **float64** regardless of the model's
dtype. The quantity the solve depends on is `1 − P(stay)`, and a 40S dwelling
hard at a start codon drives `P(stay)` towards 1; at 0.9999 float32 holds only
about three correct digits of the difference, the fundamental matrix comes out
~0.1% wrong, and the fates stop summing to 1 — which `train.py` was silently
catching and skipping. The cancellation is in `build_QR`, not the solve. These
matrices are one per transcript and a few hundred wide, so double precision
costs nothing.

## Status

Full Archer run done: 51.4 M SSU / 30.3 M RS / 17.0 M Input reads →
792 k / 463 k / 975 k uniquely-mapped 5′ ends → 2,851 transcripts. The SSU
occupancy peak lands exactly on the annotated AUG at `--offset 12`, which is
the end-to-end check on the coordinate machinery.

Known open issue: with a fixed 50 nt window only 11% of genes have a uORF in
view, and a direct test — does leaky scanning past the main AUG predict lower
80S output? — comes back flat (ρ ≈ 0 at n = 542, enough power to detect
ρ ≈ 0.12). The `--utr-table` path above is the intended fix; the fates are
internally consistent but not yet externally validated.
