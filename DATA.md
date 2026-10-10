# DATA.md — pipeline decisions

Every choice the preprocessing pipeline makes, and why. Written as the
decisions were made, not reconstructed afterwards.

## 1. Input FASTQs

Located on disk (not assumed — the paths below were confirmed by inspection):

| Fraction | SRX | Run / file | Size |
|---|---|---|---|
| **SSU** (40S footprints) — occupancy target | SRX1734358 | `../fastq/SRR3458591.fastq` | 23.2 GB |
| **RS** (80S footprints) — commitment readout | SRX1734359 | `../fastq/SRR3458592.fastq` | 13.7 GB |
| **Input** (total RNA) — abundance control | SRX1734360 | `../fastq/SRR3458593.fastq` | 7.7 GB |

The RS fraction was added after the first training runs showed that the SSU
profile alone cannot separate commitment from drop-off (both remove a walker
from the chain). 80S footprints in the coding region are the downstream
consequence of commitment and break that degeneracy; see README.

`pipeline/_common.py:FRACTIONS` holds this mapping; `find_fastqs()` resolves
the paths at run time and also accepts `.fastq.gz` and `_1.fastq`.

**Single-end, 151 nt**, Illumina HiSeq 2500 (instrument `D9S4KXP1`, run 312,
flowcell `HFK2FADXX`). `fasterq-dump` produced one file per run with no `_1`/`_2`
split, and every read header carries `length=151`.

**The SRX→SRR assignment was checked, not assumed.** Two independent lines of
evidence agree:

- Accession ordering: SRX1734358 < SRX1734360 matches SRR3458591 < SRR3458593.
- Insert-length distribution over 100k sampled reads. SRR3458591 is sharply
  peaked at **30–35 nt** — a footprint-length mode, which is what nuclease
  protection produces. SRR3458593 is flat across **17–35 nt** — random
  fragmentation, which is what a total-RNA input looks like.

## 2. Adapter and read structure

Read off the raw reads directly:

```
<footprint> AAAAAAAAAAAAAAAAAAAA TGGAATTCTCGGGTGCCAAGG AACTCCAGTCAC<index>ATCTCGTATGCCGTCTTCTGCTTG
|__insert__| |____poly(A)______| |__TruSeq Small RNA RA3__|
```

The 3′ adapter is the **Illumina TruSeq Small RNA RA3, `TGGAATTCTCGGGTGCCAAGG`**,
and it is preceded by a poly(A) stretch: the Archer TCP-seq protocol
poly(A)-tails footprints before ligating the 3′ adapter. So there are two
things to strip, in that order.

Prevalence in 100k sampled reads: full 21-nt adapter match 13%, adapter 13-mer
38–41%, a run of ≥10 A's **93–95%**. The low full-adapter number is a
sequencing-quality artefact — base quality collapses right after the
homopolymer — so **poly(A) is the more reliable trim signal**, and the trimmer
is configured to catch either.

About 5% of SRR3458591 reads begin with poly(A) at position 0 (empty insert);
the minimum-length filter removes them.

**Caveat, recorded deliberately:** trimming poly(A) also eats genomic A's at a
footprint's 3′ end, so trimmed lengths slightly underestimate true footprint
length. This does not propagate into the model, because trimming only touches
the 3′ end and every downstream quantity is anchored on the **5′ end**.

## 3. Trimmer

**cutadapt.** Chosen over fastp/trim_galore because it is pure Python, so it
installs with `pip` and runs on Windows without conda or WSL — the only tool in
the chain with that property. It also handles the two-stage trim natively:

```
cutadapt -a TGGAATTCTCGGGTGCCAAGG -a "A{20}" --times 2 -m 18 -q 20 --max-n 0
```

`--times 2` makes a second pass so the poly(A) uncovered by removing the
adapter also goes. `-m 18` drops reads too short to place uniquely in a 12 Mb
genome. `-q 20` trims the 3′ quality collapse noted above.

## 4. Reference

**Ensembl release 115, R64-1-1**, genome and annotation both:

- `Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz` → `ref/sacCer3.fa`
- `Saccharomyces_cerevisiae.R64-1-1.115.gtf.gz` → `ref/sacCer3.gtf`

sacCer3 and R64-1-1 are the same assembly. Taking both files from one source
means sequence names agree (`I`, `II`, … `Mito`) with **no chrI↔I translation
layer to get wrong** — using UCSC for the genome and Ensembl for the annotation
would require exactly that. `scripts/download_reference.py` is idempotent and
exits with the two URLs and drop-in instructions if the host is unreachable.

## 5. Aligner and multimapper policy

**bowtie2.** Justification: yeast 5′UTRs are essentially intronless and the
reads are short trimmed footprints, so nothing here needs a splice-aware
aligner. STAR's genome-generate step is heavyweight for a 12 Mb genome and it
has no native Windows build. Settings:

```
bowtie2 -x ref/sacCer3 -U <trimmed> --no-unal -L 18 -N 0
```

`-L 18` shortens the seed from the default 22, which matters when a fair share
of inserts are under 25 nt.

**Multimapper policy: uniquely-mapped reads only, MAPQ ≥ 10** (bowtie2's
conventional unique threshold — a read with a near-tied second-best alignment
scores below it). Secondary and supplementary alignments are dropped by FLAG.
Multireads are **discarded, not distributed**: splitting one read's weight
across several positions would invent occupancy that no single 40S ever had.
`data/coverage.py:parse_sam_line` implements this.

**The SSU fraction is ~93% rRNA, and the unique-mapping filter is what
removes it.** Measured on 50k trimmed SSU reads with the Windows bowtie2
build: 86% align, but 84% align to *more than one place*, and 92.6% of all
aligned reads land in the tandem rDNA repeat locus (chr XII:450–470 kb).
That is expected biology, not a defect: TCP-seq has no rRNA depletion step
and the 40S subunit is literally built from 18S rRNA. Yeast carries ~150
tandem rDNA copies, so every rRNA fragment is a multimapper with MAPQ ≈ 0–1,
and the MAPQ ≥ 10 threshold discards it. **Only ~2–3% of SSU reads are
uniquely-mapped mRNA footprints** — roughly 1 M usable 5′ ends from the full
51 M-read run, and ~180 k from a 10 M-read subsample. On a subsample expect
many transcripts to fail `--min-ssu-counts`; lower it or run the full data.

**samtools is not required.** bowtie2 writes SAM to stdout and
`pipeline/02_align_count.py` streams it straight into the 5′-end counter, so a
23 GB FASTQ never becomes a 40 GB SAM on disk. Nothing downstream wants the
alignments themselves — only where each 5′ end landed — so a sorted, indexed
BAM would be pure overhead. This is why alignment and counting are one step.

## 6. Footprint 5′-end assignment

The load-bearing step, done explicitly in `data/coverage.py:five_prime_position`
rather than with a generic coverage tool — coverage smears each read across its
whole length, which is exactly the wrong thing. A footprint tells us where one
40S *was*, and the 5′ end is the anchor for that.

Strand-aware, 0-based:

- **+ strand:** leftmost aligned base = SAM `POS`.
- **− strand:** rightmost aligned base = `POS + reference_span(CIGAR) − 1`,
  where the span counts only `M/D/N/=/X`.

Soft-clipped bases are **not** added back — `POS` already excludes them, and
clipped sequence is adapter or untrimmed poly(A), not footprint.

**Offset parameter — default 0 = raw 5′ end.** `--offset N` on
`pipeline/02_align_count.py` shifts the assignment *downstream in the direction
of scanning*, so the sign flips on the minus strand. Passing `--offset 12`
moves from the footprint 5′ end to the SSU P site. (The same shift is often
written as "−12" under the opposite bookkeeping convention — subtracting from
the P site to recover the 5′ end. Same operation.) Changing the assignment is
this one flag and nothing else.

## 7. 5′UTR definition

**Default: fixed 50 nt window** upstream of the annotated start codon
(`--utr-length`, configurable), extended 3 nt so the AUG itself is inside the
window. `main_start_index` is therefore always `len(utr_seq) − 3`, and the
occupancy vector covers the position the model cares about most.

This is the documented fallback, and it is what runs out of the box. The
TSS-based annotation (Pelechano 2013 TIF-seq, Nagalakshmi 2008) is the better
choice scientifically, but **neither is scriptably obtainable**, which was
checked before giving up on it:

- UCSC sacCer3's `sgdIsoforms` table — the obvious candidate — turns out to
  hold only a cluster-ID↔transcript map, **no coordinates**.
- The Nature supplementary files on `static-content.springer.com` return
  **HTTP 403** to anything that is not a browser.
- `sgd-archive.yeastgenome.org` serves a JavaScript S3 listing; direct bucket
  queries time out.
- Ensembl's R64 GTF has **4** `five_prime_utr` features for the whole genome —
  SGD does not annotate UTRs.

Rather than guess, `pipeline/03_build_examples.py --utr-table FILE` accepts a
hand-downloaded table (`gene_id<TAB>utr_length`, TSV). Per-gene lengths from it
override the fixed window; genes absent from it fall back to `--utr-length`.
That keeps the better annotation one file away without blocking the prototype.

Start codons are taken from the GTF's **`start_codon` features**, not derived
from CDS bounds — the GTF states them directly on both strands with no
off-by-one to reconstruct.

**Projection** (`data/coverage.py:project_onto_utr`) slices the genomic count
track to the UTR interval and reverses it on the minus strand, so the result is
oriented **cap→AUG** and indexed 1:1 with `utr_seq`.

## 7b. Commitment-readout window

For each transcript the pipeline also records 80S (RS) and Input 5′-end
counts over the **first 100 nt of coding sequence** (`--cds-length`),
starting at the annotated start codon and **following exons** — 283 yeast
genes have an intron in the CDS, often within the first few codons in
ribosomal-protein genes, and a genomic window would count intronic
positions that are never in the mRNA. Verified against the annotation: all
6,593 windows begin exactly at the start codon on both strands; 47 genes have
fewer than 100 coding nt and get a shorter window.

This region is not part of the chain's state space. The training loss sums
each track to one number per gene. The 100-nt length is a compromise between
integrating enough reads and limiting the elongation-rate confound (80S
density is initiation ÷ elongation, and elongation varies along a gene); the
exact value is absorbed by `rho_load`, only per-gene elongation differences
survive as noise.

## 7c. Data-driven 5'UTR boundaries

`scripts/call_utr_boundaries.py` replaces the fixed window with a per-gene
call from Input read coverage, written as a `--utr-table` TSV.

The Input fraction is total RNA, so reads appear wherever mRNA exists and
stop at the transcription start. Walking upstream from the start codon, the
call ends at the first run of `--min-gap` (default 10) uncovered positions; a
single dropout is noise, a sustained one is the end of the transcript.

Two guards, because the gap between yeast genes is crowded:

- a hard clip at the nearest **same-strand** upstream CDS end, so a window can
  never reach into a neighbour's coding sequence. Opposite-strand neighbours
  are ignored deliberately: their transcripts put no reads on this strand and
  so cannot contaminate this gene's coverage.
- genes with no coverage at their own start codon get **no call** and fall
  back to the fixed window, rather than inheriting a neighbour's UTR.

This needs full read coverage rather than 5′ ends, which step 2 now
accumulates for the Input fraction in the same single pass over the SAM
stream (difference array, so it costs two scalar operations per read rather
than a slice-add).

## 8. Transcripts passing

Measured against the real annotation (Ensembl R64-1-1 release 115):

| Stage | Count |
|---|---|
| `start_codon` features in the GTF | 6,601 |
| after dropping intron-split start codons (not stitched — a spliced 5′UTR is out of scope) | 6,593 |
| **pass all structural filters** (window on-chromosome, sequence reads ATG, no ambiguous bases, mitochondrial excluded) | **6,564** |
| dropped: mitochondrial (no cap-dependent scanning) | 28 |
| dropped: annotated start codon whose sequence is not ATG | 1 |

Identical at `--utr-length` 50 and 100.

The remaining filter is **coverage** (`--min-ssu-counts`, default 5), which
depends on real alignments. That number is not yet in this table: bowtie2 and
cutadapt are **not installed in this environment**, so the FASTQs have not been
run through. Everything upstream and downstream of the aligner has been
verified — see "Verification" below. Run `python pipeline/run_all.py` once the
binaries are present and step 3 prints the final tally.

## 9. Verification performed

Because the aligner is absent, the coordinate logic was tested directly against
the **real genome and real annotation** rather than left unexercised:

- **Round-trip, both strands.** Synthetic SAM records were generated whose 5′
  ends sit at known transcript offsets `[0, 17, 50]` of a real plus-strand gene
  (`YDR438W`) and a real minus-strand gene (`YDL246C`). After
  `accumulate_five_prime_ends` → `project_onto_utr`, the recovered nonzero
  indices are exactly `[0, 17, 50]` on both strands, with
  `len(counts) == len(utr_seq)` and `utr_seq[-3:] == "ATG"`.
- **Annotation cross-check.** 6,586 of 6,593 parsed start codons read `ATG` in
  the genome sequence; all 7 exceptions are caught by the filters above.
- **Import-time asserts** in `data/coverage.py` pin the CIGAR arithmetic, the
  1-based→0-based conversion, the strand-dependent offset sign, and the
  plus/minus UTR interval geometry.
- **`train.py --data archer`** was run end to end against a cache built by
  `pipeline/03_build_examples.py` from real sequence and annotation with
  stand-in count tracks, confirming the loader, feature path and training loop
  accept real-shaped examples. That temporary cache was deleted; no fabricated
  data is left in the tree.
