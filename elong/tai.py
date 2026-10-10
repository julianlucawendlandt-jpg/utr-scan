"""tRNA Adaptation Index, built from the reference GTF's own tRNA genes.

    python elong/tai.py            # print the weights and a validation

tAI scores how fast a gene's codons are decoded.  Each codon is read by a
tRNA; the more gene copies of that tRNA the genome carries, the more of it in
the cell, and the quicker the ribosome steps.  A gene built from
abundant-tRNA codons elongates fast.

The weights come only from counting tRNA genes -- no expression data enters,
which is the reason this is used here and CAI is not (see README.md).

The availability of codon i is

    W_i = sum over tRNAs that can read it of (1 - s) * copies

where s penalises non-Watson-Crick pairing.  The codon's THIRD base pairs the
anticodon's FIRST base (position 34), antiparallel, so for each two-base
prefix NN:

    anticodon ANN  ->  reads NNT  (A34 is modified to inosine in eukaryotes,
                       and I reads T, C and A)
    anticodon GNN  ->  reads NNC  Watson-Crick, and NNT by G:U wobble
    anticodon TNN  ->  reads NNA  Watson-Crick, and NNG by U:G wobble
    anticodon CNN  ->  reads NNG  Watson-Crick

giving

    W(NNT) = (1 - s_IU) * t_ANN + (1 - s_GU) * t_GNN
    W(NNC) = (1 - s_IC) * t_ANN +             t_GNN
    W(NNA) = (1 - s_IA) * t_ANN +             t_TNN
    W(NNG) =                       (1 - s_UG) * t_TNN + t_CNN

s values are dos Reis, Savva & Wernisch 2004, eukaryote set.  They are
reproduced here rather than imported; the source is one paper and the numbers
are four constants, so a dependency would cost more than it saves.
"""

import collections
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline._common import GTF

BASES = "TCAG"
CODONS = [a + b + c for a in BASES for b in BASES for c in BASES]

STOP_CODONS = ("TAA", "TAG", "TGA")

# dos Reis 2004, eukaryotes.  Higher s = worse pairing = slower.
S_IU, S_IC, S_IA, S_UG = 0.0, 0.28, 0.9999, 0.68

# tRNA genes in the Ensembl R64 GTF are named tX(ANTICODON)Chr, e.g.
# tR(ACG)D = tRNA-Arg, anticodon ACG, on chromosome IV.
TRNA_RE = re.compile(r'gene_id "t([A-Z*])\(([ACGTU]{3})\)')


def trna_gene_counts(gtf_path=GTF):
    """{anticodon: number of tRNA genes} from the reference annotation.

    Counted on `gene` lines only -- a tRNA gene also appears as a transcript
    and an exon, and counting those would triple every number.  The anticodon
    is written 5'->3', as in the gene name.
    """
    counts = collections.Counter()
    with open(gtf_path) as fh:
        for line in fh:
            if 'gene_biotype "tRNA"' not in line:
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            m = TRNA_RE.search(f[8])
            if m:
                counts[m.group(2).replace("U", "T")] += 1
    assert counts, "no tRNA genes found in {} -- has the GTF changed?".format(gtf_path)
    return counts


def codon_weights(trna_counts=None):
    """{codon: w} in (0, 1], the relative decoding speed of each sense codon.

    Zero-availability codons (no tRNA can read them at all) are given the
    geometric mean of the rest, the standard treatment: a zero would
    annihilate the whole gene's geometric mean, and the real reason such a
    codon is decoded at all is a modification this model does not represent.
    """
    t = trna_counts if trna_counts is not None else trna_gene_counts()

    W = {}
    for first in BASES:
        for second in BASES:
            nn = first + second
            a, g, u, c = (t.get(x + nn, 0) for x in ("A", "G", "T", "C"))
            W[nn + "T"] = (1 - S_IU) * a + (1 - 0.41) * g   # 0.41 = s_GU
            W[nn + "C"] = (1 - S_IC) * a + g
            W[nn + "A"] = (1 - S_IA) * a + u
            W[nn + "G"] = (1 - S_UG) * u + c

    for stop in STOP_CODONS:
        W.pop(stop, None)

    top = max(W.values())
    assert top > 0, "every codon has zero tRNA availability"
    w = {k: v / top for k, v in W.items()}

    nonzero = [v for v in w.values() if v > 0]
    fill = float(np.exp(np.mean(np.log(nonzero))))
    return {k: (v if v > 0 else fill) for k, v in w.items()}


def gene_tai(codon_counts, w):
    """Geometric mean of w over a gene's codons.  codon_counts: {codon: n}.

    Geometric rather than arithmetic because the quantity being averaged is a
    per-codon *rate*, and the time to cross the gene is the sum of reciprocal
    rates; the geometric mean is the standard tAI definition and keeps a
    single very slow codon visible instead of letting fast ones drown it.
    """
    tot, acc = 0, 0.0
    for codon, n in codon_counts.items():
        if codon in w and n:
            acc += n * np.log(w[codon])
            tot += n
    if tot == 0:
        return None
    return float(np.exp(acc / tot))


def main():
    t = trna_gene_counts()
    print("[tai] {} tRNA genes across {} anticodons in {}".format(
        sum(t.values()), len(t), os.path.basename(GTF)))
    w = codon_weights(t)
    order = sorted(w.items(), key=lambda kv: -kv[1])
    print("\nfastest codons: " + "  ".join("%s %.2f" % kv for kv in order[:6]))
    print("slowest codons: " + "  ".join("%s %.2f" % kv for kv in order[-6:]))

    # Validation: ribosomal protein genes are the textbook case of codon
    # optimisation, so they must land near the top.  If they do not, the
    # weights are wrong and nothing downstream is worth running.
    npz = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "cache", "cds_codons.npz")
    if not os.path.exists(npz):
        print("\n(run elong/extract_cds.py for the ribosomal-protein check)")
        return
    from elong.extract_cds import load_codon_counts
    genes = load_codon_counts(npz)
    scored = {tx: gene_tai(c, w) for tx, (c, _nm) in genes.items()}
    scored = {k: v for k, v in scored.items() if v is not None}
    names = {tx: nm for tx, (_c, nm) in genes.items()}
    vals = np.array(list(scored.values()))
    rp = [v for k, v in scored.items()
          if names.get(k, "").startswith(("RPL", "RPS"))]
    pct = 100.0 * np.mean(vals[:, None] < np.array(rp)[None, :]) if rp else float("nan")
    print("\n[tai] {} genes scored, median tAI {:.3f}".format(len(vals), np.median(vals)))
    print("[tai] {} ribosomal protein genes, median tAI {:.3f} "
          "(mean percentile {:.0f})".format(len(rp), np.median(rp), pct))
    print("      RP genes are the textbook codon-optimised set; a high "
          "percentile here is the check that the weights are right.")


if __name__ == "__main__":
    main()
