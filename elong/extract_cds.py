"""Full coding sequence per transcript -> codon counts.

    python elong/extract_cds.py        # -> cache/cds_codons.npz

The training cache carries only the first 100 nt of each CDS, which is the
commitment-readout window.  Codon composition needs the whole thing, so this
walks the GTF's CDS exons and splices them, strand-aware.

Nothing here touches the model or the training cache -- it writes its own
file, which only elong/fit.py reads.
"""

import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.coverage import revcomp
from pipeline._common import CACHE_DIR, GENOME_FA, GTF

OUT = os.path.join(CACHE_DIR, "cds_codons.npz")


def gene_names(gtf_path=GTF):
    """{transcript_id: gene_name}, for the ribosomal-protein sanity check."""
    out = {}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#") or 'gene_name "' not in line:
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "transcript":
                continue
            a = f[8]
            tx = _attr(a, "transcript_id")
            nm = _attr(a, "gene_name")
            if tx and nm:
                out[tx] = nm
    return out


def _attr(attrs, key):
    marker = key + ' "'
    i = attrs.find(marker)
    if i < 0:
        return None
    i += len(marker)
    j = attrs.find('"', i)
    return attrs[i:j]


def cds_sequence(genome, chrom, strand, blocks):
    """Spliced coding sequence, 5'->3', from a transcript's CDS exons."""
    blocks = sorted(blocks)
    seq = "".join(genome[chrom][s:e] for s, e in blocks)
    return revcomp(seq) if strand == "-" else seq


def codon_counts(seq):
    """{codon: n} over complete in-frame codons.  Partial tails are dropped."""
    seq = seq.upper().replace("U", "T")
    c = collections.Counter()
    for i in range(0, len(seq) - len(seq) % 3, 3):
        cod = seq[i:i + 3]
        if len(cod) == 3 and not (set(cod) - set("ACGT")):
            c[cod] += 1
    return c


def save_codon_counts(path, per_tx, names):
    """Flat npz: one row per transcript, 64 columns in a fixed codon order."""
    from elong.tai import CODONS
    tx_ids = sorted(per_tx)
    mat = np.zeros((len(tx_ids), len(CODONS)), dtype=np.int32)
    for r, tx in enumerate(tx_ids):
        for c, cod in enumerate(CODONS):
            mat[r, c] = per_tx[tx].get(cod, 0)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path, transcript_id=np.array(tx_ids),
        gene_name=np.array([names.get(t, "") for t in tx_ids]),
        codons=np.array(CODONS), counts=mat)


def load_codon_counts(path=OUT):
    """-> {transcript_id: ({codon: n}, gene_name)}"""
    z = np.load(path, allow_pickle=False)
    cods = [str(c) for c in z["codons"]]
    out = {}
    for tx, nm, row in zip(z["transcript_id"], z["gene_name"], z["counts"]):
        out[str(tx)] = ({c: int(n) for c, n in zip(cods, row) if n}, str(nm))
    return out


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "pipeline"))
    import importlib
    mod = importlib.import_module("03_build_examples")

    genome = mod.read_fasta(GENOME_FA)
    _starts, _dup, cds_blocks = mod.parse_start_codons(GTF)
    strands = {tx: (chrom, strand)
               for tx, chrom, strand, _s, _e in _starts}
    names = gene_names(GTF)

    per_tx, skipped = {}, collections.Counter()
    for tx, blocks in cds_blocks.items():
        if tx not in strands:
            skipped["no_start_codon"] += 1
            continue
        chrom, strand = strands[tx]
        if chrom not in genome:
            skipped["unknown_chrom"] += 1
            continue
        seq = cds_sequence(genome, chrom, strand, blocks)
        if len(seq) < 90:                      # under 30 codons: composition is noise
            skipped["too_short"] += 1
            continue
        c = codon_counts(seq)
        if sum(c.values()) < 30:
            skipped["too_few_clean_codons"] += 1
            continue
        per_tx[tx] = c

    save_codon_counts(OUT, per_tx, names)
    lens = np.array([sum(c.values()) for c in per_tx.values()])
    print("[cds] {} transcripts, median {} codons (range {}-{})".format(
        len(per_tx), int(np.median(lens)), lens.min(), lens.max()))
    if skipped:
        print("[cds] skipped: {}".format(dict(skipped)))
    print("[cds] wrote {}".format(OUT))


if __name__ == "__main__":
    main()
