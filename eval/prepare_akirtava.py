"""Akirtava, May & McManus 2025 (NAR) supplement -> a flat table we can score.

    python eval/prepare_akirtava.py

Reads the supplementary workbook and writes eval/akirtava_scer_wt.tsv.

The identifier does the hard work for us:

    YMR316W|chrXIII;904760;904824|WT
    <ORF>  |<chrom>;<TSS>  ;<ATG>  |<allele>

Both coordinates are sacCer3, the same assembly the pipeline uses, so the
transcript leader can be cut straight from the genome by position.  That
sidesteps the isoform problem entirely: these genes have 2-8 alternative-TSS
isoforms each, and joining gene-name -> one of our called UTRs would silently
pair our isoform with their measurement of a different one.  Using their
coordinates means we score exactly the leader they assayed.

Filters applied, each counted:
  * S. cerevisiae only -- S. paradoxus rows (Spar_*) have no TCP-seq here
  * WT allele only -- the AUG->AAG mutants are a perturbation, not natural
    biology, and the model has no notion of an engineered construct
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "gkaf165_supplemental_files",
                    "Akirtava_et_al_2024_Supp_Data_Rev1.xlsx")
OUT = os.path.join(HERE, "akirtava_scer_wt.tsv")

# chrXIII -> XIII : the workbook uses UCSC-style names, our reference is
# Ensembl R64 (same assembly, bare roman numerals).
ID_RE = re.compile(r"^(?P<orf>[^|]+)\|(?P<chrom>[^;]+);(?P<tss>\d+);(?P<atg>\d+)\|(?P<allele>.+)$")


def parse_id(s):
    m = ID_RE.match(str(s).strip())
    if not m:
        return None
    d = m.groupdict()
    chrom = d["chrom"]
    if chrom.startswith("Spar"):
        return None                      # S. paradoxus
    if not chrom.startswith("chr"):
        return None
    d["chrom"] = chrom[3:]               # chrXIII -> XIII
    d["tss"] = int(d["tss"])
    d["atg"] = int(d["atg"])
    # W = Watson (+), C = Crick (-).  Cross-checked against the coordinates
    # below rather than trusted: TSS is upstream, so TSS < ATG on +.
    d["strand"] = "+" if d["tss"] < d["atg"] else "-"
    return d


def _num(v):
    try:
        f = float(v)
        return f if f == f else None     # drop NaN
    except (TypeError, ValueError):
        return None


def main():
    if not os.path.exists(XLSX):
        sys.exit("\nWorkbook not found at {}\n".format(XLSX))
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)

    def sheet_rows(name):
        ws = wb[name]
        it = ws.iter_rows(values_only=True)
        hdr = [str(h).strip() if h is not None else "" for h in next(it)]
        for r in it:
            yield dict(zip(hdr, r))

    # Table_1: FACS-Seq, protein output per TL.
    facs, stats = {}, {"rows": 0, "parsed": 0, "spar_or_bad": 0, "not_wt": 0}
    for r in sheet_rows("Table_1"):
        stats["rows"] += 1
        d = parse_id(r.get("@UTR"))
        if d is None:
            stats["spar_or_bad"] += 1
            continue
        if d["allele"].strip().upper() != "WT":
            stats["not_wt"] += 1
            continue
        stats["parsed"] += 1
        facs[r["@UTR"]] = (d, _num(r.get("MeanYFP")), _num(r.get("MeanRNA")),
                           _num(r.get("Length")), _num(r.get("uAUG")))

    # Table_2: PoLib-Seq, ribosome loading (% translating) per construct.
    teff = {}
    for r in sheet_rows("Table_2"):
        d = parse_id(r.get("Construct"))
        if d is None or d["allele"].strip().upper() != "WT":
            continue
        vals = [_num(r.get("R1-TEff")), _num(r.get("R2-TEff"))]
        vals = [v for v in vals if v is not None]
        if vals:
            teff[r["Construct"]] = sum(vals) / len(vals)

    # Table_3: their features AND the leader sequence itself.  Keyed on
    # "Gene", not "@UTR".  UTRseq is the assayed leader, so we score exactly
    # the sequence they measured rather than re-cutting it from the genome
    # and risking a coordinate or isoform mismatch.
    feats = {}
    for r in sheet_rows("Table_3"):
        key = r.get("Gene")
        d = parse_id(key)
        if d is None or d["allele"].strip().upper() != "WT":
            continue
        seq = str(r.get("UTRseq") or "").strip().upper()
        feats[key] = (_num(r.get("LSMKozakStart")), str(r.get("Kozak") or ""),
                      seq, _num(r.get("KozakStrength")))
    wb.close()

    keys = sorted(set(facs) | set(teff))
    keys = sorted(set(keys) | set(feats))
    cols = ["id", "orf", "chrom", "strand", "tss", "atg", "tl_length",
            "uAUG", "MeanYFP", "MeanRNA", "TEff", "LSMKozakStart", "utr_seq"]
    n = 0
    with open(OUT, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for k in keys:
            if k in facs:
                d, yfp, rna, length, uaug = facs[k]
            else:
                d = parse_id(k)
                yfp = rna = length = uaug = None
            if d is None:
                continue
            lsm, koz, seq, _ks = feats.get(k, (None, "", "", None))
            if not seq:
                continue                 # no sequence -> nothing to score
            row = [k, d["orf"], d["chrom"], d["strand"], d["tss"], d["atg"],
                   length, uaug, yfp, rna, teff.get(k), lsm, seq]
            fh.write("\t".join("" if v is None else str(v) for v in row) + "\n")
            n += 1

    print("[prep] Table_1 rows {rows}, kept {parsed} (dropped {spar_or_bad} "
          "non-Scer/unparsable, {not_wt} non-WT)".format(**stats))
    print("[prep] S.cer WT with ribosome loading (TEff): {}".format(len(teff)))
    print("[prep] S.cer WT with sequence + Kozak features: {}".format(len(feats)))
    print("[prep] wrote {} rows -> {}".format(n, OUT))


if __name__ == "__main__":
    main()
