# Added evaluations — machine-generated summary

Generated from the result files; see run_missing_evaluations.py for commands.

## A. Unselected human 40S (gse139131_unsel)

| family | bin | status | dLL r1→r2 | dLL r2→r1 | null p r1→r2 | null p r2→r1 | power |
|---|---|---|---|---|---|---|---|
| S1 | 1 | dataset_specific | +0.0038 | -0.0003 | nan | nan | nan |
| S1 | 5 | dataset_specific | +0.0049 | +0.0014 | nan | nan | nan |
| S1 | 10 | dataset_specific | +0.0065 | +0.0024 | 0.00990099 | 0.00990099 | nan |
| S1 | 20 | dataset_specific | +0.0104 | +0.0028 | nan | nan | nan |
| S3 | 1 | dataset_specific | +0.0085 | +0.0008 | 0.00990099 | 0.00990099 | nan |
| S3 | 5 | dataset_specific | +0.0096 | +0.0040 | nan | nan | nan |
| S3 | 10 | coarse_resolution_recoverable | +0.0124 | +0.0057 | 0.00990099 | 0.00990099 | nan |
| S3 | 20 | null_pending | +0.0150 | +0.0066 | nan | nan | nan |
| S5 | 1 | inconclusive | -0.0067 | -0.0194 | nan | nan | nan |
| S5 | 5 | inconclusive | -0.0093 | -0.0198 | nan | nan | nan |
| S5 | 10 | inconclusive | -0.0063 | -0.0173 | 0.00990099 | 0.029703 | nan |
| S5 | 20 | inconclusive | +0.0005 | -0.0096 | nan | nan | nan |
| SW | 1 | dataset_specific | +0.0129 | +0.0021 | 0.00990099 | 0.00990099 | 0 |
| SW | 5 | dataset_specific | +0.0092 | -0.0047 | 0.00990099 | 0.306931 | 0 |
| SW | 10 | inconclusive | +0.0034 | -0.0046 | 0.00990099 | 0.029703 | 0 |
| SW | 20 | inconclusive | +0.0014 | -0.0036 | nan | nan | 0 |
| ST_paired | 1 | not_recoverable_at_depth | +0.0006 | +0.0001 | nan | nan | 1 |
| ST_paired | 5 | not_recoverable_at_depth | +0.0007 | +0.0001 | nan | nan | 1 |
| ST_paired | 10 | not_recoverable_at_depth | +0.0008 | +0.0006 | 0.237624 | 0.168317 | 1 |
| ST_paired | 20 | not_recoverable_at_depth | +0.0005 | +0.0005 | nan | nan | 1 |
| ST_dG | 1 | not_recoverable_at_depth | +0.0008 | +0.0014 | nan | nan | 1 |
| ST_dG | 5 | not_recoverable_at_depth | +0.0005 | +0.0017 | nan | nan | 1 |
| ST_dG | 10 | dataset_specific | +0.0009 | +0.0024 | 1 | 1 | 1 |
| ST_dG | 20 | not_recoverable_at_depth | +0.0019 | +0.0025 | nan | nan | 1 |
| ST | 1 | not_recoverable_at_depth | +0.0018 | +0.0022 | 0.970297 | 0.0990099 | 1 |
| ST | 5 | dataset_specific | +0.0018 | +0.0028 | 1 | 0.693069 | 1 |
| ST | 10 | dataset_specific | +0.0023 | +0.0041 | 1 | 0.633663 | 1 |
| ST | 20 | not_recoverable_at_depth | +0.0026 | +0.0034 | 1 | 0.881188 | 1 |
| AUG | 1 | recoverable | +0.0121 | +0.0063 | 0.00990099 | 0.00990099 | 0.94 |
| AUG | 5 | recoverable | +0.0141 | +0.0075 | 0.00990099 | 0.00990099 | 0.94 |
| AUG | 10 | recoverable | +0.0109 | +0.0053 | 0.00990099 | 0.019802 | 0.94 |
| AUG | 20 | recoverable | +0.0099 | +0.0058 | 0.00990099 | 0.00990099 | 0.94 |
| UORF | 1 | null_pending | +0.0120 | +0.0058 | nan | nan | nan |
| UORF | 5 | null_pending | +0.0141 | +0.0072 | nan | nan | nan |
| UORF | 10 | fails_position_matched_null | +0.0107 | +0.0051 | 0.188119 | 0.237624 | nan |
| UORF | 20 | null_pending | +0.0096 | +0.0053 | nan | nan | nan |

### Selected vs unselected on common transcripts (10 nt)

| depth | seed | family | common n | selected | unselected | difference [95% CI] |
|---|---|---|---|---|---|---|
| observed |  | SW | 1218 | +0.0103 | -0.0016 | +0.0119 [+0.0045, +0.0196] |
| observed |  | ST | 1218 | +0.0016 | +0.0033 | -0.0017 [-0.0052, +0.0016] |
| observed |  | AUG | 1218 | +0.0086 | +0.0066 | +0.0020 [-0.0010, +0.0050] |
| matched | 0 | SW | 790 | +0.0103 | -0.0050 | +0.0153 [+0.0081, +0.0228] |
| matched | 0 | ST | 790 | +0.0016 | +0.0027 | -0.0011 [-0.0048, +0.0025] |
| matched | 0 | AUG | 790 | +0.0081 | +0.0074 | +0.0007 [-0.0028, +0.0041] |
| matched | 1 | SW | 794 | +0.0076 | -0.0060 | +0.0136 [+0.0066, +0.0206] |
| matched | 1 | ST | 794 | +0.0017 | +0.0027 | -0.0009 [-0.0047, +0.0026] |
| matched | 1 | AUG | 794 | +0.0074 | +0.0069 | +0.0005 [-0.0029, +0.0041] |
| matched | 2 | SW | 778 | +0.0074 | -0.0061 | +0.0134 [+0.0070, +0.0197] |
| matched | 2 | ST | 778 | +0.0014 | +0.0030 | -0.0016 [-0.0054, +0.0023] |
| matched | 2 | AUG | 778 | +0.0080 | +0.0066 | +0.0014 [-0.0020, +0.0048] |

### Same-study control (within-library / dataset-specific only)

| depth | seed | family | dataset | direction | dLL [95% CI] | null p |
|---|---|---|---|---|---|---|
| observed |  | SW | gse139131_eif3b | rep1->rep1 | +0.0071 [+0.0037, +0.0108] | 0.00990099 |
| observed |  | SW | gse139131_unsel | rep1->rep1 | -0.0059 [-0.0133, +0.0015] | 0.049505 |
| observed |  | SW | gse139131_unsel | rep2->rep2 | +0.0053 [-0.0025, +0.0129] | 0.00990099 |
| observed |  | SW | gse139131_unsel | rep1->rep2 | -0.0018 [-0.0077, +0.0044] |  |
| observed |  | SW | gse139131_unsel | rep2->rep1 | -0.0048 [-0.0144, +0.0058] |  |
| observed |  | ST | gse139131_eif3b | rep1->rep1 | +0.0035 [+0.0012, +0.0058] | 0.00990099 |
| observed |  | ST | gse139131_unsel | rep1->rep1 | +0.0055 [+0.0015, +0.0094] | 0.386139 |
| observed |  | ST | gse139131_unsel | rep2->rep2 | +0.0027 [-0.0001, +0.0056] | 1 |
| observed |  | ST | gse139131_unsel | rep1->rep2 | +0.0022 [-0.0010, +0.0054] |  |
| observed |  | ST | gse139131_unsel | rep2->rep1 | +0.0041 [+0.0007, +0.0074] |  |
| observed |  | AUG | gse139131_eif3b | rep1->rep1 | +0.0036 [+0.0016, +0.0058] | 0.00990099 |
| observed |  | AUG | gse139131_unsel | rep1->rep1 | +0.0045 [+0.0013, +0.0081] | 0.029703 |
| observed |  | AUG | gse139131_unsel | rep2->rep2 | +0.0120 [+0.0081, +0.0158] | 0.00990099 |
| observed |  | AUG | gse139131_unsel | rep1->rep2 | +0.0098 [+0.0064, +0.0132] |  |
| observed |  | AUG | gse139131_unsel | rep2->rep1 | +0.0053 [+0.0011, +0.0097] |  |
| matched | 0 | SW | gse139131_eif3b | rep1->rep1 | +0.0051 [-0.0001, +0.0100] |  |
| matched | 0 | SW | gse139131_unsel | rep1->rep1 | -0.0107 [-0.0197, -0.0022] |  |
| matched | 0 | SW | gse139131_unsel | rep2->rep2 | +0.0039 [-0.0041, +0.0128] |  |
| matched | 0 | SW | gse139131_unsel | rep1->rep2 | -0.0061 [-0.0138, +0.0015] |  |
| matched | 0 | SW | gse139131_unsel | rep2->rep1 | -0.0031 [-0.0133, +0.0098] |  |
| matched | 0 | ST | gse139131_eif3b | rep1->rep1 | +0.0029 [+0.0003, +0.0054] |  |
| matched | 0 | ST | gse139131_unsel | rep1->rep1 | +0.0058 [+0.0013, +0.0103] |  |
| matched | 0 | ST | gse139131_unsel | rep2->rep2 | +0.0027 [-0.0005, +0.0058] |  |
| matched | 0 | ST | gse139131_unsel | rep1->rep2 | +0.0021 [-0.0015, +0.0060] |  |
| matched | 0 | ST | gse139131_unsel | rep2->rep1 | +0.0044 [+0.0006, +0.0083] |  |
| matched | 0 | AUG | gse139131_eif3b | rep1->rep1 | +0.0020 [-0.0005, +0.0046] |  |
| matched | 0 | AUG | gse139131_unsel | rep1->rep1 | +0.0042 [+0.0003, +0.0082] |  |
| matched | 0 | AUG | gse139131_unsel | rep2->rep2 | +0.0111 [+0.0072, +0.0153] |  |
| matched | 0 | AUG | gse139131_unsel | rep1->rep2 | +0.0088 [+0.0052, +0.0124] |  |
| matched | 0 | AUG | gse139131_unsel | rep2->rep1 | +0.0054 [+0.0007, +0.0103] |  |
| matched | 1 | SW | gse139131_eif3b | rep1->rep1 | +0.0038 [-0.0007, +0.0088] |  |
| matched | 1 | SW | gse139131_unsel | rep1->rep1 | -0.0110 [-0.0190, -0.0036] |  |
| matched | 1 | SW | gse139131_unsel | rep2->rep2 | +0.0029 [-0.0061, +0.0143] |  |
| matched | 1 | SW | gse139131_unsel | rep1->rep2 | -0.0065 [-0.0137, +0.0004] |  |
| matched | 1 | SW | gse139131_unsel | rep2->rep1 | -0.0031 [-0.0154, +0.0110] |  |
| matched | 1 | ST | gse139131_eif3b | rep1->rep1 | +0.0033 [+0.0010, +0.0058] |  |
| matched | 1 | ST | gse139131_unsel | rep1->rep1 | +0.0056 [+0.0014, +0.0098] |  |
| matched | 1 | ST | gse139131_unsel | rep2->rep2 | +0.0024 [-0.0005, +0.0054] |  |
| matched | 1 | ST | gse139131_unsel | rep1->rep2 | +0.0018 [-0.0016, +0.0051] |  |
| matched | 1 | ST | gse139131_unsel | rep2->rep1 | +0.0039 [+0.0001, +0.0077] |  |
| matched | 1 | AUG | gse139131_eif3b | rep1->rep1 | +0.0031 [+0.0006, +0.0058] |  |
| matched | 1 | AUG | gse139131_unsel | rep1->rep1 | +0.0037 [+0.0001, +0.0076] |  |
| matched | 1 | AUG | gse139131_unsel | rep2->rep2 | +0.0114 [+0.0074, +0.0154] |  |
| matched | 1 | AUG | gse139131_unsel | rep1->rep2 | +0.0091 [+0.0057, +0.0126] |  |
| matched | 1 | AUG | gse139131_unsel | rep2->rep1 | +0.0044 [-0.0000, +0.0089] |  |
| matched | 2 | SW | gse139131_eif3b | rep1->rep1 | +0.0042 [-0.0006, +0.0091] |  |
| matched | 2 | SW | gse139131_unsel | rep1->rep1 | -0.0040 [-0.0112, +0.0036] |  |
| matched | 2 | SW | gse139131_unsel | rep2->rep2 | +0.0037 [-0.0056, +0.0141] |  |
| matched | 2 | SW | gse139131_unsel | rep1->rep2 | +0.0017 [-0.0046, +0.0086] |  |
| matched | 2 | SW | gse139131_unsel | rep2->rep1 | -0.0036 [-0.0155, +0.0107] |  |
| matched | 2 | ST | gse139131_eif3b | rep1->rep1 | +0.0032 [+0.0006, +0.0056] |  |
| matched | 2 | ST | gse139131_unsel | rep1->rep1 | +0.0061 [+0.0015, +0.0109] |  |
| matched | 2 | ST | gse139131_unsel | rep2->rep2 | +0.0025 [-0.0007, +0.0054] |  |
| matched | 2 | ST | gse139131_unsel | rep1->rep2 | +0.0019 [-0.0018, +0.0056] |  |
| matched | 2 | ST | gse139131_unsel | rep2->rep1 | +0.0045 [+0.0007, +0.0086] |  |
| matched | 2 | AUG | gse139131_eif3b | rep1->rep1 | +0.0044 [+0.0019, +0.0071] |  |
| matched | 2 | AUG | gse139131_unsel | rep1->rep1 | +0.0030 [-0.0007, +0.0069] |  |
| matched | 2 | AUG | gse139131_unsel | rep2->rep2 | +0.0119 [+0.0079, +0.0159] |  |
| matched | 2 | AUG | gse139131_unsel | rep1->rep2 | +0.0087 [+0.0050, +0.0127] |  |
| matched | 2 | AUG | gse139131_unsel | rep2->rep1 | +0.0044 [-0.0000, +0.0092] |  |

## B. Yeast (single SSU library — within-library evidence only)

| family | bin | dLL [95% CI] | null p | status | optimistic power |
|---|---|---|---|---|---|
| SW | 1 | +0.0398 [+0.0051, +0.0687] | 0.00990099 | dataset_specific_support | 0 |
| SW | 5 | +0.0312 [+0.0127, +0.0513] | 0.00990099 | dataset_specific_support | 0 |
| SW | 10 | +0.0134 [-0.0052, +0.0310] | 0.00990099 | no_within_library_support | 0 |
| SW | 20 | +0.0018 [-0.0211, +0.0249] | nan | no_within_library_support | 0 |
| ST | 1 | +0.0045 [-0.0005, +0.0096] | 0.029703 | no_within_library_support | 1 |
| ST | 5 | +0.0025 [-0.0015, +0.0068] | 0.0891089 | no_within_library_support | 1 |
| ST | 10 | +0.0021 [-0.0014, +0.0059] | 0.178218 | no_within_library_support | 1 |
| ST | 20 | -0.0011 [-0.0041, +0.0018] | 0.782178 | no_within_library_support | 1 |
| AUG | 1 | +0.0161 [+0.0009, +0.0293] | 0.00990099 | dataset_specific_support | 0 |
| AUG | 5 | +0.0148 [+0.0000, +0.0279] | 0.00990099 | dataset_specific_support | 0 |
| AUG | 10 | +0.0172 [-0.0037, +0.0343] | 0.00990099 | no_within_library_support | 0 |
| AUG | 20 | +0.0080 [-0.0126, +0.0260] | 0.00990099 | no_within_library_support | 0 |
| UORF | 1 | +0.0167 [+0.0017, +0.0299] | 0.19802 | fails_position_matched_null | nan |
| UORF | 5 | +0.0148 [+0.0003, +0.0279] | 0.217822 | fails_position_matched_null | nan |
| UORF | 10 | +0.0161 [-0.0014, +0.0309] | 0.475248 | no_within_library_support | nan |
| UORF | 20 | +0.0080 [-0.0093, +0.0238] | nan | no_within_library_support | nan |
| SW+ST+AUG | 1 | +0.0509 [+0.0220, +0.0767] | nan | held_out_support_no_null_defined | nan |
| SW+ST+AUG | 5 | +0.0394 [+0.0250, +0.0555] | nan | held_out_support_no_null_defined | nan |
| SW+ST+AUG | 10 | +0.0344 [+0.0094, +0.0571] | nan | held_out_support_no_null_defined | nan |
| SW+ST+AUG | 20 | +0.0180 [-0.0016, +0.0382] | nan | no_within_library_support | nan |

Yeast has only one independent SSU library; cross-replicate recoverability is untestable.

## C. Reporter utility

Model choice (frozen, from TCP-seq results only):

- yeast: SW+ST+AUG at 1 nt (held-out chromosome dLL/read, rep1->rep1 (single library) = 0.0508506)
- human: SW+ST+AUG at 1 nt (min over cross-replicate directions of held-out dLL/read = 0.0254727)

| reporter | target | 40S source | split | bootstrap unit | stratum | model | n | R² | ΔR² vs seq [95% CI] | P(ΔR²≤0) |
|---|---|---|---|---|---|---|---|---|---|---|
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | all | seq | 4433 | 0.5447 |  |  |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | all | 40S | 4433 | 0.1600 | -0.3847 [-0.4150, -0.3557] | 1 |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | all | seq+40S | 4433 | 0.5611 | +0.0164 [+0.0111, +0.0222] | 0 |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | all | late_fusion | 4433 | 0.5492 | +0.0045 [+0.0006, +0.0080] | 0.0105 |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | all | residual | 4433 | 0.5558 | +0.0111 [+0.0066, +0.0156] | 0 |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | no_uAUG | seq | 3701 | 0.2435 |  |  |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | no_uAUG | 40S | 3701 | -0.1640 | -0.4074 [-0.4638, -0.3531] | 1 |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | no_uAUG | seq+40S | 3701 | 0.2814 | +0.0379 [+0.0275, +0.0487] | 0 |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | no_uAUG | late_fusion | 3701 | 0.2470 | +0.0035 [-0.0029, +0.0101] | 0.1445 |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | no_uAUG | residual | 3701 | 0.2678 | +0.0243 [+0.0170, +0.0323] | 0 |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | >=1_uAUG | seq | 732 | 0.2135 |  |  |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | >=1_uAUG | 40S | 732 | -0.9191 | -1.1325 [-1.3416, -0.9517] | 1 |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | >=1_uAUG | seq+40S | 732 | 0.2205 | +0.0070 [-0.0112, +0.0254] | 0.2225 |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | >=1_uAUG | late_fusion | 732 | 0.2290 | +0.0155 [+0.0017, +0.0289] | 0.015 |
| akirtava_yeast | TEff | rep1 | chrom_folds | gene | >=1_uAUG | residual | 732 | 0.2210 | +0.0075 [-0.0077, +0.0236] | 0.175 |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | all | seq | 5179 | 0.6607 |  |  |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | all | 40S | 5179 | 0.3073 | -0.3535 [-0.3828, -0.3260] | 1 |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | all | seq+40S | 5179 | 0.6622 | +0.0015 [-0.0003, +0.0033] | 0.0415 |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | all | late_fusion | 5179 | 0.6614 | +0.0007 [-0.0008, +0.0023] | 0.178 |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | all | residual | 5179 | 0.6618 | +0.0011 [-0.0001, +0.0022] | 0.031 |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | no_uAUG | seq | 4156 | 0.4258 |  |  |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | no_uAUG | 40S | 4156 | -0.1532 | -0.5789 [-0.6457, -0.5147] | 1 |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | no_uAUG | seq+40S | 4156 | 0.4319 | +0.0061 [+0.0022, +0.0098] | 0 |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | no_uAUG | late_fusion | 4156 | 0.4211 | -0.0047 [-0.0075, -0.0019] | 0.9985 |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | no_uAUG | residual | 4156 | 0.4339 | +0.0081 [+0.0060, +0.0103] | 0 |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | >=1_uAUG | seq | 1023 | -0.2105 |  |  |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | >=1_uAUG | 40S | 1023 | -1.5179 | -1.3075 [-1.6288, -1.0501] | 1 |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | >=1_uAUG | seq+40S | 1023 | -0.2138 | -0.0033 [-0.0130, +0.0060] | 0.743 |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | >=1_uAUG | late_fusion | 1023 | -0.1936 | +0.0169 [+0.0079, +0.0260] | 0.0005 |
| akirtava_yeast | MeanYFP | rep1 | chrom_folds | gene | >=1_uAUG | residual | 1023 | -0.2218 | -0.0113 [-0.0176, -0.0051] | 1 |
| sample2019_human | rl | rep1 | random_balanced | leader | all | seq | 9486 | 0.5542 |  |  |
| sample2019_human | rl | rep1 | random_balanced | leader | all | 40S | 9486 | 0.2192 | -0.3350 [-0.3562, -0.3131] | 1 |
| sample2019_human | rl | rep1 | random_balanced | leader | all | seq+40S | 9486 | 0.5722 | +0.0180 [+0.0127, +0.0235] | 0 |
| sample2019_human | rl | rep1 | random_balanced | leader | all | late_fusion | 9486 | 0.5666 | +0.0124 [+0.0081, +0.0168] | 0 |
| sample2019_human | rl | rep1 | random_balanced | leader | all | residual | 9486 | 0.5639 | +0.0097 [+0.0055, +0.0139] | 0 |
| sample2019_human | rl | rep1 | random_balanced | leader | no_uAUG | seq | 7557 | 0.5156 |  |  |
| sample2019_human | rl | rep1 | random_balanced | leader | no_uAUG | 40S | 7557 | -0.1840 | -0.6996 [-0.7313, -0.6701] | 1 |
| sample2019_human | rl | rep1 | random_balanced | leader | no_uAUG | seq+40S | 7557 | 0.5028 | -0.0128 [-0.0171, -0.0085] | 1 |
| sample2019_human | rl | rep1 | random_balanced | leader | no_uAUG | late_fusion | 7557 | 0.5005 | -0.0151 [-0.0194, -0.0112] | 1 |
| sample2019_human | rl | rep1 | random_balanced | leader | no_uAUG | residual | 7557 | 0.5146 | -0.0010 [-0.0046, +0.0025] | 0.731 |
| sample2019_human | rl | rep1 | random_balanced | leader | >=1_uAUG | seq | 1929 | -0.0222 |  |  |
| sample2019_human | rl | rep1 | random_balanced | leader | >=1_uAUG | 40S | 1929 | -0.3657 | -0.3435 [-0.4238, -0.2649] | 1 |
| sample2019_human | rl | rep1 | random_balanced | leader | >=1_uAUG | seq+40S | 1929 | 0.0601 | +0.0823 [+0.0634, +0.1005] | 0 |
| sample2019_human | rl | rep1 | random_balanced | leader | >=1_uAUG | late_fusion | 1929 | 0.0423 | +0.0645 [+0.0495, +0.0787] | 0 |
| sample2019_human | rl | rep1 | random_balanced | leader | >=1_uAUG | residual | 1929 | 0.0147 | +0.0370 [+0.0224, +0.0505] | 0 |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | all | seq | 9486 | 0.5542 |  |  |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | all | 40S | 9486 | 0.2192 | -0.3350 [-0.3581, -0.3133] | 1 |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | all | seq+40S | 9486 | 0.5722 | +0.0180 [+0.0128, +0.0235] | 0 |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | all | late_fusion | 9486 | 0.5666 | +0.0124 [+0.0079, +0.0167] | 0 |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | all | residual | 9486 | 0.5639 | +0.0097 [+0.0055, +0.0135] | 0 |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | no_uAUG | seq | 7557 | 0.5156 |  |  |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | no_uAUG | 40S | 7557 | -0.1840 | -0.6996 [-0.7304, -0.6705] | 1 |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | no_uAUG | seq+40S | 7557 | 0.5028 | -0.0128 [-0.0171, -0.0084] | 1 |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | no_uAUG | late_fusion | 7557 | 0.5005 | -0.0151 [-0.0192, -0.0112] | 1 |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | no_uAUG | residual | 7557 | 0.5146 | -0.0010 [-0.0045, +0.0024] | 0.7205 |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | >=1_uAUG | seq | 1929 | -0.0222 |  |  |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | >=1_uAUG | 40S | 1929 | -0.3657 | -0.3435 [-0.4252, -0.2653] | 1 |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | >=1_uAUG | seq+40S | 1929 | 0.0601 | +0.0823 [+0.0639, +0.1024] | 0 |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | >=1_uAUG | late_fusion | 1929 | 0.0423 | +0.0645 [+0.0503, +0.0798] | 0 |
| sample2019_human | rl | rep1 | random_balanced | sequence_cluster | >=1_uAUG | residual | 1929 | 0.0147 | +0.0370 [+0.0230, +0.0511] | 0 |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | all | seq | 9486 | 0.5538 |  |  |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | all | 40S | 9486 | 0.2191 | -0.3346 [-0.3576, -0.3130] | 1 |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | all | seq+40S | 9486 | 0.5715 | +0.0177 [+0.0125, +0.0231] | 0 |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | all | late_fusion | 9486 | 0.5659 | +0.0121 [+0.0077, +0.0164] | 0 |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | all | residual | 9486 | 0.5634 | +0.0096 [+0.0054, +0.0134] | 0 |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | no_uAUG | seq | 7557 | 0.5141 |  |  |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | no_uAUG | 40S | 7557 | -0.1839 | -0.6980 [-0.7285, -0.6683] | 1 |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | no_uAUG | seq+40S | 7557 | 0.5010 | -0.0131 [-0.0175, -0.0087] | 1 |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | no_uAUG | late_fusion | 7557 | 0.4993 | -0.0148 [-0.0188, -0.0108] | 1 |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | no_uAUG | residual | 7557 | 0.5134 | -0.0007 [-0.0042, +0.0027] | 0.656 |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | >=1_uAUG | seq | 1929 | -0.0218 |  |  |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | >=1_uAUG | 40S | 1929 | -0.3662 | -0.3444 [-0.4250, -0.2667] | 1 |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | >=1_uAUG | seq+40S | 1929 | 0.0597 | +0.0815 [+0.0631, +0.1014] | 0 |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | >=1_uAUG | late_fusion | 1929 | 0.0414 | +0.0632 [+0.0491, +0.0781] | 0 |
| sample2019_human | rl | rep1 | sequence_cluster | sequence_cluster | >=1_uAUG | residual | 1929 | 0.0144 | +0.0362 [+0.0224, +0.0503] | 0 |
| sample2019_human | rl | rep2 | random_balanced | leader | all | seq | 9486 | 0.5542 |  |  |
| sample2019_human | rl | rep2 | random_balanced | leader | all | 40S | 9486 | 0.1881 | -0.3661 [-0.3875, -0.3446] | 1 |
| sample2019_human | rl | rep2 | random_balanced | leader | all | seq+40S | 9486 | 0.5691 | +0.0149 [+0.0102, +0.0197] | 0 |
| sample2019_human | rl | rep2 | random_balanced | leader | all | late_fusion | 9486 | 0.5637 | +0.0095 [+0.0058, +0.0133] | 0 |
| sample2019_human | rl | rep2 | random_balanced | leader | all | residual | 9486 | 0.5622 | +0.0080 [+0.0044, +0.0116] | 0 |
| sample2019_human | rl | rep2 | random_balanced | leader | no_uAUG | seq | 7557 | 0.5156 |  |  |
| sample2019_human | rl | rep2 | random_balanced | leader | no_uAUG | 40S | 7557 | -0.2196 | -0.7352 [-0.7684, -0.7046] | 1 |
| sample2019_human | rl | rep2 | random_balanced | leader | no_uAUG | seq+40S | 7557 | 0.5041 | -0.0116 [-0.0158, -0.0073] | 1 |
| sample2019_human | rl | rep2 | random_balanced | leader | no_uAUG | late_fusion | 7557 | 0.5021 | -0.0135 [-0.0173, -0.0100] | 1 |
| sample2019_human | rl | rep2 | random_balanced | leader | no_uAUG | residual | 7557 | 0.5146 | -0.0011 [-0.0044, +0.0022] | 0.7495 |
| sample2019_human | rl | rep2 | random_balanced | leader | >=1_uAUG | seq | 1929 | -0.0222 |  |  |
| sample2019_human | rl | rep2 | random_balanced | leader | >=1_uAUG | 40S | 1929 | -0.4350 | -0.4128 [-0.4944, -0.3326] | 1 |
| sample2019_human | rl | rep2 | random_balanced | leader | >=1_uAUG | seq+40S | 1929 | 0.0471 | +0.0693 [+0.0526, +0.0856] | 0 |
| sample2019_human | rl | rep2 | random_balanced | leader | >=1_uAUG | late_fusion | 1929 | 0.0297 | +0.0519 [+0.0393, +0.0640] | 0 |
| sample2019_human | rl | rep2 | random_balanced | leader | >=1_uAUG | residual | 1929 | 0.0085 | +0.0307 [+0.0180, +0.0426] | 0 |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | all | seq | 9486 | 0.5542 |  |  |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | all | 40S | 9486 | 0.1881 | -0.3661 [-0.3890, -0.3449] | 1 |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | all | seq+40S | 9486 | 0.5691 | +0.0149 [+0.0103, +0.0195] | 0 |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | all | late_fusion | 9486 | 0.5637 | +0.0095 [+0.0057, +0.0131] | 0 |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | all | residual | 9486 | 0.5622 | +0.0080 [+0.0044, +0.0113] | 0 |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | no_uAUG | seq | 7557 | 0.5156 |  |  |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | no_uAUG | 40S | 7557 | -0.2196 | -0.7352 [-0.7674, -0.7041] | 1 |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | no_uAUG | seq+40S | 7557 | 0.5041 | -0.0116 [-0.0158, -0.0073] | 1 |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | no_uAUG | late_fusion | 7557 | 0.5021 | -0.0135 [-0.0172, -0.0099] | 1 |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | no_uAUG | residual | 7557 | 0.5146 | -0.0011 [-0.0044, +0.0020] | 0.7375 |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | >=1_uAUG | seq | 1929 | -0.0222 |  |  |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | >=1_uAUG | 40S | 1929 | -0.4350 | -0.4128 [-0.4971, -0.3324] | 1 |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | >=1_uAUG | seq+40S | 1929 | 0.0471 | +0.0693 [+0.0532, +0.0865] | 0 |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | >=1_uAUG | late_fusion | 1929 | 0.0297 | +0.0519 [+0.0400, +0.0651] | 0 |
| sample2019_human | rl | rep2 | random_balanced | sequence_cluster | >=1_uAUG | residual | 1929 | 0.0085 | +0.0307 [+0.0186, +0.0429] | 0 |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | all | seq | 9486 | 0.5538 |  |  |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | all | 40S | 9486 | 0.1877 | -0.3660 [-0.3888, -0.3445] | 1 |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | all | seq+40S | 9486 | 0.5684 | +0.0146 [+0.0100, +0.0193] | 0 |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | all | late_fusion | 9486 | 0.5631 | +0.0093 [+0.0054, +0.0129] | 0 |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | all | residual | 9486 | 0.5617 | +0.0079 [+0.0043, +0.0112] | 0 |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | no_uAUG | seq | 7557 | 0.5141 |  |  |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | no_uAUG | 40S | 7557 | -0.2199 | -0.7340 [-0.7663, -0.7025] | 1 |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | no_uAUG | seq+40S | 7557 | 0.5022 | -0.0119 [-0.0162, -0.0076] | 1 |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | no_uAUG | late_fusion | 7557 | 0.5009 | -0.0133 [-0.0169, -0.0097] | 1 |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | no_uAUG | residual | 7557 | 0.5132 | -0.0009 [-0.0042, +0.0023] | 0.712 |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | >=1_uAUG | seq | 1929 | -0.0218 |  |  |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | >=1_uAUG | 40S | 1929 | -0.4358 | -0.4140 [-0.4981, -0.3336] | 1 |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | >=1_uAUG | seq+40S | 1929 | 0.0467 | +0.0685 [+0.0524, +0.0858] | 0 |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | >=1_uAUG | late_fusion | 1929 | 0.0290 | +0.0508 [+0.0388, +0.0635] | 0 |
| sample2019_human | rl | rep2 | sequence_cluster | sequence_cluster | >=1_uAUG | residual | 1929 | 0.0083 | +0.0301 [+0.0182, +0.0423] | 0 |
| sample2019_human | rl | mean | random_balanced | leader | all | seq | 9486 | 0.5542 |  |  |
| sample2019_human | rl | mean | random_balanced | leader | all | 40S | 9486 | 0.2047 | -0.3495 [-0.3706, -0.3280] | 1 |
| sample2019_human | rl | mean | random_balanced | leader | all | seq+40S | 9486 | 0.5708 | +0.0166 [+0.0116, +0.0218] | 0 |
| sample2019_human | rl | mean | random_balanced | leader | all | late_fusion | 9486 | 0.5652 | +0.0110 [+0.0070, +0.0152] | 0 |
| sample2019_human | rl | mean | random_balanced | leader | all | residual | 9486 | 0.5631 | +0.0089 [+0.0050, +0.0129] | 0 |
| sample2019_human | rl | mean | random_balanced | leader | no_uAUG | seq | 7557 | 0.5156 |  |  |
| sample2019_human | rl | mean | random_balanced | leader | no_uAUG | 40S | 7557 | -0.2011 | -0.7168 [-0.7493, -0.6866] | 1 |
| sample2019_human | rl | mean | random_balanced | leader | no_uAUG | seq+40S | 7557 | 0.5034 | -0.0122 [-0.0165, -0.0079] | 1 |
| sample2019_human | rl | mean | random_balanced | leader | no_uAUG | late_fusion | 7557 | 0.5013 | -0.0144 [-0.0184, -0.0106] | 1 |
| sample2019_human | rl | mean | random_balanced | leader | no_uAUG | residual | 7557 | 0.5146 | -0.0010 [-0.0045, +0.0024] | 0.732 |
| sample2019_human | rl | mean | random_balanced | leader | >=1_uAUG | seq | 1929 | -0.0222 |  |  |
| sample2019_human | rl | mean | random_balanced | leader | >=1_uAUG | 40S | 1929 | -0.3973 | -0.3750 [-0.4566, -0.2946] | 1 |
| sample2019_human | rl | mean | random_balanced | leader | >=1_uAUG | seq+40S | 1929 | 0.0540 | +0.0762 [+0.0584, +0.0935] | 0 |
| sample2019_human | rl | mean | random_balanced | leader | >=1_uAUG | late_fusion | 1929 | 0.0363 | +0.0586 [+0.0447, +0.0719] | 0 |
| sample2019_human | rl | mean | random_balanced | leader | >=1_uAUG | residual | 1929 | 0.0119 | +0.0341 [+0.0204, +0.0470] | 0 |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | all | seq | 9486 | 0.5538 |  |  |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | all | 40S | 9486 | 0.2045 | -0.3493 [-0.3721, -0.3277] | 1 |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | all | seq+40S | 9486 | 0.5700 | +0.0162 [+0.0113, +0.0213] | 0 |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | all | late_fusion | 9486 | 0.5646 | +0.0108 [+0.0067, +0.0147] | 0 |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | all | residual | 9486 | 0.5626 | +0.0088 [+0.0049, +0.0125] | 0 |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | no_uAUG | seq | 7557 | 0.5141 |  |  |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | no_uAUG | 40S | 7557 | -0.2012 | -0.7153 [-0.7465, -0.6847] | 1 |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | no_uAUG | seq+40S | 7557 | 0.5016 | -0.0125 [-0.0170, -0.0082] | 1 |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | no_uAUG | late_fusion | 7557 | 0.5001 | -0.0140 [-0.0179, -0.0102] | 1 |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | no_uAUG | residual | 7557 | 0.5133 | -0.0008 [-0.0042, +0.0026] | 0.6685 |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | >=1_uAUG | seq | 1929 | -0.0218 |  |  |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | >=1_uAUG | 40S | 1929 | -0.3979 | -0.3761 [-0.4598, -0.2968] | 1 |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | >=1_uAUG | seq+40S | 1929 | 0.0536 | +0.0754 [+0.0581, +0.0942] | 0 |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | >=1_uAUG | late_fusion | 1929 | 0.0355 | +0.0574 [+0.0443, +0.0711] | 0 |
| sample2019_human | rl | mean | sequence_cluster | sequence_cluster | >=1_uAUG | residual | 1929 | 0.0116 | +0.0334 [+0.0204, +0.0467] | 0 |

Validation checks:

- [x] no reporter label used in 40S model selection
- [x] yeast 40S fit excludes the reporter's chromosome fold
- [x] Akirtava TEff primary rows satisfy combined PoLib depth >= 5000
- [x] no outer-test label used (yeast TEff)
- [x] scaling uses training rows only (yeast TEff)
- [x] no outer-test label used (yeast MeanYFP)
- [x] scaling uses training rows only (yeast MeanYFP)
- [x] bootstrap unit for Akirtava is the gene/ORF
- [x] human 40S fitted separately on rep1 and rep2
- [x] identical rows receive deterministic fold assignments
- [x] no outer-test label used (human, 40S rep1)
- [x] scaling uses training rows only (human, 40S rep1)
- [x] no outer-test label used (human, 40S rep2)
- [x] scaling uses training rows only (human, 40S rep2)
- [x] no outer-test label used (human, 40S mean)
- [x] scaling uses training rows only (human, 40S mean)
- [x] residual training target uses inner out-of-fold sequence predictions
- [x] late-fusion stacker uses inner out-of-fold predictions
- [x] all prediction rows reproduce the summary R2
