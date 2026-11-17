# Post-hoc overlap check (17/11/2026), code/speaker_overlap.py

- Audio duplicates: MSDWild few contains 5 clips identical in audio to training clips (00844, 00991 byte-identical;
  01731, 01772, 01070 identical in every 1-s window) and one partial (00535, 5 s). RTTMs were annotated separately.
  MSDWild many: no near-duplicate (longest matching run 2 s). Removing the 6 few clips lowers every system's few DER by
  0.08-0.10 points; no ranking changes.
- Speaker matches (non-overlapped speech >=3 s; ECAPA + WavLM-SV; thresholds = 99.9/99th pct of within-clip
  different-speaker pairs). Assessable: train 4615/5639, few 1003/1274, many 486/1031.
  Match rate of test speakers against MSDWild train (N=1852 subsample, 10 draws) vs external null pool
  (VoxConverse test + AMI train, N=1852):
    many strict 5.9% vs 0.0; moderate 12.4% vs 0.0; lenient 42.9% vs 5.8
    few  strict 18.8% vs 0.1; moderate 34.7% vs 0.6; lenient 65.6% vs 12.8
  Against the full train pool: many 8.6 / 17.9 / 54.9 %, few 26.8 / 43.6 / 76.2 %.
  => strict/moderate matches are real shared speakers.
- MSDWild many without flagged recordings (CrowdSort vs FT-clips / SimConv-MSD-M / N3D, pooled DER points):
    strict (142 rec)  -3.34 / -4.11 / -6.14
    moderate (108)    -1.06 / -2.58 / -3.03
    lenient (39)      -1.84 / -2.82 / -4.40
- Duration: the 5 recordings over 10 min (02341, 00884, 01065, 00206, 01151) hold 27.7% of the reference speech.
  Without them (172 rec): CrowdSort vs N3D -3.29 [-4.63,-1.97]; vs FT-clips -1.02 [-1.94,-0.06] p=0.019;
  vs SimConv-MSD -2.46; vs SimConv-MSD-M -2.34; vs FastMSS -1.67 (all p<0.001 except FT-clips).
  Flagged and clean recordings below 120 s show similar gains; the shrinkage on the "clean" subsets follows from the
  long recordings being flagged more often.
