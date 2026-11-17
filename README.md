# CrowdSort

Code, data lists and per-recording results for the paper

> D. T. Nguyen, L. P. Nguyen, V. H. Nguyen, M. R. Elara and A. V. Le. *CrowdSort: Many-speaker conversation synthesis for streaming Sortformer diarization in the wild.*
> Submitted to Computer Speech & Language, 2026.

CrowdSort fine-tunes the released streaming diarization model
[Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) (N3D) with crowded sessions built
from real multi-speaker units of annotated MSDWild clips. An arrival operator (concatenation, `code/make_concat.py`)
lets new speakers enter after the speaker cache is full, and a return operator (interleaving,
`code/make_interleave.py`) lets speakers leave and return. CrowdSort-P adds a boundary perturbation
(`--perturb`, `code/perturb.py`). The architecture is unchanged, so inference cost equals that of the released model.

Main results (DER %, 1.04 s latency, collar 0, overlap scored; three seeds averaged per recording where marked):

| System | MSDWild many | Pooled, 7 test sets | AVA-AVD | NOTSOFAR-1 (FA refs) | NOTSOFAR-1 (word refs) |
|---|---:|---:|---:|---:|---:|
| N3D (released model) | 41.94 | 13.49 | 48.25 | 12.91 | 17.05 |
| FT-clips (3 seeds) | 39.09 | 13.01 | 48.81 | 16.82 | 16.04 |
| SimConv-MSD | 40.12 | 13.27 | 48.30 | 16.75 | 16.67 |
| FastMSS | 38.62 | 12.90 | 47.36 | 17.15 | 15.88 |
| CrowdSort-C, arrival only (3 seeds) | 34.74 | 12.68 | 46.66 | 16.82 | 15.32 |
| **CrowdSort** (3 seeds) | **33.81** | **12.64** | 46.44 | 16.29 | 15.33 |
| CrowdSort-P (3 seeds) | 34.94 | 12.70 | **45.76** | 16.37 | 15.19 |

On NOTSOFAR-1, whose training data the released model used, every fine-tuned system is worse than the released
model with the forced-alignment references of its published evaluation; the comparisons between fine-tuned systems
agree under both references (see the paper, Section 5.4).

## Layout

| Path | Content |
|---|---|
| `code/` | Data preparation, synthesis, training (`train_sfx.py`, `sfx/`), streaming inference, scoring, bootstrap, tables (`make_tables.py`, `make_tables2.py`) and figures |
| `code/fastmss_cfg/` | FastMSS configuration of the simulator baseline |
| `data_lists/*.txt` | Recording IDs of the held-out development splits and training subsets |
| `data_lists/synth_rttm/` | Reference RTTMs of every synthesised or simulated session (speaker labels carry the source clip ID) |
| `results/results.jsonl` | One line per system, test set, latency and decoding, with DER, its components, SCA and MAE |
| `exp/<model>/<set>/<latency>/_perfile_<decoding>.json` | Per-recording errors of every evaluated system |
| `REPRODUCE.md` | Environment, data preparation and every command used for the paper |

Model names: `base` N3D; `c3_nomsd` FT-meet; `c3_noconcat*` FT-clips; `ft3_s*` CrowdSort-C; `c3_inter*`
CrowdSort-I; `c3_ci_s*` CrowdSort; `c3_cip_s*` CrowdSort-P; `c3_simconvmsd_s1` SimConv-MSD; `c3_fastmss_s1`
FastMSS; `c3_concatu_s1`, `c3_concat25_s1` arrival-only controls; `ls_eend_dih3` LS-EEND (released DIHARD III
checkpoint, not fine-tuned). NOTSOFAR-1 sets: `notsofar_eval_sc` (word timings) and `notsofar_eval_sc_fa`
(forced-alignment references of FastMSS).

Scripts set `ROOT = "/mnt/ssd2/paper/paper_1"`; change it to the repository location. Fine-tuned checkpoints are
available from the corresponding author on request. All corpora are public and must be obtained from their
original sources.
