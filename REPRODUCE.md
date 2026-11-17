# CrowdSort: many-speaker conversation synthesis for streaming Sortformer diarization

Code, data recipes and results of the paper in `paper/tex/main.tex`.

## Environment

- `.venv-sfx`: conda Python 3.11 (torch 2.8 cu128) with NeMo Speech at commit `a039a6c` (copied into
  `site-packages/nemo`) and lhotse 2.0.0a6.
- `code/sfx/__init__.py` routes the NeMo Transformer encoder through its dense-mask SDPA path. The compiled
  FlexAttention kernel needs 128 KB of shared memory, which consumer Blackwell GPUs such as the RTX 5090 do not have.
- Released model: `models/base/Nemotron-3-Diarization.nemo` (Hugging Face `nvidia/Nemotron-3-Diarization`, OpenMDW-1.1).

## Data

| Step | Command |
|---|---|
| Download | `code/download_data.sh`, `code/dl_hf.py`, `code/dl_ranges.sh` (see `NOTES.md` for the sources) |
| Prepare wav, RTTM, manifests | `code/prep_data.py {ami,alimeeting,aishell4,msdwild}`, `code/prep_ali_far.py` |
| Synthesis by concatenation | `code/make_concat.py msdwild_train_v2 msdwild_concat_v2 800 7` |
| Synthesis by interleaving | `code/make_interleave.py msdwild_train_v2 msdwild_inter_v2 800 7` |
| Training mixes | `code/make_mix.py` (`data/manifests/train_mix_v3*.jsonl`) |

## Training and evaluation

| Step | Command |
|---|---|
| Fine-tune | `code/train_sfx.py --name NAME --sfx '{}' --train data/manifests/train_mix_v3ci.jsonl --steps 6000 --bs 12 --lr 3e-5 --warmup 300 --seed S` (the script defaults, 3000 steps and 200 warm-up steps, are not the paper settings) |
| Streaming inference | `code/infer.py --model exp/NAME/final.nemo --manifest M --profile {30.4,1.04,0.32}` |
| Scoring | `code/score.py` (pyannote.metrics, collar 0, overlap scored) |
| Suite | `code/evalsuite.py`, results appended to `results/results.jsonl` |
| Paired bootstrap | `code/bootstrap.py`, `code/pooled.py` |
| Tables and figures | `code/make_tables.py`, `code/fig_*.py` |

Model names in `results/results.jsonl`:

| Name | System |
|---|---|
| `base` | N3D, the released model |
| `c3_nomsd` | FT-meet |
| `c3_noconcat*` | FT-clips |
| `ft3_s*` | CrowdSort-C (arrival operator only) |
| `c3_inter*` | CrowdSort-I (return operator only) |
| `c3_ci_s*` | CrowdSort (both operators, main configuration) |
| `c3_cip_s*` | CrowdSort-P (both operators with boundary-masking perturbation) |
| `c3_simconvmsd_s1` | SimConv-MSD baseline |
| `c3_fastmss_s1` | FastMSS baseline |
| `c3_concatu_s1`, `c3_concat25_s1` | Arrival-only controls: 2386 distinct sessions; 25% synthetic share |
| `ls_eend_dih3` (profile 1.0), `ls_eend_dih3_med11` (1.5) | LS-EEND, released DIHARD III checkpoint, without / with the authors' median filter |
| `sam3A_s1`, `sam3B_s1`, `c3_hrs` | Structured memory variants |
| `base_sc*`, `*_inf396`, `*_ad*` | Inference-time memory variants |

## Revision 2: synthesis, external evaluation and statistics

```bash
# synthetic sessions (arrival / return operators; --perturb for CrowdSort-P, same units, order and gaps)
python code/make_concat.py msdwild_train_v2 msdwild_concat_v2 800 7 [--perturb]
python code/make_interleave.py msdwild_train_v2 msdwild_inter_v2 800 7 [--perturb]
# simulator baselines
python code/make_simconv.py msdwild_train_v2 msdwild_simconv 800 <seed>        # SimConv-MSD
python vendor/FastMSS/recipes/sim.py --config-path=. --config-name=crowdsort_5to8  # FastMSS (config in code/fastmss_cfg)
# training mixes with the same real data and 2388 synthetic samples
python code/make_mix_swap.py <out.jsonl> <synthetic manifests...> [--hours 26.8]
# external test set (AVA-AVD, audio from the AVA films)
python code/prep_avaavd.py
# tables, statistics, speaker-level analysis
python code/make_tables.py; python code/make_tables2.py; python code/stats_extra.py
python code/mechanism.py --sets msdwild_many,avaavd_test --systems N3D=base CrowdSort=c3_ci_s1+c3_ci_s2+c3_ci_s3
# tuned activity-guided decoding (thresholds selected on the development sets)
python code/rescore.py --label acd_tuned --decode acd --acd '{"tau_sil":0.4,"tau_lo":0.35,"tau_ov":0.35}' --models ... --sets ...
```
`bootstrap.py --group` resamples the views of one meeting together, the three clips of one AVA-AVD film together and the two single-channel devices of one NOTSOFAR-1 meeting together.

## Revision 3: held-out NOTSOFAR-1 and an external streaming baseline

```bash
# NOTSOFAR-1 evaluation set (HF microsoft/NOTSOFAR, 240629.1_eval_small_with_GT), single-channel track, word-timing references
python code/prep_notsofar.py
python code/evalsuite.py --model exp/NAME/final.nemo --name NAME --sets notsofar_eval_sc --profiles 1.04,30.4 --decodes thr
# LS-EEND (vendor/FS-EEND, commit adcdde1), released DIHARD III checkpoint; run from vendor/FS-EEND/LS-EEND
python ../../../code/ls_eend_infer.py --ckpt ckpt_dl/dih3.ckpt \
    --config conf/spk_onl_conformer_retention_enc_dec_nonautoreg_dihard3_infer.yaml \
    --manifest ../../../data/manifests/SET.jsonl --out ../../../exp/ls_eend_dih3/SET/1.0
python code/rescore.py --label thr --decode thr --models ls_eend_dih3 --sets SET --profile 1.0
python code/ls_eend_median.py SET   # authors' 11-frame median filter (not used in the paper: higher DER on 8 of 9 sets)
```
