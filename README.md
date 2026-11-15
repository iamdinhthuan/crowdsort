# CrowdSort

Code, data lists and results for the paper

> D. T. Nguyen. *CrowdSort: Many-speaker conversation synthesis for streaming Sortformer diarization in the wild.*
> Submitted to Computer Speech & Language, 2026.

CrowdSort fine-tunes the released streaming diarization model
[Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) (N3D) with synthesised
many-speaker sessions. The sessions are assembled from short in-the-wild clips by concatenation or by
interleaving. The model architecture is not changed, so inference cost is the same as for the released model.
The repository also contains the memory variants compared in the paper: cache sizes, the speaker-adaptive budget,
slot embeddings, speaker prototypes, refreshed cache scores and the 10 ms skip refinement.

Main results (DER %, 1.04 s latency, collar 0, overlap scored; mean of three seeds where marked):

| System | MSDWild many | MSDWild few | Pooled, 7 test sets |
|---|---:|---:|---:|
| N3D (released model) | 41.94 | 17.35 | 13.49 |
| FT-meet | 42.55 | 18.02 | 13.87 |
| FT-clips (3 seeds) | 39.09 | 14.92 | 13.01 |
| CrowdSort, concatenation (3 seeds) | **34.74** | 15.01 | 12.68 |
| CrowdSort, interleaving | 35.55 | 14.84 | 12.57 |

All tables of the paper can be regenerated from `results/` and `exp/` with `code/make_tables.py`.

## Repository layout

| Path | Content |
|---|---|
| `code/sfx/` | `SFXModel`, a subclass of the NeMo `SortformerEncLabelModel` with optional memory modules. With every option off it reproduces the released model exactly. |
| `code/` | Data preparation, synthesis, training, streaming inference, scoring, statistics, tables and figures |
| `data_lists/*.txt` | Recording IDs of the held-out development splits and of the training subsets |
| `data_lists/synth_rttm/` | Reference RTTMs of every synthesised session. Speaker labels carry the source clip ID (for example `00494_S1`). |
| `results/` | `results.jsonl` (one line per system, test set, latency and decoding, with DER and its components), throughput, position analysis and ACD thresholds |
| `exp/` | Per-recording scores of every evaluated system |

## Installation

Use Python 3.11 and a CUDA GPU. The paper used one RTX 5090 (32 GB).

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/NVIDIA-NeMo/Speech
git -C Speech checkout a039a6cfce00aa57a6a8cd8e96166c1379f903e8
pip install -e "Speech[asr]"
pip install -r requirements.txt
```

`code/sfx/__init__.py` makes two changes to NeMo when it is imported:

1. It routes the Transformer encoder through its SDPA path. The compiled FlexAttention kernel needs
   128 KB of shared memory per block, and consumer Blackwell GPUs such as the RTX 5090 have only 99 KB.
   The outputs are the same up to numerical precision.
2. It adds `sfx.` to the NeMo target allow-list so that `SFXModel` checkpoints can be restored.

All scripts resolve paths relative to the repository root. To use another root, set `CROWDSORT_ROOT`.
Download the released checkpoint to `models/base/Nemotron-3-Diarization.nemo`:

```bash
huggingface-cli download nvidia/Nemotron-3-Diarization --local-dir models/base
```

## Data

Every corpus is public. Run the scripts from the repository root with `PYTHONPATH=code`.

| Corpus | Source | Script |
|---|---|---|
| AMI (MHM, SDM) | Hugging Face `diarizers-community/ami`; forced-alignment RTTMs from [nttcslab-sp/diar-forced-alignment](https://github.com/nttcslab-sp/diar-forced-alignment) | `code/dl_hf.py diarizers-community/ami data/raw/ami_hf` |
| AliMeeting (near, far channel 0) | [OpenSLR 119](https://www.openslr.org/119/) | `code/download_data.sh`, `code/extract_ali_far.sh` |
| AISHELL-4 (channel 0) | [OpenSLR 111](https://www.openslr.org/111/) or Hugging Face `AISHELL/AISHELL-4` | `code/download_data.sh` or `code/dl_hf.py` |
| MSDWild | [X-LANCE/MSDWILD](https://github.com/X-LANCE/MSDWILD) (audio and RTTMs) | see the MSDWild instructions |

Then prepare wav files, RTTMs and NeMo manifests:

```bash
python code/prep_data.py ami
python code/prep_data.py alimeeting
python code/prep_ali_far.py
python code/prep_data.py aishell4
python code/prep_data.py msdwild
```

Hold out the development splits from the training data:

```bash
python code/make_splits.py
```

This keeps 10 AISHELL-4 training sessions (`aishell4_dev`) and 250 MSDWild training clips (`msdwild_fewdev`)
for development. The selection rule is in the script, and the resulting IDs are listed in `data_lists/`.

### Many-speaker synthesis

```bash
python code/make_concat.py     msdwild_train_v2 msdwild_concat_v2 800 7   # 796 sessions, 26.8 h
python code/make_interleave.py msdwild_train_v2 msdwild_inter_v2  800 7   # 800 sessions, 30.0 h
python code/make_concat.py     msdwild_fewdev   msdwild_concatdev 120 13  # development sessions
python code/make_interleave.py msdwild_fewdev   msdwild_interdev  120 13
```

The scripts are seeded, so they rebuild the sessions of the paper from the same sorted clip list. A dry run of
`make_concat.py` reproduced all 796 stored training RTTMs exactly. The rebuilt RTTMs can be checked against
`data_lists/synth_rttm/`.

### Training mixes

```bash
cd data/manifests
M="ami_train_mhm.jsonl:0.15 ami_train_sdm.jsonl:0.15 aishell4_trainsub.jsonl:0.15 ali_train_near.jsonl:0.15 ali_train_far.jsonl:0.15"
python ../../code/make_mix.py train_mix_v3.jsonl          $M msdwild_train_v2.jsonl:0.125 msdwild_concat_v2.jsonl:0.125  # CrowdSort (concatenation)
python ../../code/make_mix.py train_mix_v3i.jsonl         $M msdwild_train_v2.jsonl:0.125 msdwild_inter_v2.jsonl:0.125   # CrowdSort (interleaving)
python ../../code/make_mix.py train_mix_v3_noconcat.jsonl $M msdwild_train_v2.jsonl:0.25                                 # FT-clips
python ../../code/make_mix.py train_mix_v3_nomsd.jsonl ami_train_mhm.jsonl:0.2 ami_train_sdm.jsonl:0.2 \
    aishell4_trainsub.jsonl:0.2 ali_train_near.jsonl:0.2 ali_train_far.jsonl:0.2                            # FT-meet
cd ../..
```

## Training

```bash
export PYTHONPATH=code
python code/train_sfx.py --name ft3_s1 --sfx '{}' --train data/manifests/train_mix_v3.jsonl \
    --steps 6000 --bs 12 --lr 3e-5 --warmup 300 --seed 1 --save-every 100000 --workers 10
```

Seeds 1, 2 and 3 give `ft3_s1`, `ft3_s2` and `ft3_s3`. One run takes about 40 min on an RTX 5090.
The structured memory variants are enabled through `--sfx`, for example
`--sfx '{"slot_emb": true}'` or `--sfx '{"proto": true}'`. The options are documented in `code/sfx/model.py`.

## Evaluation

```bash
export PYTHONPATH=code
TEST=ami_test_mhm,ami_test_sdm,ali_test_near,ali_test_far,aishell4_test,msdwild_many,msdwild_few
python code/evalsuite.py --model exp/ft3_s1/final.nemo --name ft3_s1 --sets $TEST --profiles 30.4,1.04,0.32
python code/evalsuite.py --model base --name base --sets $TEST --profiles 30.4,1.04,0.32
```

`evalsuite.py` runs streaming inference (`code/infer.py`) and scoring (`code/score.py`, pyannote.metrics,
collar 0, overlap scored), and appends the results to `results/results.jsonl`.
Inference-time memory variants are passed with `--infer-args`, for example `--infer-args "--spkcache 396"`
or `--infer-args "--adaptive 50,264,396"`.
Activity-guided decoding (ACD) with the thresholds tuned on the meeting development sets
(`results/acd_base.json`) is applied at scoring time:
`python code/score.py --pred-dir exp/base/msdwild_many/1.04 --manifest data/manifests/msdwild_many.jsonl --decode acd --acd '{"tau_sil":0.4,"tau_lo":0.35,"tau_ov":0.35}'`.
The per-recording scores of the paper are stored in `exp/<system>/<set>/<latency>/_perfile_<decode>.json`,
so `code/make_tables.py` and `code/bootstrap.py` run without a GPU and without the audio.

Statistics, tables and figures:

```bash
# paired bootstrap over recordings; ',' pools sets, '+' averages the seeds of one system
python code/bootstrap.py exp/base/msdwild_many/1.04/_perfile_thr.json \
    exp/ft3_s1/msdwild_many/1.04/_perfile_thr.json+exp/ft3_s2/msdwild_many/1.04/_perfile_thr.json+exp/ft3_s3/msdwild_many/1.04/_perfile_thr.json
# -> DER 41.94 -> 34.74, delta -7.20, 95% CI [-11.66, -2.85]
python code/make_tables.py        # writes the LaTeX tables of the paper
python code/fig_nspk.py; python code/fig_position.py; python code/fig_timeline.py
python code/bench_speed.py        # throughput (RTFx)
```

System names in `results.jsonl`:

| Name | System |
|---|---|
| `base` | N3D, the released model |
| `c3_nomsd` | FT-meet |
| `c3_noconcat`, `c3_noconcat_s2`, `c3_noconcat_s3` | FT-clips |
| `ft3_s1`, `ft3_s2`, `ft3_s3` | CrowdSort with concatenation |
| `c3_inter` | CrowdSort with interleaving |
| `sam3A_s1`, `sam3B_s1`, `c3_hrs` | Structured memory variants |
| `base_sc*`, `*_inf396`, `*_ad*` | Inference-time memory variants |

## Licences

The code in this repository is released under the Apache License 2.0 (see `LICENSE`). The released
Nemotron 3 Diarization checkpoint is distributed by NVIDIA under the OpenMDW-1.1 licence. Every corpus keeps its
own licence; the synthesised RTTMs in `data_lists/synth_rttm/` are derived from the MSDWild references.

## Citation

```bibtex
@article{nguyen2026crowdsort,
  author  = {Nguyen, Dinh Thuan},
  title   = {{CrowdSort}: Many-speaker conversation synthesis for streaming {Sortformer} diarization in the wild},
  journal = {Computer Speech \& Language},
  note    = {Submitted},
  year    = {2026}
}
```
