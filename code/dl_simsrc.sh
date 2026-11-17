#!/bin/bash
# Sources for the conventional simulation baseline: RIRS_NOISES (OpenSLR 28) and LibriSpeech
# train-clean-100 (OpenSLR 12), from Hugging Face mirrors whose archives have the official byte sizes
# (1311166223 and 6387309499). Archives are deleted after a successful extraction.
set -e
R=/mnt/ssd2/paper/paper_1
PY="$R/.venv-sfx/bin/python -I"
cd $R/data/raw
$PY $R/code/dl_hf.py EaseZh/rirs_noises rirs rirs_noises.zip
[ "$(stat -c %s rirs/rirs_noises.zip)" = 1311166223 ]
(cd rirs && unzip -q -o rirs_noises.zip && rm rirs_noises.zip && echo RIRS_OK)
$PY $R/code/dl_hf.py k2-fsa/LibriSpeech librispeech train-clean-100.tar.gz
[ "$(stat -c %s librispeech/train-clean-100.tar.gz)" = 6387309499 ]
(cd librispeech && tar xzf train-clean-100.tar.gz && rm train-clean-100.tar.gz && echo LIBRI_OK)
