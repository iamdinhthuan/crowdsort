#!/usr/bin/env bash
CROWDSORT_ROOT=${CROWDSORT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
# Stream-extract AliMeeting Train_Ali_far: keep only channel 0 of each 8-channel wav (16 kHz mono), no temp copies.
set -e
OUT=${CROWDSORT_ROOT}/data/wav/ali_train_far_ch0
mkdir -p $OUT
export OUT
tar -xzf ${CROWDSORT_ROOT}/data/raw/alimeeting_far/Train_Ali_far.tar.gz --wildcards '*.wav' \
  --to-command='ffmpeg -nostdin -loglevel error -y -f wav -i - -af "pan=mono|c0=c0" -ar 16000 -c:a pcm_s16le "$OUT/$(basename "$TAR_FILENAME")"'
echo EXTRACT_FAR_DONE $(ls $OUT | wc -l)
