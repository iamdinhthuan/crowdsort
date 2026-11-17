#!/usr/bin/env bash
# Download public diarization corpora used in the paper. Resumable (curl -C -).
set -u
ROOT=/mnt/ssd2/paper/paper_1/data/raw
dl() { # url out
  mkdir -p "$(dirname "$2")"
  for i in 1 2 3 4 5; do
    curl -sSL -C - --retry 5 --retry-delay 5 -o "$2" "$1" && return 0
    echo "retry $i $1" >&2; sleep 10
  done
  echo "FAILED $1" >&2; return 1
}
case "$1" in
ami)
  L=/mnt/ssd2/paper/paper_1/data/refs/AMI-diarization-setup/lists
  for m in $(cat $L/test.meetings.txt $L/dev.meetings.txt $L/train.meetings.txt); do
    for s in Mix-Headset Array1-01; do
      f=$ROOT/ami/$s/$m.$s.wav
      [ -s "$f" ] && continue
      dl "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$m/audio/$m.$s.wav" "$f.part" && mv "$f.part" "$f"
    done
  done ;;
alimeeting)
  for t in Eval_Ali; do
    f=$ROOT/alimeeting/$t.tar.gz; [ -f $ROOT/alimeeting/$t.done ] && continue
    dl "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/$t.tar.gz" "$f" \
      && (cd $ROOT/alimeeting && tar -xzf $t.tar.gz && rm $t.tar.gz && touch $t.done)
  done ;;
aishell4)
  for t in test train_L train_S train_M; do
    f=$ROOT/aishell4/$t.tar.gz; [ -f $ROOT/aishell4/$t.done ] && continue
    dl "https://www.openslr.org/resources/111/$t.tar.gz" "$f" \
      && (cd $ROOT/aishell4 && tar -xzf $t.tar.gz && rm $t.tar.gz && touch $t.done)
  done ;;
esac
echo "DONE $1"
