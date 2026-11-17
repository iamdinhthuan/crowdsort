#!/usr/bin/env bash
m=$1; s=$2; f=/mnt/ssd2/paper/paper_1/data/raw/ami/$s/$m.$s.wav
[ -s "$f" ] && exit 0
mkdir -p "$(dirname $f)"
for i in 1 2 3 4 5 6; do
  curl -sSL -C - --retry 5 -o "$f.part" "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$m/audio/$m.$s.wav" && mv "$f.part" "$f" && exit 0
  sleep 5
done
echo "FAILED $m $s"
