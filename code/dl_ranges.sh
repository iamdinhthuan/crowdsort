#!/usr/bin/env bash
# parallel range download: dl_ranges.sh URL OUT SIZE NPARTS
URL=$1; OUT=$2; SIZE=$3; N=$4
mkdir -p "$OUT.parts"
PART=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*PART)); e=$(( s+PART-1 )); [ $e -ge $SIZE ] && e=$((SIZE-1))
  echo "$i $s $e"
done | xargs -P $N -n 3 sh -c 'f="'$OUT'.parts/$0"; want=$(($2-$1+1)); for t in 1 2 3 4 5 6 7 8; do have=$(stat -c %s "$f" 2>/dev/null || echo 0); [ "$have" -ge "$want" ] && exit 0; curl -sSL -m 3600 -r $(($1+have))-$2 "'$URL'" >> "$f"; done; echo PARTFAIL $0'
cat $(for i in $(seq 0 $((N-1))); do echo "$OUT.parts/$i"; done) > "$OUT" && [ $(stat -c %s "$OUT") -eq $SIZE ] && rm -rf "$OUT.parts" && echo RANGE_DONE "$OUT"
