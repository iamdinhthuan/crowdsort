#!/usr/bin/env bash
# enqueue_memtest.sh PRIO SUFFIX "INFER_ARGS" MODEL... : test (and dev) evaluation of a memory configuration on several models
PRIO=$1; SUF=$2; IA=$3; shift 3
TEST=ami_test_mhm,ami_test_sdm,ali_test_near,ali_test_far,aishell4_test,msdwild_many,msdwild_few
DEV=ami_dev_mhm,ami_dev_sdm,ali_eval_near,ali_eval_far,aishell4_dev,msdwild_fewdev,msdwild_concatdev
F=/mnt/ssd2/paper/paper_1/queue/pending/${PRIO}_mem_${SUF}.sh
: > $F
for M in "$@"; do
  if [ "$M" = "base" ]; then MA=""; NAME=base_$SUF; else MA="--model exp/$M/final.nemo"; NAME=${M}_$SUF; fi
  echo ".venv-sfx/bin/python code/evalsuite.py $MA --name $NAME --sets $DEV,$TEST --profiles 1.04 --decodes thr --infer-args=\"$IA\"" >> $F
done
echo queued $F
