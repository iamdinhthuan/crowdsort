#!/usr/bin/env bash
CROWDSORT_ROOT=${CROWDSORT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
# enqueue_run.sh PRIO NAME SFX_JSON CR(0/1) SEED [STEPS]
PRIO=$1; NAME=$2; SFX=$3; CR=$4; SEED=$5; STEPS=${6:-6000}; MIX=${7:-train_mix_v2}
CRF=""; [ "$CR" = "1" ] && CRF="--cache-rescore"
TEST=ami_test_mhm,ami_test_sdm,ali_test_near,ali_test_far,aishell4_test,msdwild_many,msdwild_few
DEV=ami_dev_mhm,ami_dev_sdm,ali_eval_near,ali_eval_far,aishell4_dev,msdwild_fewdev,msdwild_concatdev
cat > ${CROWDSORT_ROOT}/queue/pending/${PRIO}_${NAME}.sh <<EOS
export PYTHONPATH=code
[ -f exp/$NAME/final.nemo ] || ${PYTHON:-python} code/train_sfx.py --name $NAME --sfx '$SFX' $CRF --train data/manifests/$MIX.jsonl --steps $STEPS --bs 12 --lr 3e-5 --warmup 300 --seed $SEED --save-every 100000 --workers 10
${PYTHON:-python} code/evalsuite.py --model exp/$NAME/final.nemo --name $NAME --sets $DEV --profiles 1.04 --decodes thr,acd
${PYTHON:-python} code/evalsuite.py --model exp/$NAME/final.nemo --name $NAME --sets $TEST --profiles 1.04 --decodes thr,acd
EOS
echo queued ${PRIO}_${NAME}
