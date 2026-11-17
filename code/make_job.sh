#!/bin/bash
# write a queue job: train (if needed) + full evaluation (dev, test, extra sets, AVA-AVD and NOTSOFAR-1 at 1.04 and 30.4 s)
# usage: code/make_job.sh JOBID NAME TRAIN_MANIFEST SEED
J=$1; N=$2; T=$3; S=$4
cat > /mnt/ssd2/paper/paper_1/queue/pending/${J}_${N}.sh <<EOT
export PYTHONPATH=code
[ -f exp/$N/final.nemo ] || .venv-sfx/bin/python code/train_sfx.py --name $N --sfx '{}'  --train data/manifests/$T --steps 6000 --bs 12 --lr 3e-5 --warmup 300 --seed $S --save-every 100000 --workers 10
.venv-sfx/bin/python code/evalsuite.py --model exp/$N/final.nemo --name $N --sets ami_dev_mhm,ami_dev_sdm,ali_eval_near,ali_eval_far,aishell4_dev,msdwild_fewdev,msdwild_concatdev --profiles 1.04 --decodes thr,acd
.venv-sfx/bin/python code/evalsuite.py --model exp/$N/final.nemo --name $N --sets ami_test_mhm,ami_test_sdm,ali_test_near,ali_test_far,aishell4_test,msdwild_many,msdwild_few --profiles 1.04 --decodes thr,acd
.venv-sfx/bin/python code/evalsuite.py --model exp/$N/final.nemo --name $N --sets msdwild_interdev,voxconverse_test --profiles 1.04 --decodes thr
.venv-sfx/bin/python code/evalsuite.py --model exp/$N/final.nemo --name $N --sets avaavd_test,notsofar_eval_sc,notsofar_eval_sc_fa --profiles 1.04,30.4 --decodes thr
EOT
echo queued ${J}_${N}
