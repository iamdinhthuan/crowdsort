#!/bin/bash
# LS-EEND (DIHARD III checkpoint) on all test sets, scored with the same pipeline (results.jsonl model ls_eend_dih3)
cd /mnt/ssd2/paper/paper_1
CK=vendor/FS-EEND/LS-EEND/ckpt_dl/dih3.ckpt
CF=vendor/FS-EEND/LS-EEND/conf/spk_onl_conformer_retention_enc_dec_nonautoreg_dihard3_infer.yaml
SETS="msdwild_many msdwild_few avaavd_test ami_test_mhm ami_test_sdm ali_test_near ali_test_far aishell4_test"
for s in $SETS; do
  (cd vendor/FS-EEND/LS-EEND && ../../../.venv-sfx/bin/python ../../../code/ls_eend_infer.py --ckpt ../../../$CK --config ../../../$CF \
     --manifest ../../../data/manifests/$s.jsonl --out ../../../exp/ls_eend_dih3/$s/1.0) 2>&1 | grep -v Warning | tail -1
done
.venv-sfx/bin/python code/rescore.py --label thr --decode thr --models ls_eend_dih3 --sets $(echo $SETS | tr ' ' ',') --profile 1.0
echo LS_EEND_DONE
