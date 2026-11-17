#!/bin/bash
# wait_job.sh JOBNAME MODEL [MODEL...]: wait until queue/done|failed/JOBNAME.sh exists, then compare the models.
cd /mnt/ssd2/paper/paper_1
J=$1; shift
until [ -f queue/done/$J.sh ] || [ -f queue/failed/$J.sh ]; do sleep 60; done
ls queue/done/$J.sh queue/failed/$J.sh 2>/dev/null
python3 code/compare.py "$@"
