"""Tiny sequential GPU job queue. Jobs are shell files in queue/pending/*.sh, run in name order.
A finished job moves to queue/done/ (exit 0) or queue/failed/, with its log in queue/logs/<name>.log.
Run: nohup python gpuq.py &   (stops when queue/STOP exists)"""
import os
import subprocess
import time

Q = "/mnt/ssd2/paper/paper_1/queue"
for d in ("pending", "done", "failed", "logs", "running"):
    os.makedirs(os.path.join(Q, d), exist_ok=True)

while not os.path.exists(os.path.join(Q, "STOP")):
    jobs = sorted(f for f in os.listdir(os.path.join(Q, "pending")) if f.endswith(".sh"))
    if not jobs:
        time.sleep(5)
        continue
    j = jobs[0]
    src = os.path.join(Q, "pending", j)
    run = os.path.join(Q, "running", j)
    os.rename(src, run)
    t0 = time.time()
    with open(os.path.join(Q, "logs", j[:-3] + ".log"), "w") as log:
        rc = subprocess.call(["bash", run], stdout=log, stderr=subprocess.STDOUT, cwd="/mnt/ssd2/paper/paper_1")
    dst = os.path.join(Q, "done" if rc == 0 else "failed", j)
    os.rename(run, dst)
    with open(os.path.join(Q, "history.txt"), "a") as h:
        h.write(f"{time.strftime('%H:%M:%S')} {j} rc={rc} {time.time() - t0:.0f}s\n")
