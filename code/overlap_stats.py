"""Speech and overlap proportions of a manifest from its RTTMs (10 ms grid).

python overlap_stats.py set1 set2 ...
"""
import json
import sys

import numpy as np

ROOT = "/mnt/ssd2/paper/paper_1"
for name in sys.argv[1:]:
    sp = ov = tot = 0
    for line in open(f"{ROOT}/data/manifests/{name}.jsonl"):
        d = json.loads(line)
        n = int(d["duration"] * 100) + 1
        c = np.zeros(n, int)
        for r in open(d["rttm_filepath"]):
            p = r.split()
            s = int(float(p[3]) * 100)
            c[s:s + int(float(p[4]) * 100)] += 1
        sp += (c > 0).sum(); ov += (c > 1).sum(); tot += n
    print(f"{name:22s} speech {100 * sp / tot:5.1f}%  overlap/speech {100 * ov / sp:5.1f}%")
