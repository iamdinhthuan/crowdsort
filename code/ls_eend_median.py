"""LS-EEND decoding of the authors' scoring recipe (metrics.py): threshold 0.5, then a median filter of 11 frames
of 100 ms on the decisions. A median filter commutes with thresholding, so it is applied to the 100 ms
probabilities and the result is scored with the usual threshold decoding. The centred window adds 0.5 s of
look-ahead, so the latency becomes 1.5 s; outputs go to exp/ls_eend_dih3_med11/<set>/1.5.

python ls_eend_median.py msdwild_many,msdwild_few,...
"""
import glob
import os
import sys

import numpy as np
from scipy.signal import medfilt

ROOT = "/mnt/ssd2/paper/paper_1"
for s in sys.argv[1].split(","):
    src, dst = f"{ROOT}/exp/ls_eend_dih3/{s}/1.0", f"{ROOT}/exp/ls_eend_dih3_med11/{s}/1.5"
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(f"{src}/*.npz")):
        d = np.load(f)
        n10 = len(d["probs"])
        p = d["p100"].astype(np.float32) if "p100" in d else d["probs"][::10].astype(np.float32)
        q = np.stack([medfilt(p[:, j], 11) for j in range(p.shape[1])], axis=1)
        q10 = np.repeat(q, 10, axis=0)[:n10]
        if len(q10) < n10:
            q10 = np.pad(q10, ((0, n10 - len(q10)), (0, 0)))
        np.savez_compressed(f"{dst}/{os.path.basename(f)}", probs=q10.astype(np.float16), act=d["act"])
        n += 1
    print("median", s, n)
