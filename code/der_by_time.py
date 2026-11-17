"""DER as a function of position in the recording (does long-term memory help later in a session?).

python der_by_time.py --model ft_base_s1 --sets aishell4_test,ami_test_sdm --profile 1.04 --edges 0,300,900,1800,99999
"""
import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.filterwarnings("ignore")
ROOT = "/mnt/ssd2/paper/paper_1"


def one(job):
    import warnings

    warnings.filterwarnings("ignore")
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    from score import read_rttm, to_annotation

    item, pdir, edges = job
    uid = os.path.basename(item["audio_filepath"])[:-4]
    d = np.load(os.path.join(pdir, uid + ".npz"))
    hyp = to_annotation(d["probs"].astype(np.float32) > 0.5, uid)
    ref = Annotation(uri=uid)
    for s, e, k in read_rttm(item["rttm_filepath"]):
        ref[Segment(s, e)] = k
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        b = min(b, item["duration"])
        if b <= a:
            out.append(None)
            continue
        m = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        r = m(ref, hyp, detailed=True, uem=Timeline([Segment(a, b)]))
        out.append((r["total"], r["missed detection"] + r["false alarm"] + r["confusion"], r["confusion"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--profile", default="1.04")
    ap.add_argument("--edges", default="0,300,900,1800,99999")
    a = ap.parse_args()
    edges = [float(x) for x in a.edges.split(",")]
    jobs = []
    for s in a.sets.split(","):
        for l in open(f"{ROOT}/data/manifests/{s}.jsonl"):
            jobs.append((json.loads(l), f"{ROOT}/exp/{a.model}/{s}/{a.profile}", edges))
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(one, jobs))
    rows = []
    for w in range(len(edges) - 1):
        tot = sum(r[w][0] for r in res if r[w])
        err = sum(r[w][1] for r in res if r[w])
        conf = sum(r[w][2] for r in res if r[w])
        rows.append({"window": [edges[w], edges[w + 1]], "DER": round(100 * err / tot, 2) if tot else None,
                     "conf": round(100 * conf / tot, 2) if tot else None, "speech_h": round(tot / 3600, 2)})
    out = {"model": a.model, "sets": a.sets, "profile": a.profile, "rows": rows}
    print(json.dumps(out))
    with open(f"{ROOT}/results/der_by_time.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
