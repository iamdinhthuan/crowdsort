"""Decode saved 10 ms probabilities and score DER (pyannote.metrics, collar 0, overlap scored).

python score.py --pred-dir exp/x/ami_test_mhm/1.04 --manifest data/manifests/ami_test_mhm.jsonl \
    [--decode thr|acd|oracle_count] [--thr 0.5] [--acd '{"tau_sil":0.5,"tau_lo":0.3,"tau_ov":0.3}']
Prints one JSON line; --per-file writes per-file results next to the predictions.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import warnings

warnings.filterwarnings("ignore")


def read_rttm(path):
    segs = []
    for line in open(path):
        p = line.split()
        if p and p[0] == "SPEAKER":
            segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return segs


def ref_count(segs, n):
    c = np.zeros(n + 1, dtype=np.int32)
    for s, e, _ in segs:
        a, b = int(round(s * 100)), int(round(e * 100))
        c[min(a, n) : min(b, n)] += 1
    return c[:n]


def decode(probs, act, method, thr, acd, segs=None):
    P = probs.astype(np.float32)
    on = P > thr
    if method == "thr":
        return on
    order = np.argsort(-P, axis=1)
    rows = np.arange(P.shape[0])
    if method == "oracle_count":
        n = np.minimum(ref_count(segs, P.shape[0]), P.shape[1])
        rank = np.empty_like(order)
        rank[rows[:, None], order] = np.arange(P.shape[1])[None, :]
        return rank < n[:, None]
    # activity-guided count-constrained decoding
    a = act.astype(np.float32)
    a = np.exp(a - a.max(1, keepdims=True))
    a /= a.sum(1, keepdims=True)
    nhat = a.argmax(1)
    p1 = P[rows, order[:, 0]]
    p2 = P[rows, order[:, 1]]
    out = on.copy()
    sil = (nhat == 0) & (a[:, 0] > acd.get("tau_sil", 0.5))
    out[sil] = False
    single = nhat == 1
    out[single] = False
    keep1 = single & (p1 > acd.get("tau_lo", thr))
    out[rows[keep1], order[keep1, 0]] = True
    ov = nhat == 2
    add2 = ov & (p2 > acd.get("tau_ov", thr))
    out[rows[add2], order[add2, 1]] = True
    keep1o = ov & (p1 > acd.get("tau_lo", thr))
    out[rows[keep1o], order[keep1o, 0]] = True
    return out


def to_annotation(binary, uri, step=0.01):
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri=uri)
    T, S = binary.shape
    for k in range(S):
        x = np.concatenate([[0], binary[:, k].astype(np.int8), [0]])
        d = np.diff(x)
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            ann[Segment(s * step, e * step), f"spk{k}"] = f"spk{k}"
    return ann


def score_file(job):
    import warnings

    warnings.filterwarnings("ignore")
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    item, pred_dir, method, thr, acd, collar = job
    uid = os.path.basename(item["audio_filepath"])[:-4]
    d = np.load(os.path.join(pred_dir, uid + ".npz"))
    segs = read_rttm(item["rttm_filepath"])
    hyp_bin = decode(d["probs"], d["act"], method, thr, acd, segs)
    hyp = to_annotation(hyp_bin, uid)
    ref = Annotation(uri=uid)
    for s, e, k in segs:
        ref[Segment(s, e)] = k
    m = DiarizationErrorRate(collar=collar, skip_overlap=False)
    r = m(ref, hyp, detailed=True)
    n_ref = len(set(k for _, _, k in segs))
    n_hyp = int((hyp_bin.sum(0) > 0).sum())
    return uid, {
        "total": r["total"], "miss": r["missed detection"], "fa": r["false alarm"], "conf": r["confusion"],
        "n_ref": n_ref, "n_hyp": n_hyp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--decode", default="thr", choices=["thr", "acd", "oracle_count"])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--acd", default="{}")
    ap.add_argument("--collar", type=float, default=0.0)
    ap.add_argument("--per-file", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--label", default=None,
                    help="name recorded as 'decode' and used for the per-file json (default: --decode), "
                         "e.g. acd_tuned or thr_c025")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.manifest)]
    items = [it for it in items if os.path.exists(os.path.join(a.pred_dir, os.path.basename(it["audio_filepath"])[:-4] + ".npz"))]
    acd = json.loads(a.acd)
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(score_file, [(it, a.pred_dir, a.decode, a.thr, acd, a.collar) for it in items]))
    tot = sum(r["total"] for r in res.values())
    agg = {k: sum(r[k] for r in res.values()) / tot * 100 for k in ["miss", "fa", "conf"]}
    der = sum(agg.values())
    sca = np.mean([r["n_hyp"] == r["n_ref"] for r in res.values()]) * 100
    mae = np.mean([abs(r["n_hyp"] - r["n_ref"]) for r in res.values()])
    label = a.label or a.decode
    out = {"tag": a.tag, "pred_dir": a.pred_dir, "decode": label, "n_files": len(res), "DER": round(der, 2),
           "miss": round(agg["miss"], 2), "fa": round(agg["fa"], 2), "conf": round(agg["conf"], 2),
           "SCA": round(sca, 2), "MAE": round(float(mae), 4), "ref_hours": round(tot / 3600, 2)}
    if a.label:
        out.update({"decode_method": a.decode, "acd_params": acd, "thr": a.thr, "collar": a.collar})
    print(json.dumps(out))
    if a.per_file:
        with open(os.path.join(a.pred_dir, f"_perfile_{label}.json"), "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
