"""Grid-search ACD thresholds on dev predictions of one model (CPU only).

python tune_acd.py --model base --profile 1.04
Prints the mean dev DER for every setting, best last; writes results/acd_<model>.json.
"""
import os as _os
CROWDSORT_ROOT = _os.environ.get("CROWDSORT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import score_file  # noqa: E402

ROOT = CROWDSORT_ROOT
DEV = ["ami_dev_mhm", "ami_dev_sdm", "ali_eval_near", "ali_eval_far", "aishell4_dev"]


def der_for(args):
    model, profile, method, acd = args
    out = []
    for s in DEV:
        items = [json.loads(l) for l in open(f"{ROOT}/data/manifests/{s}.jsonl")]
        pd = f"{ROOT}/exp/{model}/{s}/{profile}"
        res = [score_file((it, pd, method, 0.5, acd, 0.0))[1] for it in items]
        tot = sum(r["total"] for r in res)
        out.append(100 * sum(r["miss"] + r["fa"] + r["conf"] for r in res) / tot)
    return acd, sum(out) / len(out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base")
    ap.add_argument("--profile", default="1.04")
    a = ap.parse_args()
    grid = [{"tau_sil": s, "tau_lo": l, "tau_ov": o}
            for s, l, o in itertools.product([0.4, 0.6, 0.8, 0.95], [0.2, 0.35, 0.5], [0.2, 0.35, 0.5, 1.01])]
    jobs = [(a.model, a.profile, "thr", {})] + [(a.model, a.profile, "acd", g) for g in grid]
    with ProcessPoolExecutor(14) as ex:
        res = list(ex.map(der_for, jobs))
    res.sort(key=lambda r: -r[1])
    for acd, m, per in res:
        print(json.dumps(acd), round(m, 3), [round(x, 2) for x in per])
    json.dump({"best": res[-1][0], "best_dev": res[-1][1], "thr_dev": [r for r in res if r[0] == {}][0][1]},
              open(f"{ROOT}/results/acd_{a.model}.json", "w"))


if __name__ == "__main__":
    main()
