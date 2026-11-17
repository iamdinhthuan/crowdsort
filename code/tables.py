"""Summarise results/results.jsonl: latest record per (model, set, profile, decode).

python tables.py [--models a,b] [--decode thr] [--profile 1.04] [--metric DER]
"""
import argparse
import json

SETS = ["ami_test_mhm", "ami_test_sdm", "ali_test_near", "ali_test_far", "aishell4_test", "msdwild_few", "msdwild_many"]
DEV = ["ami_dev_mhm", "ami_dev_sdm", "ali_eval_near", "ali_eval_far", "aishell4_dev"]
SHORT = {"ami_test_mhm": "AMI-MHM", "ami_test_sdm": "AMI-SDM", "ali_test_near": "Ali-N", "ali_test_far": "Ali-F",
         "aishell4_test": "AIS4", "msdwild_few": "MSDW-few", "msdwild_many": "MSDW-many",
         "ami_dev_mhm": "dAMI-MHM", "ami_dev_sdm": "dAMI-SDM", "ali_eval_near": "dAli-N", "ali_eval_far": "dAli-F",
         "aishell4_dev": "dAIS4"}


def load():
    best = {}
    for l in open("/mnt/ssd2/paper/paper_1/results/results.jsonl"):
        r = json.loads(l)
        best[(r["model"], r["set"], r["profile"], r["decode"])] = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--decode", default="thr")
    ap.add_argument("--profile", default="1.04")
    ap.add_argument("--metric", default="DER")
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    R = load()
    sets = DEV if a.dev else SETS
    models = a.models.split(",") if a.models else sorted({k[0] for k in R})
    print("model".ljust(22) + "".join(SHORT[s].rjust(10) for s in sets) + "   avg")
    for m in models:
        vals = [R.get((m, s, a.profile, a.decode), {}).get(a.metric) for s in sets]
        ok = [v for v in vals if v is not None]
        avg = sum(ok) / len(ok) if len(ok) == len(sets) else float("nan")
        print(m.ljust(22) + "".join(("%.2f" % v if v is not None else "-").rjust(10) for v in vals) + f"  {avg:6.2f}")


if __name__ == "__main__":
    main()
