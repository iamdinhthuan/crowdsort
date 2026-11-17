"""Pooled DER over a group of sets for several models, with paired bootstrap against a reference model.

python pooled.py --ref base --models base_sc396,base_ad66_528 --group dev|test [--profile 1.04]
"""
import argparse
import json
import os
import subprocess

ROOT = "/mnt/ssd2/paper/paper_1"
GROUPS = {
    "dev": ["ami_dev_mhm", "ami_dev_sdm", "ali_eval_near", "ali_eval_far", "aishell4_dev", "msdwild_fewdev",
            "msdwild_concatdev"],
    "test": ["ami_test_mhm", "ami_test_sdm", "ali_test_near", "ali_test_far", "aishell4_test", "msdwild_few",
             "msdwild_many"],
    "meet": ["ami_test_mhm", "ami_test_sdm", "ali_test_near", "ali_test_far", "aishell4_test"],
    "many": ["aishell4_test", "msdwild_many"],
}


def files(model, sets, profile):
    return [f"{ROOT}/exp/{model}/{s}/{profile}/_perfile_thr.json" for s in sets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--group", default="dev")
    ap.add_argument("--profile", default="1.04")
    a = ap.parse_args()
    sets = GROUPS[a.group]
    ref = files(a.ref, sets, a.profile)
    if not all(os.path.exists(f) for f in ref):
        print("missing reference results")
        return
    for m in a.models.split(","):
        fs = files(m, sets, a.profile)
        if not all(os.path.exists(f) for f in fs):
            print(m.ljust(24), "incomplete")
            continue
        out = subprocess.run(["python3", f"{ROOT}/code/bootstrap.py", ",".join(ref), ",".join(fs), "--n", "3000"],
                             capture_output=True, text=True).stdout.strip()
        r = json.loads(out)
        print(m.ljust(24), f"ref {r['DER_A']:.2f} -> {r['DER_B']:.2f}  delta {r['delta']:+.2f}  CI {r['ci95']}  p {r['p_B_not_better']}")


if __name__ == "__main__":
    main()
