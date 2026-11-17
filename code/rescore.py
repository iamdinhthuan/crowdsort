"""Re-score saved predictions with a named decoding and append the result to results/results.jsonl.

python rescore.py --label acd_tuned --decode acd --acd '{"tau_sil":0.4,"tau_lo":0.35,"tau_ov":0.35}' \
    --models base,ft3_s1 --sets ami_test_mhm,... [--profile 1.04] [--collar 0.0]
The label is stored as 'decode', together with the decoding method, thresholds and collar, so that default and
tuned decodings never share an identifier. Per-file errors go to exp/<model>/<set>/<profile>/_perfile_<label>.json.
"""
import argparse
import json
import os
import subprocess

ROOT = "/mnt/ssd2/paper/paper_1"
PY = f"{ROOT}/.venv-sfx/bin/python"

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--decode", default="thr")
ap.add_argument("--acd", default="{}")
ap.add_argument("--collar", default="0.0")
ap.add_argument("--models", required=True)
ap.add_argument("--sets", required=True)
ap.add_argument("--profile", default="1.04")
a = ap.parse_args()
for m in a.models.split(","):
    for s in a.sets.split(","):
        pd = f"{ROOT}/exp/{m}/{s}/{a.profile}"
        if not os.path.isdir(pd):
            print("missing", pd)
            continue
        r = subprocess.run([PY, f"{ROOT}/code/score.py", "--pred-dir", pd, "--manifest", f"{ROOT}/data/manifests/{s}.jsonl",
                            "--decode", a.decode, "--acd", a.acd, "--collar", a.collar, "--label", a.label,
                            "--tag", f"{m}|{s}|{a.profile}|{a.label}", "--per-file"], capture_output=True, text=True,
                           env=dict(os.environ, PYTHONPATH=f"{ROOT}/code"))
        line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if not line.startswith("{"):
            print("FAILED", m, s, r.stderr[-500:])
            continue
        rec = json.loads(line)
        rec.update({"model": m, "set": s, "profile": a.profile})
        with open(f"{ROOT}/results/results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(m, s, rec["DER"])
