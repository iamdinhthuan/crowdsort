"""Run inference + scoring for a model over sets x profiles; append results to results/results.jsonl."""
import os as _os
CROWDSORT_ROOT = _os.environ.get("CROWDSORT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, json, os, subprocess, sys

ROOT = CROWDSORT_ROOT
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=f"{ROOT}/code")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base")
    ap.add_argument("--name", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--profiles", default="1.04")
    ap.add_argument("--infer-args", default="")
    ap.add_argument("--decodes", default="thr")
    ap.add_argument("--acd", default='{}')
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    for s in a.sets.split(","):
        man = f"{ROOT}/data/manifests/{s}.jsonl"
        for p in a.profiles.split(","):
            out = f"{ROOT}/exp/{a.name}/{s}/{p}"
            cmd = [PY, f"{ROOT}/code/infer.py", "--model", a.model, "--manifest", man, "--profile", p, "--out", out,
                   "--batch-size", str(a.batch_size)] + a.infer_args.split()
            r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
            if r.returncode != 0:
                print("INFER FAILED", s, p, r.stderr[-3000:], flush=True)
                continue
            for d in a.decodes.split(","):
                r = subprocess.run([PY, f"{ROOT}/code/score.py", "--pred-dir", out, "--manifest", man, "--decode", d,
                                    "--acd", a.acd, "--tag", f"{a.name}|{s}|{p}|{d}", "--per-file"],
                                   env=ENV, capture_output=True, text=True)
                line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "SCORE FAILED " + r.stderr[-2000:]
                print(line, flush=True)
                if line.startswith("{"):
                    rec = json.loads(line)
                    rec.update({"model": a.name, "set": s, "profile": p})
                    with open(f"{ROOT}/results/results.jsonl", "a") as f:
                        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
