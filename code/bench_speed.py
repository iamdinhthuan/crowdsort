"""Throughput (RTFx) and parameter count of a model under a latency profile.

python bench_speed.py --model base|<nemo> [--sfx-json ..] --profile 1.04 --batch 1 --seconds 600
Uses the first files of AMI test (Mix-Headset) cut to --seconds; reports the median of 3 runs after warm-up.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import load_model, set_profile  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base")
    ap.add_argument("--name", default=None)
    ap.add_argument("--sfx-json", default="{}")
    ap.add_argument("--cache-rescore", action="store_true")
    ap.add_argument("--profile", default="1.04")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--spkcache", type=int, default=None)
    a = ap.parse_args()
    m = load_model(a.model, a.sfx_json, a.cache_rescore, torch.device("cuda"))
    set_profile(m, a.profile, a.spkcache)
    items = [json.loads(l) for l in open("/mnt/ssd2/paper/paper_1/data/manifests/ami_test_mhm.jsonl")][: a.batch]
    x = np.stack([sf.read(it["audio_filepath"], dtype="float32", frames=a.seconds * 16000)[0] for it in items])
    xa = torch.from_numpy(x).cuda()
    la = torch.full((len(items),), x.shape[1], device="cuda")
    times = []
    for r in range(4):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            m(audio_signal=xa, audio_signal_length=la)
        torch.cuda.synchronize()
        if r > 0:
            times.append(time.time() - t0)
    rtfx = a.batch * a.seconds / float(np.median(times))
    n_params = sum(p.numel() for p in m.parameters())
    n_new = sum(p.numel() for n, p in m.named_parameters() if n.startswith("sfx_"))
    out = {"name": a.name or a.model, "profile": a.profile, "spkcache": a.spkcache or 264, "batch": a.batch, "rtfx": round(rtfx, 1),
           "params_M": round(n_params / 1e6, 2), "new_params_M": round(n_new / 1e6, 3)}
    print(json.dumps(out))
    with open("/mnt/ssd2/paper/paper_1/results/speed.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
