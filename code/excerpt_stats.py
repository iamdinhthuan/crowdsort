"""Statistics of the training excerpts that the sampler actually feeds the model (reviewer check of Section 3.3).

The released model's own training dataset (NeMo AudioToSpeechE2ESpkDiarDataset in subsegment mode, with the
train_ds settings of the released checkpoint and the session length of train_sfx.py) draws excerpts from each
source of the CrowdSort training mix. From the frame targets (80 ms) of each excerpt it reports the number of
active speakers, the share of speakers whose first activity comes after the first chunk of 264 frames of 80 ms (21.12 s,
when the cache already holds earlier speakers), the share of excerpts with at least one such speaker, and returns
per minute (a speaker resuming after at least 10 s of its own silence during which another speaker talked).

python excerpt_stats.py [--n 400] [--seed 0]   -> results/excerpt_stats.json
python excerpt_stats.py --baselines           -> results/excerpt_stats_baselines.json (simulated sessions of the baselines)
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = "/mnt/ssd2/paper/paper_1"
BASE_NEMO = f"{ROOT}/models/base/Nemotron-3-Diarization.nemo"
MIX = f"{ROOT}/data/manifests/train_mix_v3ci.jsonl"
SOURCES = {"Arrival sessions": ["msdwild_concat_v2"], "Return sessions": ["msdwild_inter_v2"],
           "MSDWild clips": ["msdwild_train"],
           "Meetings": ["ami_train_mhm", "ami_train_sdm", "ali_train_near", "ali_train_far_ch0", "aishell4_train"]}
BASELINES = {"SimConv-MSD sessions": ("train_mix_v3sc.jsonl", ["msdwild_simconv"]),
             "SimConv-MSD-M sessions": ("train_mix_v3scm.jsonl", ["msdwild_simconv_m"]),
             "FastMSS sessions": ("train_mix_v3fm.jsonl", ["out_5to8"])}
CHUNK_SEC = 264 * 0.08  # one chunk of 264 frames of 80 ms


def stats(t, dur):
    t = t.numpy() > 0.5
    fps = t.shape[0] / dur  # target frame rate from the excerpt duration
    act = [j for j in range(t.shape[1]) if t[:, j].any()]
    first = [int(np.argmax(t[:, j])) for j in act]
    late = sum(f / fps >= CHUNK_SEC for f in first)
    returns = 0
    for j in act:
        idx = np.flatnonzero(t[:, j])
        gaps = np.flatnonzero(np.diff(idx) > 1)
        for g in gaps:
            a, b = idx[g] + 1, idx[g + 1]
            others = np.delete(t[a:b], j, axis=1).any(axis=1).sum() if t.shape[1] > 1 else 0
            if (b - a) / fps >= 10 and others > 0:
                returns += 1
    return len(act), late, returns, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nemo.collections.asr.models import SortformerEncLabelModel
    from omegaconf import open_dict
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    model = SortformerEncLabelModel.restore_from(BASE_NEMO, map_location="cpu")
    if a.baselines:
        groups = {n: (f"{ROOT}/data/manifests/{m}", k) for n, (m, k) in BASELINES.items()}
    else:
        groups = {n: (MIX, k) for n, k in SOURCES.items()}
    tmp = f"{ROOT}/results/_excerpt_manifest.jsonl"
    out = {}
    for name, (mix, keys) in groups.items():
        lines = open(mix).readlines()
        sel = [l for l in lines if any(f"/{k}/" in json.loads(l)["audio_filepath"] for k in keys)]
        sel = random.sample(sel, min(a.n, len(sel)))
        open(tmp, "w").writelines(sel)
        cfg = model.cfg.train_ds
        with open_dict(cfg):
            cfg.manifest_filepath = tmp
            cfg.session_len_sec = 105.0
            cfg.subsegment_single_chunk_min_len_sec = 15.0
            cfg.num_workers = 0
            cfg.batch_size = 1
            cfg.shuffle = False
        model.setup_training_data(cfg)
        ds = model._train_dl.dataset
        rows = []
        for i in range(len(ds)):
            it = ds[i]
            rows.append(stats(it[2], float(it[1]) / 16000))
        nspk = np.array([r[0] for r in rows]); late = np.array([r[1] for r in rows])
        ret = np.array([r[2] for r in rows]); dur = np.array([r[3] for r in rows])
        out[name] = {"excerpts": len(rows), "mean_duration_s": round(float(dur.mean()), 1),
                     "mean_speakers": round(float(nspk.mean()), 2),
                     "late_speaker_share_pct": round(100 * late.sum() / max(1, nspk.sum()), 1),
                     "excerpts_with_late_speaker_pct": round(100 * float((late > 0).mean()), 1),
                     "returns_per_min": round(float(ret.sum() / (dur.sum() / 60)), 2)}
        print(name, out[name], flush=True)
    os.remove(tmp)
    json.dump(out, open(f"{ROOT}/results/excerpt_stats{'_baselines' if a.baselines else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
