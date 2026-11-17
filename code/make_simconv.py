"""Simulated-conversation baseline (after Landini et al., 2022) built from the same MSDWild training clips.

Unlike CrowdSort, which keeps natural multi-speaker units, this baseline rebuilds conversations from
single-speaker segments: every speaker's non-overlapped speech regions (>= 0.5 s) are extracted, 5-8
speakers from different clips are combined, and turns are placed one after another with transition times
(pauses, or overlaps when negative) and speaker-change probability drawn from the empirical statistics of the
training clips themselves. Session durations follow the concatenation sessions (60-200 s).

python make_simconv.py SRC_MANIFEST NAME N SEED [--match REF_MANIFEST]

--match REF_MANIFEST: matched control. Every one of the 5-8 chosen speakers must appear in the session (sessions in
which a chosen speaker never takes a turn are redrawn), and session durations are drawn from the durations of
REF_MANIFEST (the concatenated sessions), so that speaker count and duration follow CrowdSort.
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

CROWDSORT_ROOT = os.environ.get("CROWDSORT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = f"{CROWDSORT_ROOT}/data"
SR = 16000


def load_segs(c):
    uid = os.path.basename(c["audio_filepath"])[:-4]
    return uid, sorted((float(l.split()[3]), float(l.split()[3]) + float(l.split()[4]), l.split()[7])
                       for l in open(c["rttm_filepath"]))


def single_regions(segs):
    """Non-overlapped regions of each speaker (>= 0.5 s)."""
    out = defaultdict(list)
    for s, e, k in segs:
        cuts = [(s, e)]
        for s2, e2, k2 in segs:
            if k2 == k or e2 <= s or s2 >= e:
                continue
            nxt = []
            for a, b in cuts:
                if e2 <= a or s2 >= b:
                    nxt.append((a, b))
                    continue
                if s2 > a:
                    nxt.append((a, s2))
                if e2 < b:
                    nxt.append((e2, b))
            cuts = nxt
        out[k] += [(a, b) for a, b in cuts if b - a >= 0.5]
    return out


def main():
    src, name, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    match = sys.argv[sys.argv.index("--match") + 1] if "--match" in sys.argv else None
    random.seed(seed)
    ref_durs = [json.loads(l)["duration"] for l in open(f"{ROOT}/manifests/{match}.jsonl")] if match else None
    clips = [json.loads(l) for l in open(f"{ROOT}/manifests/{src}.jsonl")]
    clips = [c for c in clips if 10 <= c["duration"] <= 150]
    pools, gaps, same, turns = [], [], 0, 0
    for c in clips:
        uid, segs = load_segs(c)
        for k, regs in single_regions(segs).items():
            if sum(b - a for a, b in regs) >= 3.0:
                pools.append((c["audio_filepath"], f"{uid}_{k}", uid, sorted(regs)))
        # transition statistics between consecutive turns
        for (s1, e1, k1), (s2, e2, k2) in zip(segs, segs[1:]):
            turns += 1
            if k1 == k2:
                same += 1
            else:
                gaps.append(max(-0.5 * (e1 - s1), s2 - e1))
    p_same = same / max(1, turns)
    gaps = np.clip(np.array(gaps), -2.0, 3.0)
    print(f"{len(pools)} speakers, p_same={p_same:.2f}, gap median={np.median(gaps):.2f} s, overlap rate={np.mean(gaps < 0):.2f}")
    out_wav, out_rttm = f"{ROOT}/wav/{name}", f"{ROOT}/rttm/{name}"
    os.makedirs(out_wav, exist_ok=True)
    os.makedirs(out_rttm, exist_ok=True)
    lines, cache = [], {}

    def audio(path):
        if path not in cache:
            if len(cache) > 64:
                cache.clear()
            x = sf.read(path, dtype="float32")[0]
            cache[path] = x / (np.abs(x).max() + 1e-6)
        return cache[path]

    i, redraws = 0, 0
    while i < n:
        target_len = random.choice(ref_durs) if match else random.uniform(60, 200)
        nspk = random.randint(5, 8)
        chosen, used = [], set()
        while len(chosen) < nspk:
            p = random.choice(pools)
            if p[2] in used:
                continue
            used.add(p[2])
            chosen.append(p)
        gain = {p[1]: random.uniform(0.3, 0.9) for p in chosen}
        ptr = {p[1]: 0 for p in chosen}
        buf = np.zeros(int((target_len + 30) * SR), dtype=np.float32)
        rttm, t_end, cur = [], 0.0, random.choice(chosen)
        while t_end < target_len:
            if random.random() >= p_same:
                cur = random.choice([p for p in chosen if p[1] != cur[1]])
            path, lab, _, regs = cur
            a, b = regs[ptr[lab] % len(regs)]
            ptr[lab] += 1
            x = audio(path)[int(a * SR): int(b * SR)] * gain[lab]
            gap = 0.3 if not rttm else float(random.choice(gaps))
            start = max(0.0, t_end + gap)
            i0 = int(start * SR)
            if i0 + len(x) > len(buf):
                break
            buf[i0:i0 + len(x)] += x
            rttm.append((start, len(x) / SR, lab))
            t_end = max(t_end, start + len(x) / SR)
        if match and len({k for _, _, k in rttm}) < nspk:
            redraws += 1
            continue
        wav_all = buf[: int(t_end * SR)]
        sid = f"{name}_{i:04d}"
        wav = f"{out_wav}/{sid}.wav"
        sf.write(wav, np.clip(wav_all, -1, 1), SR, subtype="PCM_16")
        rp = f"{out_rttm}/{sid}.rttm"
        with open(rp, "w") as f:
            for s, d, k in sorted(rttm):
                f.write(f"SPEAKER {sid} 1 {s:.3f} {d:.3f} <NA> <NA> {k} <NA> <NA>\n")
        present = len({k for _, _, k in rttm})
        lines.append({"audio_filepath": wav, "offset": 0.0, "duration": round(t_end, 3), "label": "infer", "text": "-",
                      "num_speakers": present, "rttm_filepath": rp, "uem_filepath": None})
        i += 1
    with open(f"{ROOT}/manifests/{name}.jsonl", "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    if match:
        print("redrawn sessions:", redraws)
    print(len(lines), "sessions", round(sum(l["duration"] for l in lines) / 3600, 1), "h",
          sorted(Counter(l["num_speakers"] for l in lines).items()))


if __name__ == "__main__":
    main()
