"""Memory-aware scheduled synthesis (CrowdSort-S): many-speaker episodes that fit one training sample.

Each episode lasts L ~ U(80, 105) s, so that the subsegment sampler (at most 105 s) sees the whole schedule.
Clips are cut at silent gaps into conversational units that keep their internal turns and overlaps (as in
make_interleave.py). An episode combines one anchor clip and 1-3 newcomer clips with distinct speakers
(5-8 in total) and follows three phases defined relative to the speaker memory of the released model
(cache and training chunk of 264 frames = 21.1 s):
  1. establish  [0, T1), T1 ~ U(25, 40) s: anchor units only, so that the cache is filled and compressed;
  2. arrive     [T1, T2), T2 ~ U(0.65, 0.8) L: newcomer clips enter one after another (late arrivals),
                 their units interleaved with occasional anchor units (probability 0.25);
  3. return     [T2, L]: anchor units resume (speakers return after a long absence), mixed with the
                 remaining newcomer units.
Units keep their order inside each clip. With --perturb, every clip gets a random band limitation
(high-pass 50-300 Hz, low-pass 3.4-8 kHz) and a per-unit gain jitter of +-3 dB, and the whole episode gets a
continuous noise floor (pink-like noise, 25-40 dB below the speech level), so that digital silence and
abrupt channel changes no longer mark the arrival of a new clip.

python make_schedule.py SRC_MANIFEST NAME N SEED [--perturb]
"""
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

CROWDSORT_ROOT = os.environ.get("CROWDSORT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = f"{CROWDSORT_ROOT}/data"
SR = 16000
MIN_GAP = 0.3


def chunks_of(c):
    uid = os.path.basename(c["audio_filepath"])[:-4]
    segs = sorted((float(l.split()[3]), float(l.split()[3]) + float(l.split()[4]), f"{uid}_{l.split()[7]}")
                  for l in open(c["rttm_filepath"]))
    out, cur, cur_end = [], [], -1.0
    for s, e, k in segs:
        if cur and s - cur_end >= MIN_GAP:
            out.append(cur)
            cur = []
        cur.append((s, e, k))
        cur_end = max(cur_end, e)
    if cur:
        out.append(cur)
    res = []
    for ch in out:
        a = max(0.0, min(s for s, _, _ in ch) - 0.1)
        b = min(c["duration"], max(e for _, e, _ in ch) + 0.1)
        res.append((a, b, [(s - a, e - a, k) for s, e, k in ch]))
    return uid, res


def pink(n, rng):
    w = rng.standard_normal(n + 4096).astype(np.float32)
    sos = butter(1, 0.02, btype="low", output="sos")
    x = 0.6 * sosfilt(sos, w) + 0.4 * w * 0.2
    x = x[4096:]
    return x / (np.std(x) + 1e-8)


def band(x, rng):
    hp = rng.uniform(50, 300)
    lp = rng.uniform(3400, 7900)
    return sosfilt(butter(2, [hp, lp], btype="band", fs=SR, output="sos"), x).astype(np.float32)


def main():
    src, name, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    perturb = "--perturb" in sys.argv
    random.seed(seed)
    rng = np.random.default_rng(seed)
    clips = [json.loads(l) for l in open(f"{ROOT}/manifests/{src}.jsonl")]
    clips = [c for c in clips if 10 <= c["duration"] <= 150]
    info = {}
    for c in clips:
        uid, ch = chunks_of(c)
        if ch:
            info[uid] = (c, ch, {k for _, _, segs in ch for _, _, k in segs})
    uids = sorted(info)
    anchors = [u for u in uids if sum(b - a for a, b, _ in info[u][1]) >= 30 and len(info[u][2]) >= 2]
    out_wav, out_rttm = f"{ROOT}/wav/{name}", f"{ROOT}/rttm/{name}"
    os.makedirs(out_wav, exist_ok=True)
    os.makedirs(out_rttm, exist_ok=True)
    lines, stats = [], Counter()
    for i in range(n):
        L = random.uniform(80, 105)
        T1 = random.uniform(25, 40)
        T2 = random.uniform(0.65, 0.8) * L
        target = random.randint(5, 8)
        anchor = random.choice(anchors)
        chosen, spk, tries = [anchor], set(info[anchor][2]), 0
        while len(spk) < target and tries < 60 and len(chosen) < 4:
            tries += 1
            u = random.choice(uids)
            if u in chosen or len(spk | info[u][2]) > 8:
                continue
            chosen.append(u)
            spk |= info[u][2]
        if len(spk) < 5:
            stats["skip_spk"] += 1
            continue
        queues = {u: list(info[u][1]) for u in chosen}
        audio = {}
        for u in chosen:
            x = sf.read(info[u][0]["audio_filepath"], dtype="float32")[0]
            if perturb:
                x = band(x, rng)
            audio[u] = x / (np.abs(x).max() + 1e-6) * random.uniform(0.3, 0.9)
        newcomers = chosen[1:]
        entered = []
        parts, rttm, t = [], [], 0.0

        def place(u):
            nonlocal t
            a, b, segs = queues[u].pop(0)
            x = audio[u][int(a * SR): int(b * SR)]
            if perturb:
                x = x * (10 ** (random.uniform(-3, 3) / 20))
            parts.append(x)
            for s, e, k in segs:
                rttm.append((t + s, e - s, k))
            t += len(x) / SR
            g = np.zeros(int(SR * random.uniform(0.2, 0.8)), dtype=np.float32)
            parts.append(g)
            t += len(g) / SR

        while t < L:
            if t < T1:
                pool = [anchor] if queues[anchor] else [u for u in newcomers if queues[u]][:1]
            elif t < T2:
                waiting = [u for u in newcomers if u not in entered and queues[u]]
                # admit the next newcomer when the current ones have spoken at least one unit
                if waiting and (not entered or random.random() < 0.35):
                    entered.append(waiting[0])
                    pool = [waiting[0]]
                elif queues[anchor] and random.random() < 0.25:
                    pool = [anchor]
                else:
                    pool = [u for u in entered if queues[u]] or [u for u in chosen if queues[u]]
            else:
                ret = [anchor] if queues[anchor] and random.random() < 0.5 else []
                pool = ret or [u for u in entered if queues[u]] or [u for u in chosen if queues[u]]
            if not pool:
                break
            place(random.choice(pool))
        wav_all = np.concatenate(parts)[: int(L * SR)] if t > L else np.concatenate(parts)
        dur = len(wav_all) / SR
        rttm = [(s, min(d, dur - s), k) for s, d, k in rttm if s < dur - 0.05]
        present = {k for _, _, k in rttm}
        if len(present) < 5:
            stats["skip_present"] += 1
            continue
        if perturb:
            rms = np.sqrt(np.mean(wav_all[np.abs(wav_all) > 1e-4] ** 2) + 1e-12)
            snr = random.uniform(25, 40)
            wav_all = wav_all + pink(len(wav_all), rng) * rms * 10 ** (-snr / 20)
        first = {}
        for s, d, k in sorted(rttm):
            first.setdefault(k, s)
        stats["late_arrivals"] += sum(1 for v in first.values() if v > T1)
        stats["speakers"] += len(present)
        sid = f"{name}_{i:04d}"
        wav = f"{out_wav}/{sid}.wav"
        sf.write(wav, np.clip(wav_all, -1, 1), SR, subtype="PCM_16")
        rp = f"{out_rttm}/{sid}.rttm"
        with open(rp, "w") as f:
            for s, d, k in sorted(rttm):
                if d > 0:
                    f.write(f"SPEAKER {sid} 1 {s:.3f} {d:.3f} <NA> <NA> {k} <NA> <NA>\n")
        lines.append({"audio_filepath": wav, "offset": 0.0, "duration": round(dur, 3), "label": "infer", "text": "-",
                      "num_speakers": len(present), "rttm_filepath": rp, "uem_filepath": None})
    with open(f"{ROOT}/manifests/{name}.jsonl", "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    print(len(lines), "episodes", round(sum(l["duration"] for l in lines) / 3600, 1), "h",
          sorted(Counter(l["num_speakers"] for l in lines).items()),
          f"late arrivals per episode {stats['late_arrivals'] / max(1, len(lines)):.2f}", dict(stats))


if __name__ == "__main__":
    main()
