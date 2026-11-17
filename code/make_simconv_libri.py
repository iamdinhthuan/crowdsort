"""Read-speech simulation baseline: simulated conversations with 5-8 speakers from LibriSpeech.

This is the generator that produced data/manifests/simconv_v1.jsonl (python make_simconv_libri.py simconv_v1
840 1 -> 823 sessions, 31.3 h, 12.7% overlapped speech). It follows the idea of simulated conversations
(Landini et al., 2022) with single-speaker utterances from LibriSpeech train-clean-100, so that the source of
the speech differs from CrowdSort while the session statistics are matched:
- K ~ U{5..8} speakers; the session is a sequence of turns; the next speaker is drawn uniformly from the
  other K-1 speakers.
- A turn is a run of consecutive speech regions of one utterance (energy VAD, internal pauses >= 0.2 s)
  of target length L ~ LogUniform(1, 12) s.
- After a turn the next one starts after a pause ~ Exp(mean 0.5 s) clipped to [0.1, 3] s, or, with
  probability 0.65, overlaps the end of the previous turn by U(0.3, min(3, 0.6 x the shorter turn)) s;
  these values give 12-13% overlapped speech, as in the CrowdSort sessions. Sessions with fewer than five
  speakers are discarded.
- Session length ~ U(60, 200) s, as the concatenated CrowdSort sessions (mean about 120 s).
- Acoustics: with probability 0.8 every speaker is convolved with a different simulated RIR of one room
  (OpenSLR 28); per-speaker peak level U(0.3, 0.9); with probability 0.7 a point-source noise at
  SNR U(5, 20) dB is added.

python make_simconv_libri.py NAME N SEED
"""
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

ROOT = "/mnt/ssd2/paper/paper_1/data"
LIBRI = f"{ROOT}/raw/librispeech/LibriSpeech/train-clean-100"
RIRS = f"{ROOT}/raw/rirs/RIRS_NOISES"
SR = 16000
P_OV, OV_MAX, PAUSE_MEAN = float(os.environ.get("P_OV", 0.65)), 3.0, 0.5


def vad(x, hop=160, win=400, min_gap=0.2, min_seg=0.1):
    """Energy VAD on clean read speech -> list of (start_s, end_s)."""
    n = 1 + max(0, (len(x) - win) // hop)
    fr = np.lib.stride_tricks.as_strided(x, (n, win), (x.strides[0] * hop, x.strides[0]))
    e = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
    act = e > max(e.max() - 35, -55)
    segs, start = [], None
    for i, a in enumerate(np.append(act, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segs.append([start * hop / SR, (i * hop + win) / SR])
            start = None
    merged = []
    for s, e_ in segs:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1][1] = e_
        else:
            merged.append([s, e_])
    return [(s, min(e_, len(x) / SR)) for s, e_ in merged if e_ - s >= min_seg]


def main():
    name, n_sess, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = random.Random(seed)
    np.random.seed(seed)
    utts = defaultdict(list)
    for f in glob.glob(f"{LIBRI}/*/*/*.flac"):
        utts[f.split("/")[-3]].append(f)
    spks = sorted(utts)
    rooms = defaultdict(list)
    for f in glob.glob(f"{RIRS}/simulated_rirs/*/Room*/*.wav"):
        rooms[os.path.dirname(f)].append(f)
    room_keys = sorted(rooms)
    noises = sorted(glob.glob(f"{RIRS}/pointsource_noises/*.wav"))
    print(len(spks), "speakers", len(room_keys), "rooms", len(noises), "noises", flush=True)

    out_wav, out_rttm = f"{ROOT}/wav/{name}", f"{ROOT}/rttm/{name}"
    os.makedirs(out_wav, exist_ok=True)
    os.makedirs(out_rttm, exist_ok=True)
    lines = []
    for i in range(n_sess):
        K = rng.randint(5, 8)
        who = rng.sample(spks, K)
        target = rng.uniform(60, 200)
        reverb = rng.random() < 0.8
        rirs = rng.sample(rooms[rng.choice(room_keys)], K) if reverb else [None] * K
        if reverb and len(set(rirs)) < K:
            reverb, rirs = False, [None] * K
        gains = [rng.uniform(0.3, 0.9) for _ in range(K)]
        pool = {k: rng.sample(utts[s], len(utts[s])) for k, s in enumerate(who)}
        used = Counter()
        tracks = [np.zeros(int(SR * (target + 30)), dtype=np.float32) for _ in range(K)]
        rttm, t, prev_len, cur = [], 0.0, 0.0, None
        while t < target:
            k = rng.randrange(K) if cur is None else rng.choice([j for j in range(K) if j != cur])
            x, _ = sf.read(pool[k][used[k] % len(pool[k])], dtype="float32")
            used[k] += 1
            regs = vad(x)
            if not regs:
                continue
            L = float(np.exp(rng.uniform(np.log(1.0), np.log(12.0))))
            j0 = rng.randrange(len(regs))
            j1 = j0
            while j1 + 1 < len(regs) and regs[j1][1] - regs[j0][0] < L:
                j1 += 1
            a, b = regs[j0][0], regs[j1][1]
            seg = x[int(a * SR):int(b * SR)]
            turn_len = len(seg) / SR
            if cur is not None:
                if rng.random() < P_OV:
                    t -= rng.uniform(0.3, max(0.31, min(OV_MAX, 0.6 * min(prev_len, turn_len))))
                else:
                    t += min(3.0, max(0.1, rng.expovariate(1 / PAUSE_MEAN)))
                t = max(t, 0.0)
            s0 = int(t * SR)
            if s0 + len(seg) > len(tracks[k]):
                break
            tracks[k][s0:s0 + len(seg)] += seg
            for s, e in regs[j0:j1 + 1]:
                rttm.append((t + s - a, e - s, f"{name}_{i:04d}_s{k}"))
            t += turn_len
            prev_len, cur = turn_len, k
        T = int(SR * (t + 0.5))
        mix = np.zeros(T, dtype=np.float32)
        for k in range(K):
            y = tracks[k][:T]
            if not np.any(y):
                continue
            if reverb:
                h, _ = sf.read(rirs[k], dtype="float32")
                h = h[:, 0] if h.ndim > 1 else h
                h = h[int(np.argmax(np.abs(h))):] / (np.abs(h).max() + 1e-9)
                y = fftconvolve(y, h)[:T].astype(np.float32)
            mix += gains[k] * y / (np.abs(y).max() + 1e-6)
        if rng.random() < 0.7:
            nz, _ = sf.read(rng.choice(noises), dtype="float32")
            nz = nz[:, 0] if nz.ndim > 1 else nz
            nz = np.tile(nz, T // len(nz) + 1)[:T]
            snr = rng.uniform(5, 20)
            p_s, p_n = (mix ** 2).mean() + 1e-10, (nz ** 2).mean() + 1e-10
            mix += nz * np.sqrt(p_s / (p_n * 10 ** (snr / 10)))
        mix = mix / (np.abs(mix).max() + 1e-6) * 0.9
        sid = f"{name}_{i:04d}"
        wav, rp = f"{out_wav}/{sid}.wav", f"{out_rttm}/{sid}.rttm"
        nspk = len({k for _, _, k in rttm})
        if nspk < 5:
            continue
        sf.write(wav, mix, SR, subtype="PCM_16")
        with open(rp, "w") as f:
            for s, d, k in sorted(rttm):
                f.write(f"SPEAKER {sid} 1 {s:.3f} {d:.3f} <NA> <NA> {k} <NA> <NA>\n")
        lines.append({"audio_filepath": wav, "offset": 0.0, "duration": round(T / SR, 3), "label": "infer",
                      "text": "-", "num_speakers": nspk, "rttm_filepath": rp, "uem_filepath": None})
    with open(f"{ROOT}/manifests/{name}.jsonl", "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    print(len(lines), "sessions", round(sum(l["duration"] for l in lines) / 3600, 1), "h",
          sorted(Counter(l["num_speakers"] for l in lines).items()))


if __name__ == "__main__":
    main()
