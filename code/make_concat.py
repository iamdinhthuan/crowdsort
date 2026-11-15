"""Many-speaker training sessions by concatenating MSDWild train clips from different videos.

Each session joins clips (with 0.3-1.0 s silence gaps) until it holds 5-8 distinct speakers and lasts at
most 200 s. Speaker labels are prefixed with the clip id, so speakers from different clips are distinct.
"""
import os as _os
CROWDSORT_ROOT = _os.environ.get("CROWDSORT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os
import random
from collections import defaultdict

import numpy as np
import soundfile as sf

import sys

ROOT = CROWDSORT_ROOT + "/data"
SRC, NAME, N, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
random.seed(SEED)
np.random.seed(SEED)

clips = [json.loads(l) for l in open(f"{ROOT}/manifests/{SRC}.jsonl")]
clips = [c for c in clips if 10 <= c["duration"] <= 120]
segs = {}
for c in clips:
    uid = os.path.basename(c["audio_filepath"])[:-4]
    s = []
    for line in open(c["rttm_filepath"]):
        p = line.split()
        s.append((float(p[3]), float(p[4]), f"{uid}_{p[7]}"))
    segs[uid] = s

out_wav = f"{ROOT}/wav/{NAME}"
out_rttm = f"{ROOT}/rttm/{NAME}"
os.makedirs(out_wav, exist_ok=True)
os.makedirs(out_rttm, exist_ok=True)
lines = []
for i in range(N):
    target = random.randint(5, 8)
    pieces, spk, dur = [], set(), 0.0
    tries = 0
    while len(spk) < target and tries < 50:
        tries += 1
        c = random.choice(clips)
        uid = os.path.basename(c["audio_filepath"])[:-4]
        new = {k for _, _, k in segs[uid]}
        if len(spk | new) > 8 or dur + c["duration"] > 200:
            continue
        pieces.append((c, uid))
        spk |= new
        dur += c["duration"] + 0.65
    if len(spk) < 5:
        continue
    sid = f"{NAME}_{i:04d}"
    audio, rttm, t = [], [], 0.0
    for c, uid in pieces:
        x, sr = sf.read(c["audio_filepath"], dtype="float32")
        x = x / (np.abs(x).max() + 1e-6) * random.uniform(0.3, 0.9)
        audio.append(x)
        for s, d, k in segs[uid]:
            rttm.append((t + s, d, k))
        t += len(x) / 16000
        gap = np.zeros(int(16000 * random.uniform(0.3, 1.0)), dtype=np.float32)
        audio.append(gap)
        t += len(gap) / 16000
    wav = f"{out_wav}/{sid}.wav"
    sf.write(wav, np.concatenate(audio), 16000, subtype="PCM_16")
    rp = f"{out_rttm}/{sid}.rttm"
    with open(rp, "w") as f:
        for s, d, k in sorted(rttm):
            f.write(f"SPEAKER {sid} 1 {s:.3f} {d:.3f} <NA> <NA> {k} <NA> <NA>\n")
    lines.append({"audio_filepath": wav, "offset": 0.0, "duration": round(t, 3), "label": "infer", "text": "-",
                  "num_speakers": len(spk), "rttm_filepath": rp, "uem_filepath": None})
with open(f"{ROOT}/manifests/{NAME}.jsonl", "w") as f:
    for l in lines:
        f.write(json.dumps(l) + "\n")
from collections import Counter
print(len(lines), "sessions", round(sum(l["duration"] for l in lines) / 3600, 1), "h",
      sorted(Counter(l["num_speakers"] for l in lines).items()))
