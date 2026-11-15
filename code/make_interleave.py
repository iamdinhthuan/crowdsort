"""Many-speaker sessions by interleaving speech chunks of several in-the-wild clips (speakers return).

Each clip is cut at silent gaps (>= 0.3 s with no reference speech) into chunks that keep their internal
turns and overlaps. A session combines 2-4 clips whose speakers are distinct (5-8 in total) and draws the next
chunk from a random clip, keeping the chunk order inside each clip, with 0.2-0.8 s gaps. Unlike plain
concatenation, a speaker of one clip reappears after speakers of other clips have talked.

python make_interleave.py SRC_MANIFEST NAME N SEED
"""
import os as _os
CROWDSORT_ROOT = _os.environ.get("CROWDSORT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import soundfile as sf

ROOT = CROWDSORT_ROOT + "/data"
SRC, NAME, N, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
random.seed(SEED)
np.random.seed(SEED)
MIN_GAP = 0.3

clips = [json.loads(l) for l in open(f"{ROOT}/manifests/{SRC}.jsonl")]
clips = [c for c in clips if 10 <= c["duration"] <= 150]


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


info = {}
for c in clips:
    uid, ch = chunks_of(c)
    spk = {k for _, _, segs in ch for _, _, k in segs}
    if ch:
        info[uid] = (c, ch, spk)
uids = list(info)

out_wav, out_rttm = f"{ROOT}/wav/{NAME}", f"{ROOT}/rttm/{NAME}"
os.makedirs(out_wav, exist_ok=True)
os.makedirs(out_rttm, exist_ok=True)
lines = []
for i in range(N):
    target = random.randint(5, 8)
    chosen, spk, tries = [], set(), 0
    while len(spk) < target and tries < 60 and len(chosen) < 4:
        tries += 1
        u = random.choice(uids)
        if u in chosen or len(spk | info[u][2]) > 8:
            continue
        chosen.append(u)
        spk |= info[u][2]
    if len(spk) < 5 or len(chosen) < 2:
        continue
    queues = {u: list(info[u][1]) for u in chosen}
    gains = {u: random.uniform(0.3, 0.9) for u in chosen}
    audio = {u: sf.read(info[u][0]["audio_filepath"], dtype="float32")[0] for u in chosen}
    for u in chosen:
        x = audio[u]
        audio[u] = x / (np.abs(x).max() + 1e-6) * gains[u]
    parts, rttm, t = [], [], 0.0
    while any(queues.values()):
        u = random.choice([v for v in chosen if queues[v]])
        a, b, segs = queues[u].pop(0)
        x = audio[u][int(a * 16000): int(b * 16000)]
        parts.append(x)
        for s, e, k in segs:
            rttm.append((t + s, e - s, k))
        t += len(x) / 16000
        gap = np.zeros(int(16000 * random.uniform(0.2, 0.8)), dtype=np.float32)
        parts.append(gap)
        t += len(gap) / 16000
    sid = f"{NAME}_{i:04d}"
    wav = f"{out_wav}/{sid}.wav"
    sf.write(wav, np.concatenate(parts), 16000, subtype="PCM_16")
    rp = f"{out_rttm}/{sid}.rttm"
    with open(rp, "w") as f:
        for s, d, k in sorted(rttm):
            f.write(f"SPEAKER {sid} 1 {s:.3f} {d:.3f} <NA> <NA> {k} <NA> <NA>\n")
    lines.append({"audio_filepath": wav, "offset": 0.0, "duration": round(t, 3), "label": "infer", "text": "-",
                  "num_speakers": len(spk), "rttm_filepath": rp, "uem_filepath": None})
with open(f"{ROOT}/manifests/{NAME}.jsonl", "w") as f:
    for l in lines:
        f.write(json.dumps(l) + "\n")
print(len(lines), "sessions", round(sum(l["duration"] for l in lines) / 3600, 1), "h",
      sorted(Counter(l["num_speakers"] for l in lines).items()))
