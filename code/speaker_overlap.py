"""Post-hoc check of audio and speaker overlap between the MSDWild clips used for fine-tuning and the MSDWild test sets.

python speaker_overlap.py extract   -> results/speaker_overlap/{emb,fp}.npz, meta.json
python speaker_overlap.py analyse   -> results/speaker_overlap/summary.json (+ printed report)

Training side: the 2226 MSDWild training clips of the fine-tuning mix (msdwild_train_v2; the 250 development clips
are excluded). Test side: MSDWild few (490) and many (177).
1. Exact duplicates: MD5 of the 16 kHz PCM samples.
2. Near duplicates: log-mel fingerprints of 3 s windows (1 s hop), mean-normalised per band, pooled to 32 bands x
   15 steps and L2-normalised; maximum cosine between any test window and any training window.
3. Speakers: one embedding per (clip, speaker) from the speaker's non-overlapped speech (segments of at least 0.5 s,
   at least 3 s in total, at most 30 s), with two public models trained on VoxCeleb (SpeechBrain ECAPA-TDNN and
   microsoft/wavlm-base-plus-sv). Speakers with less non-overlapped speech are reported as not assessable.
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch

ROOT = "/mnt/ssd2/paper/paper_1"
OUT = f"{ROOT}/results/speaker_overlap"
SETS = {"train": "msdwild_train_v2", "few": "msdwild_few", "many": "msdwild_many"}
MIN_SEG, MIN_TOT, MAX_TOT = 0.5, 3.0, 30.0
SR = 16000
WAVLM = os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--wavlm-base-plus-sv/snapshots/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e")


def manifest(name):
    return [json.loads(l) for l in open(f"{ROOT}/data/manifests/{name}.jsonl")]


def rttm(path):
    d = defaultdict(list)
    for l in open(path):
        x = l.split()
        d[x[7]].append((float(x[3]), float(x[3]) + float(x[4])))
    return d


def solo_segments(spk):
    """Non-overlapped segments of each speaker (speech of other speakers removed)."""
    out = {}
    for s, segs in spk.items():
        others = sorted(seg for t, v in spk.items() if t != s for seg in v)
        keep = []
        for a, b in sorted(segs):
            cur = [(a, b)]
            for oa, ob in others:
                nxt = []
                for ca, cb in cur:
                    if ob <= ca or oa >= cb:
                        nxt.append((ca, cb))
                    else:
                        if oa > ca:
                            nxt.append((ca, oa))
                        if ob < cb:
                            nxt.append((ob, cb))
                cur = nxt
            keep += [(ca, cb) for ca, cb in cur if cb - ca >= MIN_SEG]
        out[s] = keep
    return out


def fingerprint(x, mel):
    m = torch.log(mel(torch.from_numpy(x).float().cuda()) + 1e-6)  # (64, T) at 100 frames/s
    m = m - m.mean(dim=1, keepdim=True)
    w = m.unfold(1, 300, 100)  # (64, nwin, 300)
    if w.shape[1] == 0:
        return np.zeros((0, 480), np.float32)
    w = w.permute(1, 0, 2).unsqueeze(1)  # (nwin, 1, 64, 300)
    p = torch.nn.functional.adaptive_avg_pool2d(w, (32, 15)).flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    return torch.nn.functional.normalize(p, dim=1).cpu().numpy().astype(np.float32)


def extract(sets=None, tag=""):
    sets = sets or SETS
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import AutoFeatureExtractor, WavLMForXVector
    os.makedirs(OUT, exist_ok=True)
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"},
                                           savedir=f"{OUT}/_ecapa")
    wfe = AutoFeatureExtractor.from_pretrained(WAVLM)
    wlm = WavLMForXVector.from_pretrained(WAVLM).cuda().eval()
    mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=400, hop_length=160, n_mels=64).cuda()
    keys, ee, ew, tot, md5, fp_key, fp = [], [], [], [], {}, [], []
    skipped = []
    for split, name in sets.items():
        for r in manifest(name):
            uid = os.path.basename(r["audio_filepath"])[:-4]
            x, sr = sf.read(r["audio_filepath"], dtype="float32")
            assert sr == SR, (r["audio_filepath"], sr)
            if x.ndim > 1:
                x = x.mean(1)
            if tag:
                x = x[:int(1800 * SR)]
            md5[f"{split}:{uid}"] = hashlib.md5((x * 32768).round().astype(np.int16).tobytes()).hexdigest()
            f = fingerprint(x, mel) if not tag else np.zeros((0, 480), np.float32)
            fp.append(f)
            fp_key += [f"{split}:{uid}"] * len(f)
            for s, segs in solo_segments(rttm(r["rttm_filepath"])).items():
                t = sum(b - a for a, b in segs)
                if t < MIN_TOT:
                    skipped.append(f"{split}:{uid}:{s}")
                    continue
                parts, acc = [], 0.0
                for a, b in segs:
                    b = min(b, a + MAX_TOT - acc)
                    parts.append(x[int(a * SR):int(b * SR)])
                    acc += b - a
                    if acc >= MAX_TOT:
                        break
                y = np.concatenate(parts)
                with torch.no_grad():
                    e1 = ecapa.encode_batch(torch.from_numpy(y).unsqueeze(0).cuda()).squeeze().cpu().numpy()
                    inp = wfe(y, sampling_rate=SR, return_tensors="pt").to("cuda")
                    e2 = wlm(**inp).embeddings.squeeze().cpu().numpy()
                keys.append(f"{split}:{uid}:{s}")
                ee.append(e1)
                ew.append(np.asarray(e2).reshape(-1))
                tot.append(min(t, MAX_TOT))
        print(split, "done", len(keys), flush=True)
    if tag:
        np.savez(f"{OUT}/emb{tag}.npz", keys=np.array(keys), ecapa=np.stack(ee), wavlm=np.stack(ew), dur=np.array(tot))
        return
    np.savez(f"{OUT}/emb.npz", keys=np.array(keys), ecapa=np.stack(ee), wavlm=np.stack(ew), dur=np.array(tot))
    np.savez(f"{OUT}/fp.npz", keys=np.array(fp_key), fp=np.concatenate(fp))
    json.dump({"md5": md5, "not_assessable": skipped}, open(f"{OUT}/meta.json", "w"))


def cos(a, b):
    a = torch.nn.functional.normalize(torch.from_numpy(a).float().cuda(), dim=1)
    b = torch.nn.functional.normalize(torch.from_numpy(b).float().cuda(), dim=1)
    return a @ b.T


# Levels fixed before any DER on the subsets was computed. Thresholds per model are percentiles of the cosine scores
# of different speakers within one clip (same recording channel, so these scores are higher than those of different
# speakers across clips, and the thresholds are conservative for "different").
LEVELS = {"strict": ("both", 99.9), "moderate": ("both", 99.0), "lenient": ("either", 99.0)}
SYSTEMS = {"N3D": ["base"], "FT-clips": ["c3_noconcat", "c3_noconcat_s2", "c3_noconcat_s3"],
           "SimConv-MSD-M": ["c3_simconvm_s1", "c3_simconvm_s2", "c3_simconvm_s3"],
           "CrowdSort": ["c3_ci_s1", "c3_ci_s2", "c3_ci_s3"]}
PAIRS = [("N3D", "CrowdSort"), ("FT-clips", "CrowdSort"), ("SimConv-MSD-M", "CrowdSort")]


def analyse():
    import subprocess
    meta = json.load(open(f"{OUT}/meta.json"))
    rep = {}
    # 1. exact duplicates
    by = defaultdict(list)
    for k, h in meta["md5"].items():
        by[h].append(k)
    dup = [v for v in by.values() if len({x.split(":")[0] for x in v}) > 1 and any(x.startswith("train:") for x in v)]
    rep["exact_duplicates_train_test"] = dup
    # 2. near duplicates: max window cosine of each test clip against all training windows
    f = np.load(f"{OUT}/fp.npz")
    fk, fp = f["keys"], f["fp"]
    tr = np.array([k.startswith("train:") for k in fk])
    trk = fk[tr]
    near = {}
    for split in ("few", "many"):
        te = np.array([k.startswith(split + ":") for k in fk])
        best, arg = [], []
        for i in range(0, te.sum(), 4096):
            s = cos(fp[te][i:i + 4096], fp[tr])
            v, j = s.max(dim=1)
            best.append(v.cpu().numpy())
            arg.append(j.cpu().numpy())
        best, arg = np.concatenate(best), np.concatenate(arg)
        clip = {}
        for k, v, j in zip(fk[te], best, arg):
            if v > clip.get(k, (-1, ""))[0]:
                clip[k] = (float(v), str(trk[j]))
        near[split] = clip
    rep["near_duplicate_clips"] = {s: {t: sum(v[0] >= t for v in c.values()) for t in (0.90, 0.95, 0.98)}
                                   for s, c in near.items()}
    rep["near_duplicate_top"] = {s: sorted(c.items(), key=lambda kv: -kv[1][0])[:10] for s, c in near.items()}
    # 3. speakers
    e = np.load(f"{OUT}/emb.npz")
    keys = e["keys"]
    split = np.array([k.split(":")[0] for k in keys])
    clip = np.array([":".join(k.split(":")[:2]) for k in keys])
    S = {m: cos(e[m], e[m]).cpu().numpy() for m in ("ecapa", "wavlm")}
    same_clip = clip[:, None] == clip[None, :]
    off = ~np.eye(len(keys), dtype=bool)
    thr = {m: {q: float(np.percentile(S[m][same_clip & off], q)) for q in (99.0, 99.9)} for m in S}
    rep["thresholds"] = thr
    rep["n_within_clip_pairs"] = int((same_clip & off).sum() // 2)
    tot_spk = defaultdict(int)
    for k in list(keys) + meta["not_assessable"]:
        tot_spk[k.split(":")[0]] += 1
    rep["speakers_assessable"] = {s: int((split == s).sum()) for s in SETS}
    rep["speakers_total"] = dict(tot_spk)
    ids = np.array([int(k.split(":")[1]) for k in keys])
    match = {}
    for lev, (rule, q) in LEVELS.items():
        hit = [S[m] >= thr[m][q] for m in S]
        M = (hit[0] & hit[1]) if rule == "both" else (hit[0] | hit[1])
        M &= ~same_clip
        out = {}
        for s in ("few", "many", "train"):
            rows = split == s
            cand = M[rows][:, split == "train"]
            out[s] = float(cand.any(axis=1).mean())
            if s != "train":
                gaps = np.abs(ids[rows][:, None] - ids[split == "train"][None, :])[cand]
                out[s + "_median_id_gap"] = float(np.median(gaps)) if len(gaps) else None
        rand = np.abs(ids[split == "many"][:, None] - ids[split == "train"][None, :])
        out["random_median_id_gap"] = float(np.median(rand))
        match[lev] = out
        flagged = {c.split(":")[1] for c, r in zip(clip, M[:, split == "train"].any(axis=1)) if r and c.startswith("many:")}
        flagged |= {k.split(":")[1] for k, v in near["many"].items() if v[0] >= 0.95}
        rep.setdefault("many_flagged", {})[lev] = sorted(flagged)
    rep["match_rate"] = match
    # 4. DER on MSDWild many without flagged recordings
    der = {}
    work = f"{OUT}/perfile"
    for lev, flagged in rep["many_flagged"].items():
        paths = {}
        for name, ms in SYSTEMS.items():
            ps = []
            for m in ms:
                d = json.load(open(f"{ROOT}/exp/{m}/msdwild_many/1.04/_perfile_thr.json"))
                d = {k: v for k, v in d.items() if k not in flagged}
                os.makedirs(f"{work}/{lev}", exist_ok=True)
                p = f"{work}/{lev}/{m}.json"
                json.dump(d, open(p, "w"))
                ps.append(p)
            paths[name] = "+".join(ps)
        res = {"n_recordings": 177 - len(flagged)}
        for a, b in PAIRS:
            r = subprocess.run(["python3", f"{ROOT}/code/bootstrap.py", paths[a], paths[b], "--group"],
                               capture_output=True, text=True)
            res[f"{b} vs {a}"] = json.loads(r.stdout)
        der[lev] = res
    rep["der_many_clean"] = der
    json.dump(rep, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("exact_duplicates_train_test", "near_duplicate_clips", "thresholds",
                                          "speakers_assessable", "speakers_total", "match_rate")}, indent=1))
    for lev, r in der.items():
        print(lev, r["n_recordings"], {k: (v["DER_A"], v["DER_B"], v["delta"], v["ci95"], v["p_B_not_better"])
                                       for k, v in r.items() if k != "n_recordings"})


if __name__ == "__main__":
    POOL = {"vox": "voxconverse_test", "ami": "ami_train_mhm"}
    {"extract": extract, "analyse": analyse, "pool": lambda: extract(POOL, "_pool")}[sys.argv[1]]()
