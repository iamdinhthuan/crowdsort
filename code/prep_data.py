"""Prepare public diarization corpora: 16 kHz mono wavs, reference RTTMs, NeMo manifests.

Layout:
  data/wav/<set>/<id>.wav      16 kHz mono int16
  data/rttm/<set>/<id>.rttm    reference (file-id column == <id>)
  data/manifests/<set>.jsonl   NeMo diarization manifest (one full recording per line)

Usage: python prep_data.py {ami,alimeeting,aishell4,msdwild} [--splits ...]
"""
import argparse
import glob
import io
import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

ROOT = "/mnt/ssd2/paper/paper_1/data"
RAW = f"{ROOT}/raw"
FA = f"{ROOT}/refs/diar-forced-alignment"


def write_rttm(segs, uid, path):
    """segs: list of (start, dur, spk)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s, d, k in sorted(segs):
            if d <= 0:
                continue
            f.write(f"SPEAKER {uid} 1 {s:.3f} {d:.3f} <NA> <NA> {k} <NA> <NA>\n")


def read_rttm(path):
    segs = []
    for line in open(path):
        p = line.split()
        if not p or p[0] != "SPEAKER":
            continue
        segs.append((float(p[3]), float(p[4]), p[7]))
    return segs


def manifest_line(wav, rttm):
    info = sf.info(wav)
    nspk = len({k for _, _, k in read_rttm(rttm)})
    return {
        "audio_filepath": wav,
        "offset": 0.0,
        "duration": round(info.frames / info.samplerate, 3),
        "label": "infer",
        "text": "-",
        "num_speakers": nspk,
        "rttm_filepath": rttm,
        "uem_filepath": None,
    }


def write_manifest(name, pairs):
    os.makedirs(f"{ROOT}/manifests", exist_ok=True)
    lines = [manifest_line(w, r) for w, r in sorted(pairs)]
    with open(f"{ROOT}/manifests/{name}.jsonl", "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    hours = sum(l["duration"] for l in lines) / 3600
    print(f"{name}: {len(lines)} recordings, {hours:.1f} h")


def ffmpeg_ch0(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return dst
    tmp = dst + ".tmp.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-af", "pan=mono|c0=c0", "-ar", "16000",
         "-c:a", "pcm_s16le", tmp],
        check=True,
    )
    os.replace(tmp, dst)
    return dst


# ---------------------------------------------------------------- AMI
def ami(args):
    import pyarrow.parquet as pq

    split_map = {"train": "train", "validation": "dev", "test": "test"}
    cond_map = {"ihm": "mhm", "sdm": "sdm"}
    pairs = defaultdict(list)
    for cfg, cond in cond_map.items():
        for pf in sorted(glob.glob(f"{RAW}/ami_hf/{cfg}/*.parquet")):
            split = split_map[os.path.basename(pf).split("-")[0]]
            setname = f"ami_{split}_{cond}"
            f = pq.ParquetFile(pf)
            for rg in range(f.num_row_groups):
                tab = f.read_row_group(rg, columns=["audio"])
                for a in tab.column("audio").to_pylist():
                    mid = a["path"].split(".")[0]
                    ref = f"{FA}/AMI/{split}/{mid}.rttm"
                    if not os.path.exists(ref):
                        print("no FA ref for", mid, split)
                        continue
                    wav = f"{ROOT}/wav/{setname}/{mid}.wav"
                    if not os.path.exists(wav):
                        x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="int16")
                        assert sr == 16000, sr
                        if x.ndim > 1:
                            x = x[:, 0]
                        os.makedirs(os.path.dirname(wav), exist_ok=True)
                        sf.write(wav, x, 16000, subtype="PCM_16")
                    rttm = f"{ROOT}/rttm/{setname}/{mid}.rttm"
                    write_rttm(read_rttm(ref), mid, rttm)
                    pairs[setname].append((wav, rttm))
    for name, p in pairs.items():
        write_manifest(name, p)


# ---------------------------------------------------------------- AliMeeting
def _near_mix(session, files, dst):
    if os.path.exists(dst):
        return dst
    sigs = [sf.read(f, dtype="float32")[0] for f in files]
    sigs = [s if s.ndim == 1 else s[:, 0] for s in sigs]
    n = max(len(s) for s in sigs)
    mix = np.zeros(n, dtype=np.float32)
    for s in sigs:
        mix[: len(s)] += s
    peak = np.abs(mix).max()
    if peak > 0.99:
        mix *= 0.99 / peak
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    sf.write(dst, mix, 16000, subtype="PCM_16")
    return dst


def alimeeting(args):
    jobs = []
    pairs = defaultdict(list)
    with ProcessPoolExecutor(8) as ex:
        for split, src in [("test", "Test_Ali"), ("eval", "Eval_Ali"), ("train", "Train_Ali")]:
            fa_dir = f"{FA}/AliMeeting/{src}_far"
            if not os.path.isdir(fa_dir):
                continue
            far_dir = f"{RAW}/alimeeting/{src}/{src}_far/audio_dir"
            near_dir = f"{RAW}/alimeeting/{src}/{src}_near/audio_dir"
            if split == "train":
                near_dir = f"{RAW}/alimeeting_hf/Train_Ali_near/audio_dir"
            if split == "test":  # HF mirror ggfox00000/dia-alimeeting-test
                far_dir = f"{RAW}/alimeeting_test_hf/audio/far"
                near_dir = f"{RAW}/alimeeting_test_hf/audio/near"
            for ref in sorted(glob.glob(f"{fa_dir}/*.rttm")):
                sess = os.path.basename(ref)[:-5]
                far = glob.glob(f"{far_dir}/{sess}_*.wav")
                near = sorted(glob.glob(f"{near_dir}/{sess}_*SPK*.wav"))
                for cond, ok in [("far", bool(far)), ("near", bool(near))]:
                    if not ok:
                        continue
                    setname = f"ali_{split}_{cond}"
                    wav = f"{ROOT}/wav/{setname}/{sess}.wav"
                    if cond == "far":
                        jobs.append(ex.submit(ffmpeg_ch0, far[0], wav))
                    else:
                        jobs.append(ex.submit(_near_mix, sess, near, wav))
                    rttm = f"{ROOT}/rttm/{setname}/{sess}.rttm"
                    write_rttm(read_rttm(ref), sess, rttm)
                    pairs[setname].append((wav, rttm))
        for j in jobs:
            j.result()
    for name, p in pairs.items():
        write_manifest(name, p)


# ---------------------------------------------------------------- AISHELL-4
def aishell4(args):
    pairs = defaultdict(list)
    jobs = []
    with ProcessPoolExecutor(8) as ex:
        for split in ["test", "train_L", "train_M", "train_S"]:
            setname = "aishell4_test" if split == "test" else "aishell4_train"
            for flac in sorted(glob.glob(f"{RAW}/aishell4/{split}/wav/*.flac")):
                uid = os.path.basename(flac)[:-5]
                ref = f"{RAW}/aishell4/{split}/TextGrid/{uid}.rttm"
                if not os.path.exists(ref):
                    print("missing ref", ref)
                    continue
                wav = f"{ROOT}/wav/{setname}/{uid}.wav"
                jobs.append(ex.submit(ffmpeg_ch0, flac, wav))
                rttm = f"{ROOT}/rttm/{setname}/{uid}.rttm"
                write_rttm(read_rttm(ref), uid, rttm)
                pairs[setname].append((wav, rttm))
        for j in jobs:
            j.result()
    for name, p in pairs.items():
        write_manifest(name, p)


# ---------------------------------------------------------------- MSDWild
def msdwild(args):
    pairs = defaultdict(list)
    for sub, rttm_name in [("many", "many.val"), ("few", "few.val"), ("train", "few.train")]:
        setname = f"msdwild_{sub}"
        refs = defaultdict(list)
        for line in open(f"{ROOT}/refs/msdwild/{rttm_name}.rttm"):
            p = line.split()
            refs[p[1]].append((float(p[3]), float(p[4]), "S" + p[7]))
        for uid, segs in sorted(refs.items()):
            src = f"{RAW}/msdwild/audio/{sub}/{uid}.wav"
            if sub == "train":
                src = f"{RAW}/msdwild_all/raw_wav/{uid}.wav"
            if not os.path.exists(src):
                print("missing audio", src)
                continue
            wav = f"{ROOT}/wav/{setname}/{uid}.wav"
            if not os.path.exists(wav):
                x, sr = sf.read(src, dtype="float32")
                if x.ndim > 1:
                    x = x[:, 0]
                if sr != 16000:
                    ffmpeg_ch0(src, wav)
                else:
                    os.makedirs(os.path.dirname(wav), exist_ok=True)
                    sf.write(wav, x, 16000, subtype="PCM_16")
            rttm = f"{ROOT}/rttm/{setname}/{uid}.rttm"
            write_rttm(segs, uid, rttm)
            pairs[setname].append((wav, rttm))
    for name, p in pairs.items():
        write_manifest(name, p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=["ami", "alimeeting", "aishell4", "msdwild"])
    a = ap.parse_args()
    globals()[a.corpus](a)
