"""AVA-AVD test (audio only), an in-the-wild many-speaker set that is not in the N3D training data.

Annotations: official AVA-AVD release (labs/, rttms/, split/test.list; github.com/zcxu-eric/AVA-AVD).
Audio: the 900-1800 s region of each AVA movie (s3.amazonaws.com/ava-dataset/trainval), read with ffmpeg
over HTTP, mono 16 kHz. Clip c_k covers [900 + 300 (k - 1), 900 + 300 k) s of the movie; RTTM times are
absolute movie times and are shifted to the clip start.
Writes data/wav/avaavd_test, data/rttm/avaavd_test, data/manifests/avaavd_test.jsonl.
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from prep_data import ROOT, RAW, read_rttm, write_manifest  # noqa: E402

SRC = f"{RAW}/avaavd"
URL = "https://s3.amazonaws.com/ava-dataset/trainval"


def main():
    names = {l.split(".")[0]: l.strip() for l in open(f"{SRC}/video.list") if l.strip()}
    clips = [l.strip() for l in open(f"{SRC}/test.list") if l.strip()]
    os.makedirs(f"{SRC}/audio_full", exist_ok=True)
    wdir, rdir = f"{ROOT}/wav/avaavd_test", f"{ROOT}/rttm/avaavd_test"
    os.makedirs(wdir, exist_ok=True)
    os.makedirs(rdir, exist_ok=True)
    pairs = []
    for vid in sorted({c.rsplit("_c_", 1)[0] for c in clips}):
        full = f"{SRC}/audio_full/{vid}.wav"
        if not os.path.exists(full):
            tmp = full + ".tmp.wav"
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", "900", "-t", "900", "-i",
                            f"{URL}/{names[vid]}", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp],
                           check=True)
            os.rename(tmp, full)
    for c in clips:
        vid, k = c.rsplit("_c_", 1)
        t0 = 900 + 300 * (int(k) - 1)
        x, sr = sf.read(f"{SRC}/audio_full/{vid}.wav", dtype="int16")
        seg = x[(t0 - 900) * sr:(t0 - 900 + 300) * sr]
        sf.write(f"{wdir}/{c}.wav", seg, sr, subtype="PCM_16")
        dur = len(seg) / sr
        with open(f"{rdir}/{c}.rttm", "w") as f:
            for s, d, spk in sorted(read_rttm(f"{SRC}/rttms/{c}.rttm")):
                s0, e0 = max(s - t0, 0.0), min(s + d - t0, dur)
                if e0 > s0:
                    f.write(f"SPEAKER {c} 1 {s0:.3f} {e0 - s0:.3f} <NA> <NA> {spk} <NA> <NA>\n")
        pairs.append((f"{wdir}/{c}.wav", f"{rdir}/{c}.rttm"))
    write_manifest("avaavd_test", pairs)
    # alignment check: labelled speech must carry more energy than unlabelled regions
    ratios = []
    for w, r in pairs:
        x, sr = sf.read(w, dtype="float32")
        e = np.array([np.mean(x[i:i + 160] ** 2) for i in range(0, len(x) - 160, 160)]) + 1e-10
        lab = np.zeros(len(e), bool)
        for s, d, _ in read_rttm(r):
            lab[int(s * 100):int((s + d) * 100)] = True
        if lab.any() and (~lab).any():
            ratios.append(10 * np.log10(np.median(e[lab]) / np.median(e[~lab])))
    print(f"speech/non-speech median energy ratio: mean {np.mean(ratios):.1f} dB, min {np.min(ratios):.1f} dB")


if __name__ == "__main__":
    main()
