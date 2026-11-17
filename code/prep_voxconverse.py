"""VoxConverse test (cross-corpus, never used for training or tuning).

Audio: Hugging Face mirror ggfox00000/dia-voxconverse-test (16 kHz mono wav).
References: official RTTMs from github.com/joonson/voxconverse (test, 232 recordings).
Writes data/wav/voxconverse_test (symlinks), data/rttm/voxconverse_test and
data/manifests/voxconverse_test.jsonl. References that overrun the audio by up to 0.15 s (22 official
RTTMs end 0.06-0.09 s after the audio) are clipped to the audio duration; larger overruns are rejected.
"""
import glob, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from prep_data import ROOT, RAW, read_rttm, write_manifest  # noqa: E402

SRC_WAV = f"{RAW}/voxconverse/ggfox/audio/test"
SRC_RTTM = f"{RAW}/voxconverse/rttm_repo/test"


def main():
    import soundfile as sf
    wdir, rdir = f"{ROOT}/wav/voxconverse_test", f"{ROOT}/rttm/voxconverse_test"
    os.makedirs(wdir, exist_ok=True)
    os.makedirs(rdir, exist_ok=True)
    rttms = sorted(glob.glob(f"{SRC_RTTM}/*.rttm"))
    pairs, bad = [], []
    for r in rttms:
        uid = os.path.basename(r)[:-5]
        src = f"{SRC_WAV}/{uid}.wav"
        if not os.path.exists(src):
            bad.append((uid, "missing audio"))
            continue
        info = sf.info(src)
        dur = info.frames / info.samplerate
        segs = read_rttm(r)
        end = max(s + d for s, d, _ in segs)
        if info.samplerate != 16000 or info.channels != 1 or end > dur + 0.15:
            bad.append((uid, f"sr={info.samplerate} ch={info.channels} ref_end={end:.2f} dur={dur:.2f}"))
            continue
        dst = f"{wdir}/{uid}.wav"
        if not os.path.exists(dst):
            os.symlink(src, dst)
        with open(f"{rdir}/{uid}.rttm", "w") as f:
            for s, d, k in sorted(segs):
                d = min(d, dur - s)
                if d > 0:
                    f.write(f"SPEAKER {uid} 1 {s:.3f} {d:.3f} <NA> <NA> {k} <NA> <NA>\n")
        pairs.append((dst, f"{rdir}/{uid}.rttm"))
    print(f"references {len(rttms)}, usable {len(pairs)}, rejected {len(bad)}")
    for b in bad:
        print("REJECT", *b)
    write_manifest("voxconverse_test", pairs)


if __name__ == "__main__":
    main()
