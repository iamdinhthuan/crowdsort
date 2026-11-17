"""NOTSOFAR-1 training and development sets (single-channel devices) as an extra fine-tuning source, with the
forced-alignment references of FastMSS (notsofar1-channels_mfa_rttms.tar.gz, splits train and dev), the data the
released model was trained on. Used to test whether adding them back removes the NOTSOFAR-1 regression.

Device codes as in prep_notsofar_fa.py. Writes data/wav/notsofar_train_sc, data/rttm/notsofar_train_sc and
data/manifests/notsofar_train_sc.jsonl (recordings without an MFA reference are skipped).

tar -xzf vendor/FastMSS/resources/notsofar1-channels_mfa_rttms.tar.gz -C data/raw/notsofar_mfa train/ dev/
python prep_notsofar_train.py
"""
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prep_data import ROOT, RAW, write_manifest  # noqa: E402
from prep_notsofar_fa import CODE  # noqa: E402

SETS = [f"{RAW}/notsofar_train/benchmark-datasets/train_set/240825.1_train/MTG",
        f"{RAW}/notsofar_train/benchmark-datasets/dev_set/240825.1_dev1/MTG"]


def main():
    import soundfile as sf
    wdir, rdir = f"{ROOT}/wav/notsofar_train_sc", f"{ROOT}/rttm/notsofar_train_sc"
    os.makedirs(wdir, exist_ok=True)
    os.makedirs(rdir, exist_ok=True)
    pairs, skipped = [], 0
    for base in SETS:
        split = "train" if "train_set" in base else "dev"
        for dev in sorted(glob.glob(f"{base}/MTG_*/sc_*/ch0.wav")):
            mid = os.path.basename(os.path.dirname(os.path.dirname(dev)))[4:]
            dname = os.path.basename(os.path.dirname(dev))[3:]
            ref = f"{RAW}/notsofar_mfa/{split}/S{mid}{CODE[dname]}.rttm"
            if not os.path.exists(ref):
                skipped += 1
                continue
            uid = f"MTG_{mid}_sc_{dname}"
            out = f"{wdir}/{uid}.wav"
            if not os.path.exists(out):
                info = sf.info(dev)
                if info.samplerate == 16000 and info.channels == 1:
                    os.symlink(os.path.abspath(dev), out)
                else:
                    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", dev, "-ac", "1", "-ar", "16000",
                                    "-c:a", "pcm_s16le", out], check=True)
            dur = sf.info(out).duration
            with open(ref) as fi, open(f"{rdir}/{uid}.rttm", "w") as fo:
                for l in fi:
                    p = l.split()
                    if p[0] != "SPEAKER":
                        continue
                    s, e = float(p[3]), min(float(p[3]) + float(p[4]), dur)
                    if e > s:
                        fo.write(f"SPEAKER {uid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {p[7]} <NA> <NA>\n")
            pairs.append((out, f"{rdir}/{uid}.rttm"))
    write_manifest("notsofar_train_sc", pairs)
    print("skipped without reference:", skipped)


if __name__ == "__main__":
    main()
