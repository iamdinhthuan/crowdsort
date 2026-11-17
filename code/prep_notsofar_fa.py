"""NOTSOFAR-1 single-channel evaluation with the forced-alignment references used for the published results of the
released model (FastMSS resources/notsofar1-channels_mfa_rttms.tar.gz, split eval; one RTTM per device recording).

Session ids S<meeting><code>: device codes 101 plaza_0, 201 rockfall_0, 211 rockfall_1, 221 rockfall_2, 301 meetup_0,
401 studio_0, 501 panacast_0 (each device name co-occurs with exactly one code over the 80 meetings).
Same audio as notsofar_eval_sc; writes data/rttm/notsofar_eval_sc_fa and data/manifests/notsofar_eval_sc_fa.jsonl,
and links the saved predictions of every scored system into exp/<model>/notsofar_eval_sc_fa/<profile> so that
rescore.py can score them against these references without new inference.

tar -xzf vendor/FastMSS/resources/notsofar1-channels_mfa_rttms.tar.gz -C data/raw/notsofar_mfa eval/
python prep_notsofar_fa.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prep_data import ROOT, RAW, write_manifest  # noqa: E402

CODE = {"plaza_0": "101", "rockfall_0": "201", "rockfall_1": "211", "rockfall_2": "221", "meetup_0": "301",
        "studio_0": "401", "panacast_0": "501"}
EXP = "/mnt/ssd2/paper/paper_1/exp"


def main():
    import soundfile as sf
    rdir = f"{ROOT}/rttm/notsofar_eval_sc_fa"
    os.makedirs(rdir, exist_ok=True)
    pairs = []
    for w in sorted(glob.glob(f"{ROOT}/wav/notsofar_eval_sc/*.wav")):
        uid = os.path.basename(w)[:-4]                      # MTG_32000_sc_meetup_0
        mid, dev = uid.split("_sc_")
        sid = f"S{mid[4:]}{CODE[dev]}"
        src = f"{RAW}/notsofar_mfa/eval/{sid}.rttm"
        dur = sf.info(w).duration
        with open(src) as fi, open(f"{rdir}/{uid}.rttm", "w") as fo:
            for l in fi:
                p = l.split()
                if p[0] != "SPEAKER":
                    continue
                s, d = float(p[3]), float(p[4])
                e = min(s + d, dur)
                if e > s:
                    fo.write(f"SPEAKER {uid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {p[7]} <NA> <NA>\n")
        pairs.append((w, f"{rdir}/{uid}.rttm"))
    write_manifest("notsofar_eval_sc_fa", pairs)
    n = 0
    for src in glob.glob(f"{EXP}/*/notsofar_eval_sc/*"):
        dst = src.replace("/notsofar_eval_sc/", "/notsofar_eval_sc_fa/")
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(f"{src}/*.npz"):
            t = f"{dst}/{os.path.basename(f)}"
            if not os.path.lexists(t):
                os.symlink(f, t)
        n += 1
    print("linked prediction dirs:", n)


if __name__ == "__main__":
    main()
