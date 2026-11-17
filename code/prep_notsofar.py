"""NOTSOFAR-1 evaluation set (240629.1_eval_small_with_GT), single-channel track: an untouched test corpus.

Evaluation protocol fixed before any system was scored:
- audio: channel 0 of both single-channel devices of each of the 80 meetings (two views of one meeting, resampled
  to 16 kHz mono if needed);
- references: official word timings, merged per speaker when the gap between consecutive words is below 0.2 s,
  consistent with the word-aligned references used for AMI and AliMeeting;
- recording ids <meeting>_<device>, so that bootstrap.py --group can resample the two views of a meeting together.
The released model was trained on the NOTSOFAR-1 train and development sets, not on this evaluation set.
Writes data/wav/notsofar_eval_sc, data/rttm/notsofar_eval_sc and data/manifests/notsofar_eval_sc.jsonl.
"""
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prep_data import ROOT, RAW, write_manifest  # noqa: E402

SRC = f"{RAW}/notsofar/benchmark-datasets/eval_set/240629.1_eval_small_with_GT/MTG"
GAP = 0.2


def main():
    import soundfile as sf
    wdir, rdir = f"{ROOT}/wav/notsofar_eval_sc", f"{ROOT}/rttm/notsofar_eval_sc"
    os.makedirs(wdir, exist_ok=True)
    os.makedirs(rdir, exist_ok=True)
    pairs, nspk = [], []
    for mdir in sorted(glob.glob(f"{SRC}/MTG_*")):
        mid = os.path.basename(mdir)
        words = {}
        for seg in json.load(open(f"{mdir}/gt_transcription.json")):
            for w in seg.get("word_timing") or []:
                words.setdefault(seg["speaker_id"], []).append((float(w[1]), float(w[2])))
        merged = []
        for spk, ws in words.items():
            ws.sort()
            cur = list(ws[0])
            for s, e in ws[1:]:
                if s - cur[1] < GAP:
                    cur[1] = max(cur[1], e)
                else:
                    merged.append((cur[0], cur[1], spk))
                    cur = [s, e]
            merged.append((cur[0], cur[1], spk))
        for dev in sorted(glob.glob(f"{mdir}/sc_*/ch0.wav")):
            dname = os.path.basename(os.path.dirname(dev))
            uid = f"{mid}_{dname}"
            out = f"{wdir}/{uid}.wav"
            info = sf.info(dev)
            if not os.path.exists(out):
                if info.samplerate == 16000 and info.channels == 1:
                    os.symlink(dev, out)
                else:
                    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", dev, "-ac", "1", "-ar", "16000",
                                    "-c:a", "pcm_s16le", out], check=True)
            dur = sf.info(out).duration
            with open(f"{rdir}/{uid}.rttm", "w") as f:
                for s, e, k in sorted(merged):
                    e = min(e, dur)
                    if e - s > 0:
                        f.write(f"SPEAKER {uid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {k} <NA> <NA>\n")
            pairs.append((out, f"{rdir}/{uid}.rttm"))
            nspk.append(len(words))
    write_manifest("notsofar_eval_sc", pairs)
    print("speakers per meeting: min", min(nspk), "max", max(nspk), "mean", round(sum(nspk) / len(nspk), 2))


if __name__ == "__main__":
    main()
