"""Batched streaming inference; saves 10 ms speaker probabilities and activity logits per file.

python infer.py --model base|<ckpt.nemo|.ckpt> --manifest data/manifests/ami_test_mhm.jsonl \
    --profile 1.04 --out exp/<name>/<set>/<profile> [--oracle-cache] [--sfx-json '{}']
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_NEMO = "/mnt/ssd2/paper/paper_1/models/base/Nemotron-3-Diarization.nemo"
PROFILES = {  # spkcache, fifo, chunk, right_ctx, update_period (80 ms frames)
    "30.4": (264, 40, 340, 40, 300),
    "1.04": (264, 264, 9, 4, 222),
    "0.64": (264, 264, 6, 2, 222),
    "0.32": (264, 264, 3, 1, 222),
}


def set_profile(model, profile, spkcache=None, fifo_len=None, update=None):
    sc, fifo, chunk, rc, upd = PROFILES[profile]
    fifo = fifo_len or fifo
    upd = update or upd
    sm = model.sortformer_modules
    sm.spkcache_len = spkcache or sc
    sm.fifo_len, sm.chunk_len, sm.chunk_right_context, sm.spkcache_update_period = fifo, chunk, rc, upd
    sm.chunk_left_context = 0
    sm._check_streaming_parameters()


def read_rttm(path):
    segs = []
    for line in open(path):
        p = line.split()
        if p and p[0] == "SPEAKER":
            segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return segs


def oracle_matrix(segs, n_frames80, n_spk=8):
    """80 ms reference activity in arrival order."""
    first = {}
    for s, e, k in segs:
        first[k] = min(first.get(k, 1e9), s)
    order = sorted(first, key=first.get)[:n_spk]
    n10 = n_frames80 * 8
    m = np.zeros((n10, n_spk), dtype=np.float32)
    for s, e, k in segs:
        if k in order:
            m[int(round(s * 100)) : int(round(e * 100)), order.index(k)] = 1
    m = m.reshape(n_frames80, 8, n_spk).mean(1)
    return (m > 0.5).astype(np.float32)


def load_model(spec, sfx_json, cache_rescore, device):
    from sfx.model import SFXModel, build_sfx_from_base

    import sfx  # noqa: F401
    if spec == "base":
        model = build_sfx_from_base(BASE_NEMO, json.loads(sfx_json or "{}"), cache_rescore=cache_rescore)
    elif spec == "stock":
        from nemo.collections.asr.models import SortformerEncLabelModel

        model = SortformerEncLabelModel.restore_from(BASE_NEMO, map_location="cpu")
    else:
        model = SFXModel.restore_from(spec, map_location="cpu")
        model.sortformer_modules.cache_rescore = cache_rescore if cache_rescore else model.sortformer_modules.cache_rescore
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base")
    ap.add_argument("--sfx-json", default="{}")
    ap.add_argument("--cache-rescore", action="store_true")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--profile", default="1.04", choices=list(PROFILES))
    ap.add_argument("--spkcache", type=int, default=None)
    ap.add_argument("--fifo", type=int, default=None)
    ap.add_argument("--update", type=int, default=None)
    ap.add_argument("--strong-rate", type=float, default=None)
    ap.add_argument("--weak-rate", type=float, default=None)
    ap.add_argument("--sil-frames", type=int, default=None)
    ap.add_argument("--adaptive", default=None, help="beta,mmin,mmax (speaker-adaptive cache budget)")
    ap.add_argument("--oracle-cache", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bf16", type=int, default=1)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    model = load_model(args.model, args.sfx_json, args.cache_rescore, device)
    set_profile(model, args.profile, args.spkcache, args.fifo, args.update)
    sm = model.sortformer_modules
    if args.strong_rate is not None:
        sm.strong_boost_rate = args.strong_rate
    if args.weak_rate is not None:
        sm.weak_boost_rate = args.weak_rate
    if args.sil_frames is not None:
        sm.spkcache_sil_frames_per_spk = args.sil_frames
    if args.adaptive:
        b, lo, hi = [int(v) for v in args.adaptive.split(",")]
        sm.adaptive_budget = {"beta": b, "mmin": lo, "mmax": hi}
        sm.spkcache_len = lo
    os.makedirs(args.out, exist_ok=True)

    items = [json.loads(l) for l in open(args.manifest)]
    if args.max_files:
        items = items[: args.max_files]
    items = [it for it in items if not os.path.exists(os.path.join(args.out, os.path.basename(it["audio_filepath"])[:-4] + ".npz"))]
    items.sort(key=lambda it: -it["duration"])
    t_audio, t0 = 0.0, time.time()
    for bi in range(0, len(items), args.batch_size):
        batch = items[bi : bi + args.batch_size]
        sigs = [sf.read(it["audio_filepath"], dtype="float32")[0] for it in batch]
        lens = [len(s) for s in sigs]
        x = np.zeros((len(sigs), max(lens)), dtype=np.float32)
        for i, s in enumerate(sigs):
            x[i, : len(s)] = s
        xa = torch.from_numpy(x).to(device)
        la = torch.tensor(lens, device=device)
        if args.oracle_cache:
            n80 = int(np.ceil(max(lens) / 1280)) + 2
            tg = np.stack([oracle_matrix(read_rttm(it["rttm_filepath"]), n80) for it in batch])
            model.sortformer_modules.oracle_targets = torch.from_numpy(tg).to(device)
        torch.cuda.synchronize()
        tb = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.bf16)):
            preds, logits, act = model(audio_signal=xa, audio_signal_length=la, return_logits=True)
        torch.cuda.synchronize()
        model.sortformer_modules.oracle_targets = None if hasattr(model.sortformer_modules, "oracle_targets") else None
        dt = time.time() - tb
        t_audio += sum(lens) / 16000
        for i, it in enumerate(batch):
            n = int(np.ceil(lens[i] / 160))
            uid = os.path.basename(it["audio_filepath"])[:-4]
            np.savez_compressed(
                os.path.join(args.out, uid + ".npz"),
                probs=preds[i, :n].float().cpu().numpy().astype(np.float16),
                act=(act[i, :n].float().cpu().numpy().astype(np.float16) if act is not None else np.zeros((0, 3))),
            )
        print(f"batch {bi // args.batch_size}: {len(batch)} files, {sum(lens) / 16000:.0f}s audio, {dt:.1f}s, RTFx {sum(lens) / 16000 / dt:.0f}", flush=True)
    with open(os.path.join(args.out, "_timing.json"), "a") as f:
        f.write(json.dumps({"audio_s": t_audio, "wall_s": time.time() - t0, "batch_size": args.batch_size}) + "\n")


if __name__ == "__main__":
    main()
