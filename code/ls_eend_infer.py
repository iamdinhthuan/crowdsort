"""External streaming baseline: LS-EEND (Liang and Li, IEEE/ACM TASLP 2025) with its released checkpoints.

The model is causal with a 9-frame (0.9 s) look-ahead at 100 ms frames, so its batch output equals its
frame-by-frame streaming output (checked on one file against the authors' streaming loop, max difference 3.7e-4).
Audio is resampled from 16 kHz to 8 kHz as in the original recipe. Probabilities are repeated to the 10 ms grid
and saved in the npz format of infer.py, so that score.py scores them with the same pipeline.

python ls_eend_infer.py --ckpt vendor/FS-EEND/LS-EEND/ckpt_dl/dih3.ckpt \
    --config vendor/FS-EEND/LS-EEND/conf/spk_onl_conformer_retention_enc_dec_nonautoreg_dihard3_infer.yaml \
    --manifest data/manifests/msdwild_many.jsonl --out exp/ls_eend_dih3/msdwild_many/1.0
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = "/mnt/ssd2/paper/paper_1"
LS = f"{ROOT}/vendor/FS-EEND/LS-EEND"
sys.path.insert(0, LS)
import importlib.util  # noqa: E402

import hyperpyyaml  # noqa: E402

# load LS-EEND's datasets/feature.py by path: its folder name clashes with the Hugging Face "datasets" package
_spec = importlib.util.spec_from_file_location("lseend_feature", f"{LS}/datasets/feature.py")
_feat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feat)
splice, stft, transform = _feat.splice, _feat.stft, _feat.transform
from nnet.model.onl_conformer_retention_enc_1dcnn_tfm_retention_enc_linear_non_autoreg_pos_enc_l2norm_emb_loss_mask import (  # noqa: E402
    OnlineConformerRetentionDADiarization,
    StreamingConv1d,
)


# frame-by-frame streaming loop, copied from the authors' streaming_infer_dia.py (commit adcdde1); used when the
# parallel (batch) form does not fit in GPU memory for long recordings. Both forms give the same output.
def build_streaming_cnn(model, device):
    streaming_cnn = StreamingConv1d(model.n_units, model.n_units, kernel_size=2 * model.delay + 1).to(device)
    streaming_cnn.conv.load_state_dict(model.cnn.state_dict())
    streaming_cnn.eval()
    return streaming_cnn


def streaming_predict(model, streaming_cnn, feat, max_nspks, device):
    n_enc, n_dec = len(model.enc.encoder.layers), len(model.dec.layers)
    k = model.enc.encoder._conv_kernel_size
    ret_states = [dict() for _ in range(n_enc)]
    conv_caches = [torch.zeros(1, model.n_units, k - 1, device=device) for _ in range(n_enc)]
    dec_states = [dict() for _ in range(n_dec)]
    streaming_cnn.buffer.clear()
    streaming_cnn.t = 0
    preds, dec_t = [], 0

    def step(emb_t, dt):
        e = streaming_cnn(emb_t.transpose(1, 2))
        if e is None:
            return None, dt
        e = e.transpose(1, 2)
        e = e / torch.norm(e, dim=-1, keepdim=True)
        att = model.dec.forward_one_step(e, dt, max_nspks, dec_states)
        att = att / torch.norm(att, dim=-1, keepdim=True)
        return torch.matmul(e.unsqueeze(dim=-2), att.transpose(-1, -2)).squeeze(dim=-2), dt + 1

    for t in range(feat.shape[0]):
        emb_t = model.enc.forward_one_step(feat[t:t + 1].unsqueeze(0), t, ret_states, conv_caches)
        y, dec_t = step(emb_t, dec_t)
        if y is not None:
            preds.append(y)
    for _ in range(model.delay):
        y, dec_t = step(torch.zeros(1, 1, model.n_units, device=device), dec_t)
        if y is not None:
            preds.append(y)
    return torch.cat(preds, dim=1).squeeze(0)


def features(wav16, cfg):
    x = resample_poly(wav16, 1, 2).astype(np.float32)
    d = cfg["data"]
    Y = stft(x, d["feat"]["win_length"], d["feat"]["hop_length"])
    Y = transform(Y, d["feat_type"])
    Y = splice(Y, d["context_recp"])[::d["subsampling"]]
    return torch.from_numpy(Y).float()


def load_model(ckpt, cfg, device):
    in_size = (2 * cfg["data"]["context_recp"] + 1) * cfg["data"]["feat"]["n_mels"]
    model = OnlineConformerRetentionDADiarization(n_speakers=cfg["data"]["num_speakers"], in_size=in_size,
                                                  **cfg["model"]["params"]).to(device)
    import collections
    # safe unpickling (weights_only) with the one extra container type used by the released checkpoints
    torch.serialization.add_safe_globals([collections.defaultdict, float, int, dict, list])
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k.replace("dec.attractor_decoder.layers.", "dec.layers."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, missing[:5]
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = hyperpyyaml.load_hyperpyyaml(open(a.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.ckpt, cfg, device)
    max_nspks = cfg["data"]["max_speakers"] + 2
    os.makedirs(a.out, exist_ok=True)
    items = [json.loads(l) for l in open(a.manifest)]
    scnn = None
    for it in items:
        uid = os.path.basename(it["audio_filepath"])[:-4]
        outf = f"{a.out}/{uid}.npz"
        if os.path.exists(outf):
            continue
        wav, sr = sf.read(it["audio_filepath"], dtype="float32")
        assert sr == 16000
        feat = features(wav, cfg).to(device)
        with torch.no_grad():
            try:
                logits = model.test([feat], [len(feat)], max_nspks=max_nspks)[0][0]
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if scnn is None:
                    scnn = build_streaming_cnn(model, device)
                logits = streaming_predict(model, scnn, feat, max_nspks, device)
        p = torch.sigmoid(logits[:, 1:]).float().cpu().numpy()      # (frames of 100 ms, speakers), as in the authors' code
        n10 = int(round(len(wav) / 160))
        p10 = np.repeat(p, 10, axis=0)[:n10]
        if len(p10) < n10:
            p10 = np.pad(p10, ((0, n10 - len(p10)), (0, 0)))
        np.savez_compressed(outf, probs=p10.astype(np.float16), act=np.zeros((n10, 3), np.float16), p100=p.astype(np.float16))
    print("done", a.out, len(items))


if __name__ == "__main__":
    main()
