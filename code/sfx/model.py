"""SFX: Streaming Sortformer with slot-aware cache conditioning, speaker prototype memory and
high-resolution boundary refinement. Built on Nemotron 3 Diarization (NeMo SortformerEncLabelModel).

All new parameters are zero-initialised so that, before fine-tuning, an SFX model with every
feature enabled reproduces the base model exactly except for the inserted prototype tokens
(prototype tokens are the only feature that changes the input sequence at initialisation).

cfg.sfx keys (all default off):
  slot_emb      learned segment embeddings (compressed cache / history / chunk / prototype) and
                per-block slot embeddings for compressed-cache frames, added to pre-encode features.
  proto         insert one prototype token at the start of every speaker block of the compressed
                cache: running mean of the speaker's single-speaker frames + zero-init residual MLP.
  hr_skip       10 ms skip branch from the log-mel chunk features into the upsampled hidden states.
  cache_rescore (lives in sortformer_modules) re-score compressed-cache frames with fresh predictions.
"""
import math
import sfx  # noqa: F401  (SDPA attention patch)
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

SEG_CACHE, SEG_HIST, SEG_CHUNK, SEG_PROTO = 0, 1, 2, 3


class HRSkip(nn.Module):
    """Local conv stack on 10 ms log-mel features; output added to upsampled hidden states."""

    def __init__(self, n_mels: int, hidden: int, kernel: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(n_mels)
        self.conv1 = nn.Conv1d(n_mels, hidden, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, feats):  # (B, T, F) -> (B, T, H)
        x = self.norm(feats).transpose(1, 2)
        x = self.conv2(F.gelu(self.conv1(x)))
        return x.transpose(1, 2)


class ProtoMLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class DeepProto(nn.Module):
    """Maps the running mean of a speaker's contextual (encoder-output) states into the input space; zero-init."""

    def __init__(self, h: int, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(h)
        self.fc1 = nn.Linear(h, d)
        self.fc2 = nn.Linear(d, d)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class SFXModel(SortformerEncLabelModel):
    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
        sfx = cfg.get("sfx", {}) or {}
        self.sfx_slot = bool(sfx.get("slot_emb", False))
        self.sfx_proto = bool(sfx.get("proto", False))
        self.sfx_hr = bool(sfx.get("hr_skip", False))
        d = self._cfg.model_defaults.fc_d_model
        h = self._cfg.model_defaults.tf_d_model
        S = self._cfg.max_num_of_spks
        if self.sfx_slot:
            self.sfx_seg_emb = nn.Parameter(torch.zeros(4, d))
            self.sfx_slot_emb = nn.Parameter(torch.zeros(S + 1, d))
        self.sfx_proto_deep = bool(sfx.get("proto_deep", False))
        if self.sfx_proto:
            self.sfx_proto_mlp = ProtoMLP(d)
            self.sfx_proto_min_w = float(sfx.get("proto_min_w", 2.0))  # frames of evidence before a prototype is used
            if self.sfx_proto_deep:
                self.sfx_deep_mlp = DeepProto(h, d)
        if self.sfx_hr:
            self.sfx_hr_skip = HRSkip(self._cfg.preprocessor.features, h, kernel=int(sfx.get("hr_kernel", 5)))
        self.sfx_last_activity = None

    def training_step(self, batch, batch_idx):
        """Capacity-elastic training: sample the speaker-cache size per step when cfg.sfx.elastic_cache=[lo, hi]."""
        el = (self._cfg.get("sfx", {}) or {}).get("elastic_cache", None)
        if el:
            import random

            S = self.sortformer_modules.n_spk
            self.sortformer_modules.spkcache_len = S * random.randint(int(el[0]) // S, int(el[1]) // S)
        return super().training_step(batch, batch_idx)

    def setup_optimizer_param_groups(self):
        """New (sfx_*) parameters train with lr * cfg.sfx.lr_mult; everything else with the base lr."""
        mult = float((self._cfg.get("sfx", {}) or {}).get("lr_mult", 1.0))
        new = [p for n, p in self.named_parameters() if n.startswith("sfx_") and p.requires_grad]
        old = [p for n, p in self.named_parameters() if not n.startswith("sfx_") and p.requires_grad]
        groups = [{"params": old}]
        if new:
            groups.append({"params": new, "lr": self._cfg.optim.lr * mult})
        self._optimizer_param_groups = groups

    # ------------------------------------------------------------------ helpers
    def _seg(self, x, seg_id):
        if not self.sfx_slot:
            return x
        return x + self.sfx_seg_emb[seg_id].to(x.dtype)

    def _build_cache_input(self, st, B, S):
        """Return cache tokens to feed (B, Lc', D) and index of each original cache frame in it (B, Lc)."""
        cache = st.spkcache
        Lc = cache.shape[1]
        ar = torch.arange(Lc, device=cache.device).unsqueeze(0).expand(B, -1)
        if Lc == 0 or not st.spkcache_compressed:
            return self._seg(cache, SEG_HIST), ar
        slots = st.sfx_slots  # (B, Lc), values 0..S (S = disabled tail)
        x = cache
        if self.sfx_slot:
            x = x + self.sfx_seg_emb[SEG_CACHE].to(x.dtype) + self.sfx_slot_emb[slots].to(x.dtype)
        if not self.sfx_proto:
            return x, ar
        # prototype for model channel c = canonical speaker spk_perm[c]
        w = st.proto_w  # (B, S) canonical
        mean = st.proto_sum / w.clamp(min=1e-6).unsqueeze(-1)
        if st.spk_perm is not None:
            mean = torch.stack([mean[b, st.spk_perm[b]] for b in range(B)])
            w = torch.stack([w[b, st.spk_perm[b]] for b in range(B)])
        tok = self.sfx_proto_mlp(mean.to(x.dtype))
        if self.sfx_proto_deep and getattr(st, "proto_sum_deep", None) is not None:
            wd = st.proto_w_deep
            md = st.proto_sum_deep / wd.clamp(min=1e-6).unsqueeze(-1)
            if st.spk_perm is not None:
                md = torch.stack([md[b, st.spk_perm[b]] for b in range(B)])
            tok = tok + self.sfx_deep_mlp(md.to(x.dtype))
        sil = self.sortformer_modules.learnable_sil_emb.to(x.dtype).view(1, 1, -1)
        tok = torch.where((w >= self.sfx_proto_min_w).unsqueeze(-1), tok, sil.expand(B, S, -1))
        if self.sfx_slot:
            tok = tok + self.sfx_seg_emb[SEG_PROTO].to(x.dtype) + self.sfx_slot_emb[:S].to(x.dtype).unsqueeze(0)
        # insert prototype c right before block c
        blk = slots.clamp(max=S)  # tail frames shifted by S
        frame_pos = ar + torch.where(blk < S, blk + 1, torch.full_like(blk, S))
        counts = torch.stack([(slots < c).sum(1) for c in range(S)], dim=1)  # (B, S) frames before block c
        proto_pos = counts + torch.arange(S, device=x.device).unsqueeze(0)
        out = x.new_zeros(B, Lc + S, x.shape[2])
        out.scatter_(1, frame_pos.unsqueeze(-1).expand(-1, -1, x.shape[2]), x)
        out.scatter_(1, proto_pos.unsqueeze(-1).expand(-1, -1, x.shape[2]), tok)
        return out, frame_pos

    # ------------------------------------------------------------------ streaming step
    def forward_streaming_step(
        self,
        processed_signal,
        processed_signal_length,
        streaming_state,
        total_preds,
        drop_extra_pre_encoded: int = 0,
        left_offset: int = 0,
        right_offset: int = 0,
        return_logits: bool = False,
        total_logits: Optional[torch.Tensor] = None,
        total_activity_logits: Optional[torch.Tensor] = None,
        return_native_resolution: bool = False,
        update_state: bool = True,
    ):
        assert not self.async_streaming, "SFXModel supports synchronous streaming only"
        assert self.high_resolution, "SFXModel expects a high-resolution (10 ms) model"
        assert drop_extra_pre_encoded == 0
        st = streaming_state
        sm = self.sortformer_modules
        S = sm.n_spk
        up = self.upsample_factor

        chunk_embs, chunk_lens = self._call_pre_encode(processed_signal, processed_signal_length)
        B = chunk_embs.shape[0]
        cache_in, cache_frame_pos = self._build_cache_input(st, B, S)
        fifo_in = self._seg(st.fifo, SEG_HIST)
        chunk_in = self._seg(chunk_embs, SEG_CHUNK)
        Lc_in, Lf = cache_in.shape[1], fifo_in.shape[1]
        seq = torch.cat([cache_in, fifo_in, chunk_in], dim=1)
        seq_lens = Lc_in + Lf + chunk_lens

        enc, enc_lens = self.frontend_encoder(
            processed_signal=seq, processed_signal_length=seq_lens, bypass_pre_encode=True
        )
        enc_mask = sm.length_to_mask(enc_lens, enc.shape[1])
        if not self.training:
            enc = enc.masked_fill(~enc_mask.unsqueeze(-1), 0.0)
        hid = sm.upsample_hidden(enc)  # (B, 8L, H)
        if self.sfx_hr:
            T = processed_signal.shape[1]
            feats = processed_signal
            fmask = (torch.arange(T, device=feats.device).unsqueeze(0) < processed_signal_length.unsqueeze(1))
            feats = feats.masked_fill(~fmask.unsqueeze(-1), 0.0)
            skip = self.sfx_hr_skip(feats.to(hid.dtype)) * fmask.unsqueeze(-1)
            n_up = chunk_embs.shape[1] * up
            if skip.shape[1] < n_up:
                skip = F.pad(skip, (0, 0, 0, n_up - skip.shape[1]))
            skip = skip[:, :n_up]
            start = (Lc_in + Lf) * up
            hid = torch.cat([hid[:, :start], hid[:, start : start + n_up] + skip, hid[:, start + n_up :]], dim=1)
        logits = sm.forward_speaker_logits(hid)
        out_mask = enc_mask.repeat_interleave(up, dim=1)
        hr_preds = torch.sigmoid(logits) * out_mask.unsqueeze(-1)
        act_logits = sm.forward_activity_logits(hid) if (return_logits or not self.training) else None

        # 80 ms predictions for the state update, with prototype positions removed
        preds80 = sm.downsample_preds(hr_preds, up).detach()
        preds80 = sm.apply_mask_to_preds(preds80, enc_lens)
        if Lc_in != st.spkcache.shape[1]:
            cache_p = torch.gather(preds80, 1, cache_frame_pos.unsqueeze(-1).expand(-1, -1, S))
            preds80 = torch.cat([cache_p, preds80[:, Lc_in:]], dim=1)

        inv = None
        if st.spk_perm is not None:
            inv = torch.stack([torch.argsort(st.spk_perm[b]) for b in range(B)])

        lc_enc = round(left_offset / self.encoder.subsampling_factor)
        rc_enc = math.ceil(right_offset / self.encoder.subsampling_factor)
        chunk_len = chunk_embs.shape[1] - lc_enc - rc_enc
        start = (Lc_in + Lf + lc_enc) * up
        sl = slice(start, start + chunk_len * up)
        chunk_hr = hr_preds[:, sl]
        chunk_logits = logits[:, sl] if return_logits else None
        chunk_act = act_logits[:, sl] if act_logits is not None else None
        if inv is not None:
            chunk_hr = torch.stack([chunk_hr[b, :, inv[b]] for b in range(B)])
            if chunk_logits is not None:
                chunk_logits = torch.stack([chunk_logits[b, :, inv[b]] for b in range(B)])

        if self.sfx_proto and self.sfx_proto_deep:
            c0 = Lc_in + Lf + lc_enc
            enc_chunk = enc[:, c0 : c0 + chunk_len]
            c0p = st.spkcache.shape[1] + Lf + lc_enc  # preds80 holds the cache without prototype positions
            pc = preds80[:, c0p : c0p + chunk_len]
            if inv is not None:
                pc = torch.stack([pc[b, :, inv[b]] for b in range(B)])
            with torch.no_grad():
                p = pc.float().clamp(max=1 - 1e-4)
                others = torch.log1p(-p).sum(-1, keepdim=True) - torch.log1p(-p)
                w = p * torch.exp(others)
                w = torch.where(w > sm.proto_conf_floor, w, torch.zeros_like(w))
            if getattr(st, "proto_sum_deep", None) is None:
                st.proto_sum_deep = enc_chunk.new_zeros(B, S, enc_chunk.shape[2], dtype=torch.float32)
                st.proto_w_deep = w.new_zeros(B, S)
            # fp32 elementwise accumulation (not autocast) keeps long sessions exact
            st.proto_sum_deep = st.proto_sum_deep + (w.unsqueeze(-1) * enc_chunk.float().unsqueeze(2)).sum(1)
            st.proto_w_deep = st.proto_w_deep + w.sum(1)

        if update_state:
            st, _ = sm.streaming_update(streaming_state=st, chunk=chunk_embs, preds=preds80, lc=lc_enc, rc=rc_enc)

        native = 1 if self.high_resolution else self.encoder.subsampling_factor
        ds = self.output_subsampling_factor // native
        if ds > 1 and not return_native_resolution:
            chunk_hr = sm.downsample_preds(chunk_hr, ds)
        total_preds = torch.cat([total_preds, chunk_hr], dim=1)
        if not self.training and chunk_act is not None:
            self.sfx_last_activity = (
                chunk_act if self.sfx_last_activity is None else torch.cat([self.sfx_last_activity, chunk_act], 1)
            )
        if return_logits:
            total_logits = chunk_logits if total_logits is None else torch.cat([total_logits, chunk_logits], 1)
            if chunk_act is not None:
                total_activity_logits = (
                    chunk_act if total_activity_logits is None else torch.cat([total_activity_logits, chunk_act], 1)
                )
            return st, total_preds, total_logits, total_activity_logits
        return st, total_preds

    def forward_streaming(self, processed_signal, processed_signal_length, return_logits: bool = False):
        self.sfx_last_activity = None
        return super().forward_streaming(processed_signal, processed_signal_length, return_logits=return_logits)


def build_sfx_from_base(base_nemo: str, sfx: dict, cache_rescore: bool = False, map_location="cpu"):
    """Load the released .nemo, add SFX options, return SFXModel with base weights (new params zero-init)."""
    from omegaconf import OmegaConf, open_dict

    base = SortformerEncLabelModel.restore_from(base_nemo, map_location=map_location)
    cfg = base.cfg
    with open_dict(cfg):
        cfg.sfx = OmegaConf.create(sfx)
        cfg.target = "sfx.model.SFXModel"
        cfg.sortformer_modules._target_ = "sfx.modules.SFXModules"
        cfg.sortformer_modules.cache_rescore = cache_rescore
        cfg.train_ds.manifest_filepath = None
        cfg.validation_ds.manifest_filepath = None
        cfg.test_ds.manifest_filepath = None
    model = SFXModel(cfg=cfg)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("sfx_") for k in missing), missing
    del base
    return model
