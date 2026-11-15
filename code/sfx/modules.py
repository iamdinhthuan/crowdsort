"""Speaker-cache extensions for Streaming Sortformer (Nemotron 3 Diarization).

SFXModules keeps the stock AOSC policy but additionally tracks, for every cached frame,
  * its speaker-block (slot) index in the compressed cache,
  * its global frame index in the recording (for oracle experiments),
and supports
  * cache re-scoring: score compressed-cache frames with the current step's predictions
    instead of the predictions frozen at the time they were cached,
  * oracle cache scoring: score frames with reference labels (diagnostic upper bound),
  * speaker prototype accumulation: per-speaker running mean of pre-encode embeddings
    weighted by single-speaker confidence (input to the prototype memory).
"""
import math
import sfx  # noqa: F401  (SDPA attention patch)

import torch

from nemo.collections.asr.modules.sortformer_modules import SortformerModules


class SFXModules(SortformerModules):
    def __init__(self, *args, cache_rescore: bool = False, proto_conf_floor: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_rescore = cache_rescore
        self.proto_conf_floor = proto_conf_floor
        # Set externally for oracle experiments: (B, T80, n_spk) reference activity in arrival order.
        self.oracle_targets = None
        # Speaker-adaptive memory budget: dict(beta=frames per detected speaker, mmin=..., mmax=...) or None.
        self.adaptive_budget = None

    def _budget(self, st, batch_size, device):
        """Per-item cache budget M_b = clamp(beta * n_detected, mmin, mmax), rounded down to a multiple of n_spk."""
        ab = self.adaptive_budget
        n_det = (st.proto_w >= 2.0).sum(1).to(device)
        m = (ab["beta"] * n_det).clamp(min=ab["mmin"], max=ab["mmax"])
        return (m // self.n_spk * self.n_spk).long()

    # ------------------------------------------------------------------ state
    def init_streaming_state(self, batch_size: int = 1, async_streaming: bool = False, device=None):
        if async_streaming:
            raise NotImplementedError("SFXModules supports synchronous streaming only.")
        st = super().init_streaming_state(batch_size=batch_size, async_streaming=False, device=device)
        st.sfx_ptr = 0  # number of chunk frames pushed so far (global encoder-frame index)
        st.fifo_gidx = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        st.cache_gidx = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        st.sfx_slots = None  # (B, spkcache_len) block index of each compressed-cache frame; n_spk = disabled tail
        st.proto_sum = torch.zeros((batch_size, self.n_spk, self.fc_d_model), device=device, dtype=torch.float32)
        st.proto_w = torch.zeros((batch_size, self.n_spk), device=device)
        return st

    def _oracle(self, gidx):
        """Reference activity for global frame indices gidx (B, T); -1 -> zeros."""
        tgt = self.oracle_targets
        B, T = gidx.shape
        safe = gidx.clamp(min=0, max=tgt.shape[1] - 1)
        out = torch.gather(tgt.to(gidx.device), 1, safe.unsqueeze(-1).expand(-1, -1, tgt.shape[2]))
        valid = (gidx >= 0) & (gidx < tgt.shape[1])
        return out * valid.unsqueeze(-1)

    # ------------------------------------------------------------------ update
    def streaming_update(self, streaming_state, chunk, preds, lc: int = 0, rc: int = 0):
        st = streaming_state
        batch_size, _, emb_dim = chunk.shape
        spkcache_len, fifo_len = st.spkcache.shape[1], st.fifo.shape[1]
        chunk_len = chunk.shape[1] - lc - rc
        dev = chunk.device

        if st.spk_perm is not None:
            inv = torch.stack([torch.argsort(st.spk_perm[b]) for b in range(batch_size)])
            preds = torch.stack([preds[b, :, inv[b]] for b in range(batch_size)])

        fifo_preds = preds[:, spkcache_len : spkcache_len + fifo_len]
        chunk = chunk[:, lc : chunk_len + lc]
        chunk_preds = preds[:, spkcache_len + fifo_len + lc : spkcache_len + fifo_len + chunk_len + lc]
        chunk_gidx = (torch.arange(chunk_len, device=dev) + st.sfx_ptr).unsqueeze(0).expand(batch_size, -1)
        st.sfx_ptr += chunk_len

        oracle = self.oracle_targets is not None
        if oracle:
            fifo_preds = self._oracle(st.fifo_gidx)
            chunk_state_preds = self._oracle(chunk_gidx)
        else:
            chunk_state_preds = chunk_preds

        # ---- speaker prototype accumulation (canonical speaker order)
        with torch.no_grad():
            p = chunk_state_preds.float()
            others = torch.log1p(-p.clamp(max=1 - 1e-4)).sum(-1, keepdim=True) - torch.log1p(-p.clamp(max=1 - 1e-4))
            w = p * torch.exp(others)  # P(only speaker k active)
            w = torch.where(w > self.proto_conf_floor, w, torch.zeros_like(w))
        st.proto_sum = st.proto_sum + (w.unsqueeze(-1) * chunk.float().unsqueeze(2)).sum(1)
        st.proto_w = st.proto_w + w.sum(1)

        st.fifo = torch.cat([st.fifo, chunk], dim=1)
        st.fifo_preds = torch.cat([fifo_preds, chunk_state_preds], dim=1)
        st.fifo_gidx = torch.cat([st.fifo_gidx, chunk_gidx], dim=1)

        if fifo_len + chunk_len > self.fifo_len:
            pop = self.spkcache_update_period
            pop = max(pop, chunk_len - self.fifo_len + fifo_len)
            pop = min(pop, fifo_len + chunk_len)

            pop_embs = st.fifo[:, :pop]
            pop_preds = st.fifo_preds[:, :pop]
            pop_gidx = st.fifo_gidx[:, :pop]
            if not self.use_learnable_sil_emb:
                st.mean_sil_emb, st.n_sil_frames = self._get_silence_profile(
                    st.mean_sil_emb, st.n_sil_frames, pop_embs, pop_preds
                )
            st.fifo = st.fifo[:, pop:]
            st.fifo_preds = st.fifo_preds[:, pop:]
            st.fifo_gidx = st.fifo_gidx[:, pop:]

            st.spkcache = torch.cat([st.spkcache, pop_embs], dim=1)
            if oracle:
                cache_preds = self._oracle(st.cache_gidx)
            elif st.spkcache_compressed and not self.cache_rescore:
                cache_preds = st.spkcache_preds
            else:
                cache_preds = preds[:, :spkcache_len]
                if st.spkcache_compressed:
                    # disabled (silence / tail) slots carry no speaker evidence
                    cache_preds = cache_preds * (st.cache_gidx >= 0).unsqueeze(-1)
            st.spkcache_preds = torch.cat([cache_preds, pop_preds], dim=1)
            st.cache_gidx = torch.cat([st.cache_gidx, pop_gidx], dim=1)

            budget = self._budget(st, batch_size, dev) if self.adaptive_budget else None
            target = int(budget.max()) if budget is not None else self.spkcache_len
            if st.spkcache.shape[1] > target:
                st.spkcache, st.spkcache_preds, st.spk_perm, st.sfx_slots, sel, disabled = self.compress_with_slots(
                    st.spkcache, st.spkcache_preds, st.mean_sil_emb, permute_spk=self.training,
                    budget=budget, n_old=spkcache_len,
                )
                st.cache_gidx = torch.gather(st.cache_gidx, 1, sel).masked_fill(disabled, -1)
                st.spkcache_compressed = True
        return st, chunk_preds

    # ------------------------------------------------------------------ compression with slot ids
    def compress_with_slots(self, emb_seq, preds, mean_sil_emb, permute_spk: bool = False, budget=None, n_old=None):
        batch_size, n_frames, n_spk = preds.shape
        M = int(budget.max()) if budget is not None else self.spkcache_len
        base_len = self.adaptive_budget["mmin"] if budget is not None else self.spkcache_len
        if self.use_learnable_sil_emb:
            mean_sil_emb = self.learnable_sil_emb.to(dtype=emb_seq.dtype, device=emb_seq.device)
            mean_sil_emb = mean_sil_emb.unsqueeze(0).expand(batch_size, -1)
        per_spk = base_len // n_spk - self.spkcache_sil_frames_per_spk
        strong = math.floor(per_spk * self.strong_boost_rate)
        weak = math.floor(per_spk * self.weak_boost_rate)
        min_pos = math.floor(per_spk * self.min_pos_scores_rate)

        scores = self._get_log_pred_scores(preds)
        scores = self._disable_low_scores(preds, scores, min_pos)
        if permute_spk:
            max_perm_index = self._get_max_perm_index(scores)
            scores, spk_perm = self._permute_speakers(scores, max_perm_index)
        else:
            spk_perm = None
        if self.scores_boost_latest > 0:
            scores[:, (n_old if budget is not None else self.spkcache_len) :, :] += self.scores_boost_latest
        if self.training and self.scores_add_rnd > 0:
            scores += torch.rand(batch_size, n_frames, n_spk, device=scores.device) * self.scores_add_rnd
        scores = self._boost_topk_scores(scores, strong, scale_factor=2)
        scores = self._boost_topk_scores(scores, weak, scale_factor=1)
        if self.spkcache_sil_frames_per_spk > 0:
            pad = torch.full(
                (batch_size, self.spkcache_sil_frames_per_spk, n_spk), float("inf"), device=scores.device
            )
            scores = torch.cat([scores, pad], dim=1)

        n_tot = scores.shape[1]
        n_no_sil = n_tot - self.spkcache_sil_frames_per_spk
        flat = scores.permute(0, 2, 1).reshape(batch_size, -1)
        vals, idx = torch.topk(flat, M, dim=1, sorted=budget is not None)
        idx = torch.where(vals != float("-inf"), idx, self.max_index)
        if budget is not None:  # keep only the M_b best pairs of item b; the rest become tail slots
            rank = torch.arange(M, device=idx.device).unsqueeze(0)
            idx = torch.where(rank < budget.to(idx.device).unsqueeze(1), idx, self.max_index)
        idx, _ = torch.sort(idx, dim=1)
        tail = idx == self.max_index
        slots = torch.where(tail, torch.full_like(idx, n_spk), idx // n_tot)
        frame = torch.remainder(idx, n_tot)
        disabled = tail | (frame >= n_no_sil)
        frame = frame.masked_fill(disabled, 0)

        emb_dim = emb_seq.shape[2]
        cache = torch.gather(emb_seq, 1, frame.unsqueeze(-1).expand(-1, -1, emb_dim))
        cache = torch.where(disabled.unsqueeze(-1), mean_sil_emb.unsqueeze(1).expand(-1, M, -1), cache)
        cpreds = torch.gather(preds, 1, frame.unsqueeze(-1).expand(-1, -1, n_spk))
        cpreds = torch.where(disabled.unsqueeze(-1), 0.0, cpreds)
        return cache, cpreds, spk_perm, slots, frame, disabled
