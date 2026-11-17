"""Fine-tune Nemotron 3 Diarization (optionally with SFX modules) with the official stage-2 streaming recipe.

python train_sfx.py --name ft_base --sfx '{}' --train data/manifests/train_mix.jsonl --steps 3000
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EXP = "/mnt/ssd2/paper/paper_1/exp"
BASE_NEMO = "/mnt/ssd2/paper/paper_1/models/base/Nemotron-3-Diarization.nemo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sfx", default="{}")
    ap.add_argument("--cache-rescore", action="store_true")
    ap.add_argument("--train", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--session-len", type=float, default=105.0)
    ap.add_argument("--min-chunk", type=float, default=15.0)
    ap.add_argument("--spkcache", type=int, default=0)
    args = ap.parse_args()

    import lightning.pytorch as pl
    from omegaconf import open_dict

    from sfx.model import build_sfx_from_base

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = os.path.join(EXP, args.name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=1)

    model = build_sfx_from_base(BASE_NEMO, json.loads(args.sfx), cache_rescore=args.cache_rescore)
    cfg = model.cfg
    with open_dict(cfg):
        cfg.train_ds.manifest_filepath = os.path.abspath(args.train)
        cfg.train_ds.batch_size = args.bs
        cfg.train_ds.num_workers = args.workers
        cfg.train_ds.session_len_sec = args.session_len
        cfg.train_ds.subsegment_single_chunk_min_len_sec = args.min_chunk
        if args.spkcache:
            cfg.sortformer_modules.spkcache_len = args.spkcache
        cfg.lr = args.lr
        cfg.optim.lr = args.lr
        cfg.optim.sched = {"name": "CosineAnnealing", "warmup_steps": args.warmup, "min_lr": args.lr * 0.05,
                           "max_steps": args.steps}
    model._cfg = cfg
    if args.spkcache:
        model.sortformer_modules.spkcache_len = args.spkcache
    model.setup_training_data(cfg.train_ds)

    class Saver(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            s = trainer.global_step
            if s > 0 and s % args.save_every == 0 and trainer.is_global_zero:
                pl_module.save_to(os.path.join(out, f"step{s}.nemo"))

    class LossLog(pl.Callback):
        def __init__(self):
            self.f = open(os.path.join(out, "train_log.jsonl"), "a")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            s = trainer.global_step
            if s % 25 == 0:
                m = {k: float(v) for k, v in trainer.callback_metrics.items() if k in (
                    "loss", "ats_loss", "pil_loss", "activity_loss", "phantom_loss", "train_f1_acc", "learning_rate")}
                m["step"] = s
                m["time"] = __import__("time").time()
                m["max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                self.f.write(json.dumps(m) + "\n")
                self.f.flush()

    trainer = pl.Trainer(
        devices=1, accelerator="gpu", precision="bf16-mixed", max_steps=args.steps,
        accumulate_grad_batches=args.accum, gradient_clip_val=1.0, limit_val_batches=0, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False, log_every_n_steps=25,
        callbacks=[Saver(), LossLog()],
    )
    model.set_trainer(trainer)
    trainer.fit(model)
    model.save_to(os.path.join(out, "final.nemo"))
    print("TRAIN_DONE", out, flush=True)


if __name__ == "__main__":
    main()
