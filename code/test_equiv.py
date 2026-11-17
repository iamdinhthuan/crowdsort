import sys, json, numpy as np, soundfile as sf, torch
sys.path.insert(0, 'code')
from infer import load_model, set_profile
x, sr = sf.read(sys.argv[1], dtype='float32'); x = x[: 16000 * int(sys.argv[2])]
xa = torch.from_numpy(x)[None].cuda(); la = torch.tensor([len(x)]).cuda()
outs = {}
for name, spec, sfx, cr in [('stock','stock','{}',False), ('sfx0','base','{}',False),
                            ('sfx_slot_hr','base','{"slot_emb":true,"hr_skip":true}',False),
                            ('sfx_cr','base','{}',True),
                            ('sfx_full','base','{"slot_emb":true,"proto":true,"hr_skip":true}',True)]:
    m = load_model(spec, sfx, cr, torch.device('cuda'))
    set_profile(m, sys.argv[3])
    with torch.inference_mode():
        p, lg, act = m(audio_signal=xa, audio_signal_length=la, return_logits=True)
    outs[name] = p[0].float().cpu().numpy()
    print(name, p.shape, None if act is None else act.shape, flush=True)
    del m; torch.cuda.empty_cache()
ref = outs['stock']
for k, v in outs.items():
    print(k, 'max|diff| vs stock', float(np.abs(v - ref).max()), 'mean', float(np.abs(v - ref).mean()), 'bin-agree', float(((v>0.5)==(ref>0.5)).mean()))
