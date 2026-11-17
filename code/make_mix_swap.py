"""Training mix with the concatenation sessions of train_mix_v3 replaced by other synthetic sources.

The real-data lines of train_mix_v3 are kept unchanged and the 2388 synthetic lines (13.2% of the samples)
are filled with the given sources in equal parts, replicating sessions and topping up with a random
subset, so that every compared mix has the same real data and the same synthetic share.

python make_mix_swap.py OUT.jsonl SRC1.jsonl [SRC2.jsonl ...] [--hours H]
  --hours H : first truncate each source to its first H hours (to match the 26.8 h of msdwild_concat_v2)
  --share F : synthetic share of the samples instead of the 13.2% of train_mix_v3 (the real lines stay unchanged)
"""
import json
import random
import sys

args = sys.argv[1:]
hours = None
if "--hours" in args:
    i = args.index("--hours")
    hours = float(args[i + 1])
    args = args[:i] + args[i + 2:]
share = None
if "--share" in args:
    i = args.index("--share")
    share = float(args[i + 1])
    args = args[:i] + args[i + 2:]
out, srcs = args[0], args[1:]
random.seed(0)
v3 = open("train_mix_v3.jsonl").readlines()
base = [l for l in v3 if "/msdwild_concat_v2/" not in l]
n = len(v3) - len(base)
if share is not None:
    n = round(share * len(base) / (1 - share))
mix = list(base)
for j, s in enumerate(srcs):
    lines = open(s).readlines()
    if hours is not None:
        kept, tot = [], 0.0
        for l in lines:
            if tot >= hours * 3600:
                break
            kept.append(l)
            tot += json.loads(l)["duration"]
        lines = kept
    k = n // len(srcs) + (1 if j < n % len(srcs) else 0)
    take = lines * (k // len(lines))
    take += random.sample(lines, k - len(take))
    mix += take
    print(s, len(lines), "sessions", round(sum(json.loads(l)["duration"] for l in lines) / 3600, 1), "h ->", k, "lines")
random.shuffle(mix)
open(out, "w").writelines(mix)
print(out, len(mix), "lines, synthetic share", round(100 * n / len(mix), 1), "%")
