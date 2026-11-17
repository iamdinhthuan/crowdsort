"""Add a real source to an existing training mix, replicating its lines to a target number of samples.

python make_mix_add.py BASE.jsonl SRC.jsonl OUT.jsonl N   (in data/manifests; random top-up, seed 0)
"""
import random
import sys

base, src, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
random.seed(0)
lines = open(src).readlines()
add = lines * (n // len(lines)) + random.sample(lines, n % len(lines))
mix = open(base).readlines() + add
open(out, "w").writelines(mix)
print(out, len(mix), "lines,", len(add), "from", src, f"({100 * len(add) / len(mix):.1f}%)")
