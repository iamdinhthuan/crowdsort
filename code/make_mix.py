"""Build a training manifest with per-source sampling shares.

The NeMo dataset draws one random 105 s subsegment per manifest line per epoch, so a source's share of
training samples is its share of lines. Long-recording sources are replicated so that every source gets
roughly the requested share.

python make_mix.py out.jsonl src1.jsonl:share src2.jsonl:share ...
"""
import json
import math
import sys


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    srcs = []
    for s in specs:
        path, share = s.rsplit(":", 1)
        lines = [json.loads(l) for l in open(path)]
        srcs.append((path, float(share), lines))
    # number of lines in the largest-by-lines source sets the scale
    unit = max(len(l) / sh for _, sh, l in srcs)
    rows = []
    for path, share, lines in srcs:
        rep = max(1, round(unit * share / len(lines)))
        rows += lines * rep
        print(f"{path}: {len(lines)} lines x{rep} -> {len(lines) * rep}  "
              f"({sum(x['duration'] for x in lines) / 3600:.1f} h unique)")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(out, len(rows), "lines")


if __name__ == "__main__":
    main()
