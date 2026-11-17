"""Additional statistics requested in review: grouped bootstrap, seed-wise effects, macro DER, influence.

- Grouped bootstrap: views of one meeting in several microphone conditions are resampled together
  (bootstrap.py --group).
- Seed-wise effects: the pooled DER difference of every seed of B against the seed-averaged A, and against
  the seed of A with the same index.
- Macro DER: mean of per-recording DER (each recording weighted equally) and the numbers of recordings that
  improve and degrade.
- Influence: the share of the total error reduction contributed by the single most influential recording,
  and the pooled difference after removing it.

python stats_extra.py  -> results/stats_extra.json and a printed summary
"""
import json
import os
import subprocess

import numpy as np

ROOT = "/mnt/ssd2/paper/paper_1"
TEST = ["ami_test_mhm", "ami_test_sdm", "ali_test_near", "ali_test_far", "aishell4_test", "msdwild_few",
        "msdwild_many"]
MEET = TEST[:5]
SYS = {"N3D": ["base"], "FT-meet": ["c3_nomsd"], "FT-clips": ["c3_noconcat", "c3_noconcat_s2", "c3_noconcat_s3"],
       "CrowdSort-C": ["ft3_s1", "ft3_s2", "ft3_s3"], "CrowdSort-I": ["c3_inter", "c3_inter_s2", "c3_inter_s3"],
       "CrowdSort": ["c3_ci_s1", "c3_ci_s2", "c3_ci_s3"], "CrowdSort-P": ["c3_cip_s1", "c3_cip_s2", "c3_cip_s3"]}


def have(ms, s, p="1.04"):
    return [m for m in ms if os.path.exists(f"{ROOT}/exp/{m}/{s}/{p}/_perfile_thr.json")]


def perfile(ms, s, p="1.04"):
    """per-recording errors averaged over seeds"""
    ds = [json.load(open(f"{ROOT}/exp/{m}/{s}/{p}/_perfile_thr.json")) for m in ms]
    return {k: {q: np.mean([d[k][q] for d in ds]) for q in ("total", "miss", "fa", "conf")} for k in ds[0]}


def boot(a, b, sets, group=True, p="1.04"):
    A = ",".join("+".join(f"{ROOT}/exp/{m}/{s}/{p}/_perfile_thr.json" for m in a) for s in sets)
    B = ",".join("+".join(f"{ROOT}/exp/{m}/{s}/{p}/_perfile_thr.json" for m in b) for s in sets)
    cmd = ["python3", f"{ROOT}/code/bootstrap.py", A, B, "--n", "10000"] + (["--group"] if group else [])
    return json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)


def per_rec(a, b, sets, p="1.04"):
    rows = []
    for s in sets:
        A, B = perfile(a, s, p), perfile(b, s, p)
        for k in A:
            ea = A[k]["miss"] + A[k]["fa"] + A[k]["conf"]
            eb = B[k]["miss"] + B[k]["fa"] + B[k]["conf"]
            rows.append((f"{s}:{k}", A[k]["total"], ea, eb))
    return rows


def macro_and_influence(rows):
    tot = sum(r[1] for r in rows)
    der_a = 100 * np.array([r[2] / r[1] for r in rows])
    der_b = 100 * np.array([r[3] / r[1] for r in rows])
    red = np.array([r[2] - r[3] for r in rows])
    i = int(np.argmax(red))
    pooled = 100 * (sum(r[3] for r in rows) - sum(r[2] for r in rows)) / tot
    without = 100 * ((sum(r[3] for r in rows) - rows[i][3]) - (sum(r[2] for r in rows) - rows[i][2])) / (tot - rows[i][1])
    return {"pooled_delta": round(pooled, 2), "macro_A": round(der_a.mean(), 2), "macro_B": round(der_b.mean(), 2),
            "macro_delta": round(der_b.mean() - der_a.mean(), 2), "n_improved": int((der_b < der_a - 1e-9).sum()),
            "n_degraded": int((der_b > der_a + 1e-9).sum()), "top_recording": rows[i][0],
            "top_share_of_reduction_pct": round(100 * red[i] / red.sum(), 1) if red.sum() > 0 else None,
            "pooled_delta_without_top": round(without, 2)}


def main():
    out = {}
    cmp = [("CrowdSort", "N3D"), ("CrowdSort-P", "N3D"), ("CrowdSort", "FT-clips"), ("CrowdSort-P", "FT-clips"),
           ("CrowdSort-C", "N3D"), ("CrowdSort-C", "FT-clips"), ("CrowdSort-I", "FT-clips"), ("FT-clips", "FT-meet")]
    groups = {"all": TEST, "many": ["msdwild_many"], "few": ["msdwild_few"], "meetings": MEET, "ava": ["avaavd_test"]}
    for b, a in cmp:
        A = have(SYS[a], "msdwild_many")
        B = have(SYS[b], "msdwild_many")
        key = f"{b} vs {a}"
        out[key] = {"seeds_A": A, "seeds_B": B}
        for g, sets in groups.items():
            out[key][f"grouped_{g}"] = boot(A, B, sets, group=True)
        out[key]["seedwise_many_vs_meanA"] = [boot(A, [m], ["msdwild_many"], group=False)["delta"] for m in B]
        if len(A) == len(B) and len(A) > 1:
            out[key]["seedwise_many_paired"] = [boot([x], [y], ["msdwild_many"], group=False)["delta"]
                                                for x, y in zip(A, B)]
        out[key]["many"] = macro_and_influence(per_rec(A, B, ["msdwild_many"]))
        out[key]["all"] = macro_and_influence(per_rec(A, B, TEST))
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results/stats_extra.json", "w"), indent=1)
    for k, v in out.items():
        print(f"== {k}  (A seeds {len(v['seeds_A'])}, B seeds {len(v['seeds_B'])})")
        for g in groups:
            r = v[f"grouped_{g}"]
            print(f"   grouped {g:9s} d={r['delta']:+.2f} CI [{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}] "
                  f"p={r['p_B_not_better']:.4f} groups={r['n_groups']}/{r['n_files']}")
        print("   seed-wise many vs mean A:", v["seedwise_many_vs_meanA"], " paired:", v.get("seedwise_many_paired"))
        print("   many:", v["many"])
        print("   all :", v["all"])


if __name__ == "__main__":
    main()
