"""Quick comparison of candidate systems with the reference systems (CPU only).

python compare.py sched_s1 c3_simconvmsd_s1 ...   (a '+' joins seeds of one system, e.g. ft3_s1+ft3_s2)
For each candidate: DER on MSDWild many/few, the meetings and all seven test sets pooled, AVA-AVD, and paired
bootstrap differences (grouped by recording) against FT-clips and CrowdSort-C (seed-averaged).
"""
import json
import os
import subprocess
import sys

ROOT = "/mnt/ssd2/paper/paper_1"
TEST = ["ami_test_mhm", "ami_test_sdm", "ali_test_near", "ali_test_far", "aishell4_test", "msdwild_few",
        "msdwild_many"]
GROUPS = {"many": ["msdwild_many"], "few": ["msdwild_few"], "meet": TEST[:5], "all": TEST, "ava": ["avaavd_test"],
          "vox": ["voxconverse_test"]}
REFS = {"N3D": "base", "FT-clips": "c3_noconcat+c3_noconcat_s2+c3_noconcat_s3", "CS-C": "ft3_s1+ft3_s2+ft3_s3"}


def files(sysname, sets, p="1.04", label="thr"):
    fs = ",".join("+".join(f"{ROOT}/exp/{m}/{s}/{p}/_perfile_{label}.json" for m in sysname.split("+")) for s in sets)
    return fs if all(os.path.exists(f) for f in fs.replace("+", ",").split(",")) else None


def boot(a, b, sets):
    A, B = files(a, sets), files(b, sets)
    if not A or not B:
        return None
    out = subprocess.run(["python3", f"{ROOT}/code/bootstrap.py", A, B, "--group"], capture_output=True, text=True)
    return json.loads(out.stdout)


def main():
    for cand in sys.argv[1:]:
        print(f"== {cand}")
        for g, sets in GROUPS.items():
            r = boot(REFS["N3D"], cand, sets)
            if r is None:
                continue
            cells = [f"{g:5s} DER {r['DER_B']:6.2f} (N3D {r['DER_A']:6.2f}, d {r['delta']:+.2f} p={r['p_B_not_better']:.3f})"]
            for ref in ("FT-clips", "CS-C"):
                rr = boot(REFS[ref], cand, sets)
                if rr:
                    cells.append(f"vs {ref} {rr['delta']:+.2f} [{rr['ci95'][0]:+.2f},{rr['ci95'][1]:+.2f}] p={rr['p_B_not_better']:.3f}")
            print("   " + " | ".join(cells))


if __name__ == "__main__":
    main()
