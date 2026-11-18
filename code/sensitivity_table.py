"""Sensitivity of the MSDWild many results to shared speakers and to long recordings (Supplementary Table S1).

Subsets of MSDWild many at 1.04 s latency: all recordings; without the recordings flagged by speaker_overlap.py at
each of its three levels (fixed before any DER on the subsets was computed); without the recordings longer than
10 min. For each subset: number of recordings, scored speaker time (DER denominator), pooled DER of each system
(three seeds averaged per file) and the paired bootstrap of CrowdSort against each system (bootstrap.py --group,
post hoc, unadjusted).

python sensitivity_table.py  ->  results/sensitivity_many.json
"""
import json
import os
import subprocess
import tempfile

ROOT = "/mnt/ssd2/paper/paper_1"
SYSTEMS = {"N3D": ["base"], "FT-clips": ["c3_noconcat", "c3_noconcat_s2", "c3_noconcat_s3"],
           "SimConv-MSD": ["c3_simconvmsd_s1", "c3_simconvmsd_s2", "c3_simconvmsd_s3"],
           "SimConv-MSD-M": ["c3_simconvm_s1", "c3_simconvm_s2", "c3_simconvm_s3"],
           "FastMSS": ["c3_fastmss_s1", "c3_fastmss_s2", "c3_fastmss_s3"],
           "CrowdSort": ["c3_ci_s1", "c3_ci_s2", "c3_ci_s3"]}
LONG_S = 600.0


def perfile(m):
    return json.load(open(f"{ROOT}/exp/{m}/msdwild_many/1.04/_perfile_thr.json"))


def main():
    flagged = json.load(open(f"{ROOT}/results/speaker_overlap/summary.json"))["many_flagged"]
    dur = {os.path.basename(c["audio_filepath"])[:-4]: c["duration"]
           for c in map(json.loads, open(f"{ROOT}/data/manifests/msdwild_many.jsonl"))}
    data = {m: perfile(m) for ms in SYSTEMS.values() for m in ms}
    allk = set(data["base"])
    assert all(set(d) == allk for d in data.values()) and set(dur) >= allk
    long_ = {k for k in allk if dur[k] > LONG_S}
    subsets = {"all": allk, "strict": allk - set(flagged["strict"]), "moderate": allk - set(flagged["moderate"]),
               "lenient": allk - set(flagged["lenient"]), "le10min": allk - long_}
    out = {"long_recordings": sorted(long_)}
    with tempfile.TemporaryDirectory() as tmp:
        for name, keys in subsets.items():
            paths = {}
            for sysname, ms in SYSTEMS.items():
                ps = []
                for m in ms:
                    p = f"{tmp}/{name}_{m}.json"
                    json.dump({k: data[m][k] for k in keys}, open(p, "w"))
                    ps.append(p)
                paths[sysname] = "+".join(ps)
            res = {"n_recordings": len(keys),
                   "speaker_time_h": round(sum(data["base"][k]["total"] for k in keys) / 3600, 2), "der": {},
                   "crowdsort_vs": {}}
            for sysname in SYSTEMS:
                if sysname == "CrowdSort":
                    continue
                r = json.loads(subprocess.run(["python3", f"{ROOT}/code/bootstrap.py", paths[sysname],
                                               paths["CrowdSort"], "--group"],
                                              capture_output=True, text=True, check=True).stdout)
                res["der"][sysname] = r["DER_A"]
                res["der"]["CrowdSort"] = r["DER_B"]
                res["crowdsort_vs"][sysname] = {"delta": r["delta"], "ci95": r["ci95"], "p": r["p_B_not_better"]}
            out[name] = res
    json.dump(out, open(f"{ROOT}/results/sensitivity_many.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
