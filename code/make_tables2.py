"""Tables of the revision: synthetic-source/operator comparison and evaluation on other corpora.

python make_tables2.py  -> paper/tex/tables/{sources,external}.tex ; prints the numbers it writes.
Pooled DER from per-recording errors (seed-averaged for multi-seed systems); paired bootstrap grouped by
recording (bootstrap.py --group), 10^4 resamples, seed 0.
"""
import json
import os
import subprocess

import numpy as np

ROOT = "/mnt/ssd2/paper/paper_1"
OUT = f"{ROOT}/paper/tex/tables"
TEST = ["ami_test_mhm", "ami_test_sdm", "ali_test_near", "ali_test_far", "aishell4_test", "msdwild_few",
        "msdwild_many"]
MEET = TEST[:5]
FTC = ["c3_noconcat", "c3_noconcat_s2", "c3_noconcat_s3"]
SCM = ["c3_simconvmsd_s1", "c3_simconvmsd_s2", "c3_simconvmsd_s3"]
FMS = ["c3_fastmss_s1", "c3_fastmss_s2", "c3_fastmss_s3"]
FTM = ["c3_nomsd", "c3_nomsd_s2", "c3_nomsd_s3"]
CSC = ["ft3_s1", "ft3_s2", "ft3_s3"]


def have(models, sets, p="1.04"):
    return [m for m in models if all(os.path.exists(f"{ROOT}/exp/{m}/{s}/{p}/_perfile_thr.json") for s in sets)]


def pooled(models, sets, p="1.04"):
    tot = err = 0.0
    for s in sets:
        ds = [json.load(open(f"{ROOT}/exp/{m}/{s}/{p}/_perfile_thr.json")) for m in models]
        for k in ds[0]:
            tot += np.mean([d[k]["total"] for d in ds])
            err += np.mean([d[k]["miss"] + d[k]["fa"] + d[k]["conf"] for d in ds])
    return 100 * err / tot


def boot(a, b, sets, p="1.04"):
    A = ",".join("+".join(f"{ROOT}/exp/{m}/{s}/{p}/_perfile_thr.json" for m in a) for s in sets)
    B = ",".join("+".join(f"{ROOT}/exp/{m}/{s}/{p}/_perfile_thr.json" for m in b) for s in sets)
    r = subprocess.run(["python3", f"{ROOT}/code/bootstrap.py", A, B, "--group"], capture_output=True, text=True)
    return json.loads(r.stdout)


def cell(models, sets, p="1.04"):
    ms = have(models, sets, p)
    return f"{pooled(ms, sets, p):.2f}" if ms else "--"


def dcell(ref, models, sets, p="1.04"):
    ms, rs = have(models, sets, p), have(ref, sets, p)
    if not ms or not rs or sorted(ms) == sorted(rs):
        return "--"
    r = boot(rs, ms, sets, p)
    star = "$^{*}$" if r["p_B_not_better"] < 0.05 else "$^{\\#}$" if r["p_B_not_better"] > 0.95 else ""
    return f"${r['delta']:+.2f}${star}"


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    if name == "sources":
        text = text.replace("\\begin{table}", "\\begin{table*}").replace("\\end{table}", "\\end{table*}")
        text = text.replace("\\begin{tabular}{lclccccccc}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lclccccccc}")
        text = text.replace("\\end{tabular}", "\\end{tabular*}")
    else:
        text = text.replace("max width=\\textwidth", "max width=\\columnwidth")
    open(f"{OUT}/{name}.tex", "w").write(text + "\n")
    print(text)


def sources_table(rows):
    lines = ["\\begin{table}[!tbph]\n\\centering",
             "\\caption{Comparison of synthetic training data. DER (\\%) of systems that differ only in how the synthetic sessions are built. Meetings, Pooled: five meeting sets and all seven test sets; $\\Delta$: difference from FT-clips; $^{*}$: significant decrease ($p<0.05$).}\\label{tab:sources}",
             "\\small\\setlength{\\tabcolsep}{2pt}",
             "\\adjustbox{max width=\\textwidth}{\\begin{tabular}{lclccccccc}\n\\toprule",
             "Synthetic data & Seeds & Source & MSDWild many & $\\Delta$ & MSDWild few & Meetings & Pooled & AVA-AVD & $\\Delta$\\\\\n\\midrule"]
    for lab, src, models in rows:
        ms = have(models, TEST)
        if not ms:
            continue
        lines.append(f"{lab} & {len(ms)} & {src} & {cell(ms, ['msdwild_many'])} & {dcell(FTC, ms, ['msdwild_many'])} & "
                     f"{cell(ms, ['msdwild_few'])} & {cell(ms, MEET)} & {cell(ms, TEST)} & {cell(models, ['avaavd_test'])} & "
                     f"{dcell(FTC, models, ['avaavd_test'])}\\\\")
    lines.append("\\bottomrule\n\\end{tabular}}\n\\end{table}")
    write("sources", "\n".join(lines))


def external_table(rows):
    """AVA-AVD (unseen corpus, exploratory) and NOTSOFAR-1 (scored once; forced-alignment and word-timing references)."""
    corp = [("avaavd_test", "AVA-AVD"), ("notsofar_eval_sc_fa", "NOTSOFAR-1, FA"), ("notsofar_eval_sc", "NOTSOFAR-1, Words")]
    corp = [c for c in corp if have(["base"], [c[0]])]
    ncol = 4 * len(corp)
    lines = ["\\begin{table*}[!tbph]\n\\centering",
             "\\caption{Results on held-out corpora. DER (\\%) on AVA-AVD and on NOTSOFAR-1 with forced-alignment (FA) and word-timing (Words) references. $\\Delta$: difference from N3D; $^{*}$/$^{\\#}$: significant decrease/increase ($p<0.05$); +N: NOTSOFAR-1 training data added to fine-tuning.}\\label{tab:external}",
             "\\small\\setlength{\\tabcolsep}{2pt}",
             "\\adjustbox{max width=\\textwidth}{\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * ncol + "}\n\\toprule",
             " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{n}}}" for _, n in corp) + "\\\\",
             "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(corp))),
             " & " + " & ".join("\\multicolumn{2}{c}{1.04~s} & \\multicolumn{2}{c}{30.4~s}" for _ in corp) + "\\\\",
             "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(2 * len(corp))),
             "System" + " & DER & $\\Delta$" * (2 * len(corp)) + "\\\\\n\\midrule"]
    for lab, models in rows:
        if lab == "MIDRULE":
            lines.append("\\midrule")
            continue
        cells = []
        for s, _ in corp:
            for p in ("1.04", "30.4"):
                cells += [cell(models, [s], p), "" if lab == "N3D" else dcell(["base"], models, [s], p)]
        if cells[0] == "--":
            continue
        lines.append(lab + " & " + " & ".join(cells) + "\\\\")
    lines.append("\\bottomrule\n\\end{tabular*}}\n\\end{table*}")
    text = "\n".join(lines)
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/external.tex", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    sources_table([
        ("None (FT-clips)", "--", FTC),
        ("SimConv-MSD", "MSDWild", SCM),
        ("SimConv-MSD-M", "MSDWild", ["c3_simconvm_s1", "c3_simconvm_s2", "c3_simconvm_s3"]),
        ("FastMSS", "LibriSpeech", FMS),
        ("Arrival operator (\\sysname{}-C)", "MSDWild", CSC),
        ("Return operator (\\sysname{}-I)", "MSDWild", ["c3_inter", "c3_inter_s2", "c3_inter_s3"]),
        ("Both operators (\\sysname{})", "MSDWild", ["c3_ci_s1", "c3_ci_s2", "c3_ci_s3"]),
        ("\\quad + boundary masking (\\sysname{}-P)", "MSDWild", ["c3_cip_s1", "c3_cip_s2", "c3_cip_s3"]),
        ("Arrival operator, 2386 distinct sessions", "MSDWild", ["c3_concatu_s1"]),
        ("Arrival operator, 2386 sessions, 25\\% share", "MSDWild", ["c3_concat25_s1"]),
        ("Both operators, 4--8 speakers per session", "MSDWild", ["c3_ck3_s1", "c3_ck3_s2", "c3_ck3_s3"]),
    ])
    external_table([
        ("N3D", ["base"]), ("FT-meet", FTM), ("FT-clips", FTC), ("SimConv-MSD", SCM),
        ("FastMSS", FMS), ("\\sysname{}-C", CSC), ("\\sysname{}-I", ["c3_inter", "c3_inter_s2", "c3_inter_s3"]),
        ("\\sysname{}", ["c3_ci_s1", "c3_ci_s2", "c3_ci_s3"]), ("\\sysname{}-P", ["c3_cip_s1", "c3_cip_s2", "c3_cip_s3"]),
        ("MIDRULE", []),
        ("FT-clips+N", ["c3_noconcat_nsf_s1", "c3_noconcat_nsf_s2", "c3_noconcat_nsf_s3"]),
        ("\\sysname{}+N", ["c3_ci_nsf_s1", "c3_ci_nsf_s2", "c3_ci_nsf_s3"]),
    ])
