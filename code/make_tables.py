"""Generate the LaTeX result tables of the paper from results/*.jsonl and per-file outputs (no hand copying).

python make_tables.py   (writes paper/tex/tables/*.tex and prints them)
"""
import json
import re
import os
import subprocess

import numpy as np

ROOT = "/mnt/ssd2/paper/paper_1"
OUT = f"{ROOT}/paper/tex/tables"
TEST = ["ami_test_mhm", "ami_test_sdm", "ali_test_near", "ali_test_far", "aishell4_test", "msdwild_few", "msdwild_many"]
DEV = ["ami_dev_mhm", "ami_dev_sdm", "ali_eval_near", "ali_eval_far", "aishell4_dev", "msdwild_fewdev",
       "msdwild_concatdev"]
FT = ["ft3_s1", "ft3_s2", "ft3_s3"]
AD = "ad50_396"


def load():
    R = {}
    for l in open(f"{ROOT}/results/results.jsonl"):
        r = json.loads(l)
        R[(r["model"], r["set"], r["profile"], r["decode"])] = r
    return R


R = load()


def perfile(model, s, p="1.04"):
    f = f"{ROOT}/exp/{model}/{s}/{p}/_perfile_thr.json"
    return json.load(open(f)) if os.path.exists(f) else None


def pooled(model, sets, p="1.04"):
    tot = err = 0.0
    for s in sets:
        pf = perfile(model, s, p)
        if pf is None:
            return None
        for r in pf.values():
            tot += r["total"]
            err += r["miss"] + r["fa"] + r["conf"]
    return 100 * err / tot


def val(model, s, p="1.04", metric="DER"):
    if s == "pooled":
        return pooled(model, TEST, p)
    if s == "pooled_dev":
        return pooled(model, DEV, p)
    r = R.get((model, s, p, "thr"))
    return None if r is None else r[metric]


def seedavg(models, s, p="1.04"):
    """pooled DER from per-recording errors averaged over seeds (same estimator as the bootstrap and Table 4)"""
    sets = TEST if s == "pooled" else [s]
    tot = err = 0.0
    for st in sets:
        pfs = [perfile(m, st, p) for m in models]
        pfs = [d for d in pfs if d is not None]
        if not pfs:
            return None
        for k in pfs[0]:
            tot += np.mean([d[k]["total"] for d in pfs])
            err += np.mean([d[k]["miss"] + d[k]["fa"] + d[k]["conf"] for d in pfs])
    return 100 * err / tot


def fmt(vals, bold=False, std=True):
    vals = [v for v in vals if v is not None] if isinstance(vals, list) else vals
    if vals is None or (isinstance(vals, list) and not vals):
        return "--"
    if not isinstance(vals, list):
        vals = [vals]
    m = float(np.mean(vals))
    t = f"{m:.2f}"
    if std and len(vals) > 1:
        t += f"\\,$\\pm$\\,{np.std(vals, ddof=1):.2f}"
    return f"\\textbf{{{t}}}" if bold else t


def boot(a_models, b_models, sets, p="1.04"):
    """paired bootstrap pooled over sets; seeds averaged per file ('+')."""
    A = ",".join("+".join(f"{ROOT}/exp/{m}/{s}/{p}/_perfile_thr.json" for m in a_models) for s in sets)
    B = ",".join("+".join(f"{ROOT}/exp/{m}/{s}/{p}/_perfile_thr.json" for m in b_models) for s in sets)
    for f in (A + "," + B).replace("+", ",").split(","):
        if not os.path.exists(f):
            return None
    out = subprocess.run(["python3", f"{ROOT}/code/bootstrap.py", A, B, "--n", "10000", "--group"], capture_output=True,
                         text=True).stdout
    return json.loads(out)


HEADER = ("\\begin{tabular}{l" + "c" * 8 + "}\n\\toprule\n"
          " & \\multicolumn{2}{c}{AMI} & \\multicolumn{2}{c}{AliMeeting} & AISHELL-4 & \\multicolumn{2}{c}{MSDWild} & \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){7-8}\n"
          "System & MHM & SDM & Near & Far & & few & many & Pooled\\\\\n\\midrule")
COLS = TEST + ["pooled"]


def table_rows(rows, p="1.04", cols=None):
    """rows: (label, [models]) or (label, [models], profile) ; bold = best mean per column."""
    cols = cols or COLS
    prof = [r[2] if len(r) > 2 else p for r in rows]
    rows = [r[:2] for r in rows]
    data = [[([seedavg(ms, c, q)] if len(ms) > 1 else [val(m, c, q) for m in ms]) for c in cols]
            for (_, ms), q in zip(rows, prof)]
    lines = []
    best = []
    for j in range(len(cols)):
        ms = [np.mean([v for v in d[j] if v is not None]) for d in data if all(v is not None for v in d[j])]
        best.append(min(ms) if ms else None)
    for (lab, ms), d in zip(rows, data):
        cells = []
        for j, vs in enumerate(d):
            ok = all(v is not None for v in vs)
            b = ok and best[j] is not None and abs(np.mean(vs) - best[j]) < 1e-9
            cells.append(fmt(vs if ok else None, bold=b, std=False))
        lines.append(lab + " & " + " & ".join(cells) + "\\\\")
    return lines


WIDE = {"main", "latency", "capacity", "ablation"}  # two-column layout: these span both columns


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    width = "\\textwidth" if name in WIDE else "\\columnwidth"
    if name in WIDE:
        # span the full text width: columns spread evenly, shrink only if the content is wider
        text = text.replace("\\begin{table}", "\\begin{table*}").replace("\\end{table}", "\\end{table*}")
        text = re.sub(r"\\begin\{tabular\}\{([^}]*)\}", lambda m: "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}" + m.group(1) + "}", text)
        text = text.replace("\\end{tabular}", "\\end{tabular*}")
        text = text.replace("\\begin{tabular*}", "\\adjustbox{max width=\\textwidth}{\\begin{tabular*}").replace("\\end{tabular*}", "\\end{tabular*}}")
        open(f"{OUT}/{name}.tex", "w").write(text)
        print(f"% ---- {name}\n{text}\n")
        return
    text = text.replace("\\begin{tabular}", "\\adjustbox{max width=" + width + "}{\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}")
    open(f"{OUT}/{name}.tex", "w").write(text)
    print(f"% ---- {name}\n{text}\n")


FTC = ["c3_noconcat", "c3_noconcat_s2", "c3_noconcat_s3"]
SCM = ["c3_simconvmsd_s1", "c3_simconvmsd_s2", "c3_simconvmsd_s3"]
SCMM = ["c3_simconvm_s1", "c3_simconvm_s2", "c3_simconvm_s3"]
FMS = ["c3_fastmss_s1", "c3_fastmss_s2", "c3_fastmss_s3"]
FTM = ["c3_nomsd", "c3_nomsd_s2", "c3_nomsd_s3"]
INTER = ["c3_inter", "c3_inter_s2", "c3_inter_s3"]
CI = ["c3_ci_s1", "c3_ci_s2", "c3_ci_s3"]
CIP = ["c3_cip_s1", "c3_cip_s2", "c3_cip_s3"]


def have(models, sets=TEST, p="1.04"):
    return [m for m in models if all(perfile(m, s, p) is not None for s in sets)]


def seeds(lab, ms):
    return f"{lab} ({len(ms)} seed{'s' if len(ms) > 1 else ''})" if len(ms) > 1 else lab


def main_table():
    ftc, cs, ci, cip = have(FTC), have(FT), have(CI), have(CIP)
    ftm, scm, fms = have(FTM), have(SCM), have(FMS)
    rows = [("N3D (released)", ["base"]), (seeds("FT-meet", ftm), ftm), (seeds("FT-clips", ftc), ftc),
            (seeds("SimConv-MSD", scm), scm), (seeds("FastMSS", fms), fms),
            (seeds("\\sysname{}-C", cs), cs), (seeds("\\sysname{}", ci), ci), (seeds("\\sysname{}-P", cip), cip)]
    rows = [r for r in rows if r[1] and have(r[1])]
    if have(["ls_eend_dih3"], TEST, "1.0"):  # external streaming baseline, released DIHARD III checkpoint, 1.0 s latency
        rows = [("LS-EEND$^{\\dagger}$", ["ls_eend_dih3"], "1.0")] + rows
    cols = COLS + ["avaavd_test"]
    body = table_rows(rows, cols=cols)
    if rows[0][0].startswith("LS-EEND"):
        body = [body[0].replace("\\textbf{", "{"), "\\midrule"] + body[1:]
    head = HEADER.replace("{l" + "c" * 8 + "}", "{l" + "c" * 9 + "}").replace(" & \\\\\n\\cmidrule", " & & \\\\\n\\cmidrule").replace("Pooled\\\\", "Pooled & AVA-AVD\\\\")
    t = ["\\begin{table}[!tbph]\n\\centering",
         "\\caption{Main results at 1.04~s latency. DER (\\%) on the seven test sets and on the unseen AVA-AVD test set. FT-meet: fine-tuned on real meetings only; "
         "FT-clips: real meetings and MSDWild clips without synthesis; SimConv-MSD and FastMSS: FT-clips plus simulated conversations; "
         "\\sysname{}-C: arrival operator only; \\sysname{}: arrival and return operators; \\sysname{}-P: with boundary masking. "
         "Multi-seed rows pool per-recording errors averaged over seeds. Pooled: DER over all 759 recordings of the seven test sets. Best value per column in bold."
         + (" $^{\\dagger}$External streaming baseline: released DIHARD~III checkpoint of LS-EEND at 8~kHz and about 1~s latency, not fine-tuned." if rows[0][0].startswith("LS-EEND") else "")
         + "}\\label{tab:main}",
         "\\small\\setlength{\\tabcolsep}{4pt}", head] + body + ["\\bottomrule\n\\end{tabular}\n\\end{table}"]
    write("main", "\n".join(t))
    for lab, ms, *_ in rows:
        if len(ms) > 1:
            print(lab, {c: (round(float(np.mean([val(m, c) for m in ms])), 2), round(float(np.std([val(m, c) for m in ms], ddof=1)), 2)) for c in cols})


def latency_table():
    lines = ["\\begin{table}[!tbph]\n\\centering",
             "\\caption{Results at three latency settings. DER (\\%) at 30.4, 1.04 and 0.32~s latency; seed 1 for the fine-tuned models.}\\label{tab:latency}",
             "\\small\\setlength{\\tabcolsep}{4pt}", HEADER]
    for p in ["30.4", "1.04", "0.32"]:
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{Input-buffer latency {p}~s}}}}\\\\")
        rows = [("N3D", ["base"]), ("FT-clips", ["c3_noconcat"]), ("\\sysname{}-C", ["ft3_s1"]),
                ("\\sysname{}", ["c3_ci_s1"]), ("\\sysname{}-P", ["c3_cip_s1"])]
        rows = [r for r in rows if have(r[1], TEST, p)]
        lines += table_rows(rows, p)
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    write("latency", "\n".join(lines))


def ablation_table():
    rows = [("\\sysname{}-C (reference)", "ft3_s1"), ("+ SAC", "c3_sac"), ("+ SAC + SPM", "c3_sacspm"), ("+ PCR", "c3_pcr"),
            ("+ HRS", "c3_hrs"), ("+ SAC + SPM + PCR + HRS", "sam3A_s1"), ("+ SAC + SPM-deep + PCR + HRS", "sam3B_s1"),
            ("trained with $M=396$, inference $M=396$", "c3_cap396"), ("capacity-elastic training, $M=264$", "c3_elastic_c264"),
            ("capacity-elastic training, adaptive", "c3_elastic_ad66_462"),
            ("\\sysname{}-C, inference $M=396$", "ft3_s1_inf396"), ("\\sysname{}-C, adaptive budget", f"ft3_s1_{AD}")]
    lines = ["\\begin{table}[!tbph]\n\\centering",
             "\\caption{Memory variants. Variants added to \\sysname{}-C (arrival operator only, seed 1), all fine-tuned with the same data, schedule and seed: "
             "pooled DER (\\%) on the development and test sets, DER on the two test sets with many speakers, and the paired bootstrap "
             "difference of the pooled test DER with respect to \\sysname{}-C (95\\% confidence interval).}\\label{tab:ablation}",
             "\\small\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lccccc}\n\\toprule",
             "System & Dev & Test & AISHELL-4 & MSDWild many & $\\Delta$ test [95\\% CI]\\\\\n\\midrule"]
    for lab, m in rows:
        d, t = pooled(m, DEV), pooled(m, TEST)
        a, w = val(m, "aishell4_test"), val(m, "msdwild_many")
        if t is None and d is None:
            continue
        b = boot(["ft3_s1"], [m], TEST) if m != "ft3_s1" and t is not None else None
        bs = f"${b['delta']:+.2f}$ $[{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}]$" if b else "--"
        lines.append(f"{lab} & {fmt(d)} & {fmt(t)} & {fmt(a)} & {fmt(w)} & {bs}\\\\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    write("ablation", "\n".join(lines))


def stats_table():
    ftc, cs, ci, cip = have(FTC), have(FT), have(CI), have(CIP)
    pairs = [("\\sysname{} vs.\\ N3D", ["base"], ci), ("\\sysname{}-P vs.\\ N3D", ["base"], cip),
             ("\\sysname{} vs.\\ FT-clips", ftc, ci), ("\\sysname{}-P vs.\\ FT-clips", ftc, cip),
             ("\\sysname{} vs.\\ SimConv-MSD", have(SCM), ci),
             ("\\sysname{} vs.\\ SimConv-MSD-M", have(SCMM), ci),
             ("\\sysname{} vs.\\ FastMSS", have(FMS), ci),
             ("\\sysname{} vs.\\ \\sysname{}-C", cs, ci),
             ("\\sysname{}-P vs.\\ \\sysname{}", ci, cip),
             ("FT-clips vs.\\ FT-meet", have(FTM), ftc)]
    groups = [("All", TEST), ("many", ["msdwild_many"]), ("few", ["msdwild_few"]),
              ("Meet.", ["ami_test_mhm", "ami_test_sdm", "ali_test_near", "ali_test_far", "aishell4_test"]),
              ("AVA", ["avaavd_test"])]
    res = {}
    for lab, a, b in pairs:
        for gname, sets in groups:
            if a and b:
                res[(lab, gname)] = boot(a, b, sets)
    ps = sorted((r["p_B_not_better"], k) for k, r in res.items() if r)
    m, holm = len(ps), set()
    for i, (pv, k) in enumerate(ps):
        if pv > 0.05 / (m - i):
            break
        holm.add(k)
    lines = ["\\begin{table}[!tbph]\n\\centering",
             "\\caption{Paired bootstrap tests. Difference of the pooled DER (B $-$ A, points) at 1.04~s latency; views of one meeting and clips of one film are resampled together, "
             "and per-recording errors are averaged over seeds. $^{*}$: one-sided $p<0.05$; $^{**}$: $p<0.001$; bold: significant after "
             f"Holm--Bonferroni correction over the {m} tests. Many/few: MSDWild; Meet.: five meeting sets; AVA: AVA-AVD.}}\\label{{tab:stats}}",
             "\\small\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lccccc}\n\\toprule",
             "B vs.\\ A & All & many & few & Meet. & AVA\\\\\n\\midrule"]
    for lab, a, b in pairs:
        cells = []
        for gname, _ in groups:
            r = res.get((lab, gname))
            if not r:
                cells.append("--")
                continue
            pv = r["p_B_not_better"]
            star = "$^{**}$" if pv < 0.001 else ("$^{*}$" if pv < 0.05 else "")
            v = f"${r['delta']:+.2f}$".replace("-0.00", "0.00").replace("+0.00", "0.00")
            cells.append((f"$\\mathbf{{{v[1:-1]}}}$" if (lab, gname) in holm else v) + star)
        lines.append(lab + " & " + " & ".join(cells) + "\\\\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    write("stats", "\n".join(lines))
    for k, r in res.items():
        if r:
            print("STAT", k, r["delta"], r["ci95"], r["p_B_not_better"], "holm" if k in holm else "")


def capacity_table():
    sizes = [132, 264, 330, 396, 462, 528]
    speed = {}
    for l in open(f"{ROOT}/results/speed.jsonl"):
        r = json.loads(l)
        if r["name"] == "N3D" and r["profile"] == "1.04" and r["batch"] == 16:
            speed[r.get("spkcache", 264)] = r["rtfx"]
    lines = ["\\begin{table}[!tbph]\n\\centering",
             "\\caption{Size of the speaker cache at inference. DER (\\%) and throughput (RTFx, batch 16, RTX 5090) of the released model at 1.04~s latency, "
             "with the cache size changed at inference without retraining. The adaptive budget uses $(\\beta,M_{\\min},M_{\\max})=(50,264,396)$.}"
             "\\label{tab:capacity}",
             "\\small\\setlength{\\tabcolsep}{4pt}", HEADER.replace("Pooled\\\\", "Pooled & RTFx\\\\").replace("{l" + "c" * 8 + "}", "{l" + "c" * 9 + "}").replace(" & \\\\\n\\cmidrule", " & & \\\\\n\\cmidrule")]
    for c in sizes:
        m = "base" if c == 264 else f"base_sc{c}"
        cells = [fmt(val(m, s)) for s in COLS]
        lines.append(f"$M={c}$ & " + " & ".join(cells) + f" & {speed.get(c, '--')}\\\\")
    cells = [fmt(val(f"base_{AD}", s)) for s in COLS]
    lines.append("Adaptive & " + " & ".join(cells) + " & --\\\\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    write("capacity", "\n".join(lines))


if __name__ == "__main__":
    main_table()
    latency_table()
    ablation_table()
    stats_table()
    capacity_table()
