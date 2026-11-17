"""Overleaf package of the manuscript as one file: main.tex with every section and table inlined, plus refs.bib,
elsarticle.cls, elsarticle-harv.bst, qualitative_supplement.tex and figs/ (the figure PDFs both documents use). Writes paper/overleaf_crowdsort
and its zip.

python make_overleaf.py
"""
import os
import re
import shutil

BASE = "/mnt/ssd2/paper/paper_1/paper/tex"
OUT = "/mnt/ssd2/paper/paper_1/paper/overleaf_crowdsort"


def flat(name):
    s = open(f"{BASE}/{name}.tex").read().rstrip("\n")
    return re.sub(r"^([^%\n]*?)\\input\{([^}]*)\}", lambda m: m.group(1) + flat(m.group(2)), s, flags=re.M)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(f"{OUT}/figs")
    m = flat("main")
    assert "\\input{" not in re.sub(r"%.*", "", m)
    figs = {g if g.endswith(".pdf") else g + ".pdf"
            for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", re.sub(r"(?<!\\)%.*", "", m))}
    for g in figs:
        shutil.copy2(f"{BASE}/{g}", f"{OUT}/figs/{os.path.basename(g)}")
    m = m.replace("\\begin{document}", "\\graphicspath{{figs/}}\n\\begin{document}", 1)
    open(f"{OUT}/main.tex", "w").write(m + "\n")
    for f in ("refs.bib", "elsarticle.cls", "elsarticle-harv.bst"):
        shutil.copy2(f"{BASE}/{f}", f"{OUT}/{f}")
    supplement = os.path.join(BASE, "qualitative_supplement.tex")
    supplement_figs = []
    if os.path.exists(supplement):
        s = open(supplement).read()
        supplement_figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", s)
        for g in supplement_figs:
            source = os.path.normpath(os.path.join(BASE, g))
            shutil.copy2(source, f"{OUT}/figs/{os.path.basename(g)}")
            s = s.replace("{" + g + "}", "{figs/" + os.path.basename(g) + "}")
        open(f"{OUT}/qualitative_supplement.tex", "w").write(s)
    if os.path.exists(OUT + ".zip"):
        os.remove(OUT + ".zip")
    shutil.make_archive(OUT, "zip", OUT)
    print("main.tex", len(m.splitlines()), "lines,", len(figs), "main figures,",
          len(supplement_figs), "supplementary figures ->", OUT + ".zip")


if __name__ == "__main__":
    main()
