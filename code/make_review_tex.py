"""Single-column review version of the manuscript: paper/tex/main_review.tex from paper/tex/main.tex
(elsarticle preprint, 11pt, line numbers; full-width tables become ordinary tables scaled to the text width).

python make_review_tex.py
"""
import re

T = "/mnt/ssd2/paper/paper_1/paper/tex"
s = open(f"{T}/main.tex").read()
s = re.sub(r"\\documentclass\[[^\]]*\]\{elsarticle\}", r"\\documentclass[preprint,11pt,times,authoryear]{elsarticle}\n\\usepackage{lineno}\n\\linenumbers", s, count=1)
s = s.replace("\\raggedbottom\n", "")
s = s.replace("\\begin{document}", "\\sloppy\n% full-width tables become ordinary tabulars that adjustbox scales to the text width\n"
              "\\renewenvironment{tabular*}[2]{\\begin{tabular}{#2}}{\\end{tabular}}\n"
              "% the front matter (five affiliations, abstract) must fit on the title page\n"
              "\\usepackage{etoolbox}\\AtBeginEnvironment{abstract}{\\small}\n\\begin{document}", 1)
open(f"{T}/main_review.tex", "w").write(s)
print("written", f"{T}/main_review.tex")
