#!/bin/bash
# Insert the public code URL into the Data availability statement, rebuild both PDFs and the release package.
# usage: bash code/set_code_url.sh https://github.com/<org>/crowdsort
set -e
URL="$1"
[[ "$URL" =~ ^https://[A-Za-z0-9./_-]+$ ]] || { echo "give a full https URL"; exit 1; }
cd /mnt/ssd2/paper/paper_1/paper/tex
grep -q "REPLACE-ME" backmatter.tex || { echo "placeholder already replaced"; exit 1; }
sed -i "s#https://github.com/REPLACE-ME/crowdsort#${URL}#" backmatter.tex
for f in main main_review; do
  pdflatex -interaction=nonstopmode $f.tex >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode $f.tex >/dev/null 2>&1 || true
  grep -c "^! " $f.log | xargs echo "$f errors:"
done
cp main.pdf ../CrowdSort_manuscript.pdf; cp main_review.pdf ../CrowdSort_manuscript_review.pdf
cd /mnt/ssd2/paper/paper_1 && .venv-sfx/bin/python code/make_release.py >/dev/null && echo "release rebuilt"
grep -o "\\\\url{[^}]*}" paper/tex/backmatter.tex
