#!/bin/bash
# Regenerate all paper figures from the saved evaluation JSONs in ../results/.
#
# The manuscript source is not part of this release (the paper is under review), so this
# script builds figures only. Once main.tex is added, follow the figure step with:
#   pdflatex main && bibtex main && pdflatex main && pdflatex main
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python}"

$PY generate_figures.py
$PY generate_pareto.py

# These two additionally need the COCO images and V*Bench; see ../DATA.md.
$PY generate_vstar_qual.py || echo "skipped fig_vstar_qual (needs V*Bench)"
$PY generate_pope_qual.py  || echo "skipped fig_pope_qual (needs COCO images)"

echo "figures written to $(pwd)/"
