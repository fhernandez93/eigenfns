#!/usr/bin/env bash
# Build both PDFs from scratch. Usage: bash report/build.sh [--figures]
#   --figures  also regenerate every number/table/figure from the saved data (CPU, ~3 min)
# tectonic fetches the REVTeX 4.2 bundle itself; latexmk is the fallback.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/francisco/miniconda3/envs/lsu_ml/bin/python
TECTONIC=/home/francisco/miniconda3/bin/tectonic

if [[ "${1:-}" == "--figures" ]]; then
  for s in s01_spectra s02_localization s03_structure s04_kpm s05_levelstats s06_tables \
           fig1_structure_dos fig2_tiles fig3_xi_pr fig4_localization_stats figs_sm build_numbers; do
    echo "== $s"; (cd scripts && "$PY" "$s.py")
  done
fi

build() {
  local tex=$1
  if "$TECTONIC" --keep-logs --keep-intermediates -Z shell-escape=false "$tex" > "build_${tex%.tex}.log" 2>&1; then
    echo "tectonic: built ${tex%.tex}.pdf"
  else
    echo "tectonic failed for $tex (see build_${tex%.tex}.log); trying latexmk"
    latexmk -pdf -interaction=nonstopmode "$tex" >> "build_${tex%.tex}.log" 2>&1
  fi
  # zero undefined references / citations
  if grep -Ei "undefined (reference|citation)|Citation .* undefined|Reference .* undefined" "${tex%.tex}.log" "build_${tex%.tex}.log" 2>/dev/null | grep -v "^.*Rerun" | head -1 | grep -q .; then
    echo "WARNING: undefined references/citations in ${tex%.tex}"; grep -Ei "undefined" "${tex%.tex}.log" | head
    exit 2
  fi
}
build main.tex
build supplement.tex
for f in main.pdf supplement.pdf; do
  echo "$f: $(pdfinfo "$f" | awk '/^Pages:/{print $2}') pages"
done
