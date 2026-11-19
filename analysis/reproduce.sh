#!/bin/sh
# Recompute every derived number, table and figure of the report from the stored answers and verdicts.
# No model, no GPU, no paid API, no dataset download. Needs uv and poppler (pdftotext); mlx wheels exist for Apple silicon.
# Usage: sh analysis/reproduce.sh
set -e
cd "$(dirname "$0")/.."

step() { echo "== $1"; shift; "$@"; }
# Pinned: the numbers use these libraries; the figures use matplotlib 3.9.2.
py() { uv run --no-project --python 3.12 --with mlx-lm --with numpy --with scipy --with rouge-score --with nltk python "$@"; }

# Result files of the six-seed comparisons, the jury and the budgets
step clsprobe-h38 py code/clsprobe.py report-h38 > /dev/null
step clsprobe-h63 py code/clsprobe.py report-h63 > /dev/null
step clsprobe-h64 py code/clsprobe.py report-h64 > /dev/null
step clsprobe-h65 py code/clsprobe.py report-h65 > /dev/null
step clsprobe-h68 py code/clsprobe.py report-h68 > /dev/null
step clsprobe-h68-paired py code/clsprobe.py report-h68-paired > /dev/null
step selrule-h35 py code/selrule.py report-h35 --n 300 > /dev/null
step jury-d py code/jury_d.py report > /dev/null
# Shaded tables, then the figures, then the check of the report against the result files
step tables py analysis/tables_gen.py
step figures uv run --no-project --python 3.12 --with matplotlib==3.9.2 --with numpy python analysis/make_figures.py
step check-numbers py analysis/check_numbers.py
