#!/usr/bin/env bash
# Rerun the sepsis example (Listing 3) under one or more Simpful versions,
# each in its own throw-away virtual environment.
#
# Usage:   bash check_old_simpful.sh 2.0.0 2.1.0 2.12.0
# Pick versions released around the paper (accepted 7 Oct 2020) from
# https://pypi.org/project/simpful/#history
# Results are appended to results/sepsis_version_check.txt
set -u
mkdir -p results
OUT=results/sepsis_version_check.txt
for V in "$@"; do
  ENV=".venv_simpful_$V"
  echo "--- simpful $V"
  python3 -m venv "$ENV"
  if "$ENV/bin/pip" install -q "simpful==$V" "numpy<2" scipy; then
    "$ENV/bin/python" sepsis_version_check.py 2>&1 | grep RESULT | tee -a "$OUT"
  else
    echo "RESULT simpful $V: could not be installed" | tee -a "$OUT"
  fi
  rm -rf "$ENV"
done
echo "Saved $OUT"
