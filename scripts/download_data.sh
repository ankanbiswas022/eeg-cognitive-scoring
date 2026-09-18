#!/usr/bin/env bash
# Download the PhysioNet "EEG During Mental Arithmetic Tasks" dataset (EEGMAT v1.0.0).
# 36 subjects x (3 min rest + 1 min arithmetic), 500 Hz, 19 EEG channels + ECG. ~330 MB.
# Zyma et al. 2019, Data 4(1):14.  https://physionet.org/content/eegmat/1.0.0/
set -euo pipefail
OUT="${1:-data/raw/eegmat}"
BASE="https://physionet.org/files/eegmat/1.0.0"
mkdir -p "$OUT"; cd "$OUT"
curl -sSO "$BASE/subject-info.csv"
for i in $(seq -w 0 35); do
  for k in 1 2; do
    f="Subject${i}_${k}.edf"
    [ -s "$f" ] || curl -sSO "$BASE/$f"
  done
done
echo "downloaded $(ls *.edf | wc -l) EDF files to $OUT"
