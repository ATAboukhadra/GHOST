#!/usr/bin/env bash
set -euo pipefail

# Add your hashcodes here.
HASHES=(
  25e67133f
)

for hash in "${HASHES[@]}"; do
  echo "[RUN] ${hash}"
  python export_arctic.py --hash "${hash}"
done
