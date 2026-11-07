#!/usr/bin/env bash
# Runs after queue.sh: generalist (app) model on deduplicated LOL + LSRW.
cd "$(dirname "$0")/.."
until grep -q ALL_DONE runs/queue.log 2>/dev/null; do sleep 30; done
mkdir -p runs/general_light
python training/train.py --bench generalist --init weights/retinexformer/LOL_v2_real.pth \
  --iters 8000 --schedule "128:8:0.5,256:4:0.5" --gamma-aug --noise-aug --workers 6 --val-every 2000 \
  --out runs/general_light > runs/general_light/train.log 2>&1
echo "general_light exit $?" >> runs/queue.log
echo ALL2_DONE >> runs/queue.log
