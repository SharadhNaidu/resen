#!/usr/bin/env bash
# GPU training queue (sequential). Logs in runs/<name>/train.log
set -u
cd "$(dirname "$0")/.."
S="128:8:0.5,256:4:0.5"
run() { name=$1; shift; mkdir -p runs/$name; python training/train.py --out runs/$name --workers 6 "$@" > runs/$name/train.log 2>&1; echo "$name exit $?" >> runs/queue.log; }
run syn_light   --bench lolv2_syn  --init weights/retinexformer/LOL_v2_synthetic.pth --iters 8000 --schedule $S
run real_light  --bench lolv2_real --init weights/retinexformer/LOL_v2_real.pth      --iters 8000 --schedule $S
run v1_light    --bench lolv1      --init weights/retinexformer/LOL_v1.pth           --iters 6000 --schedule $S
run syn_abl_nopriors --bench lolv2_syn --init weights/retinexformer/LOL_v2_synthetic.pth --iters 3000 --schedule $S --no-priors
run syn_abl_plainloss --bench lolv2_syn --init weights/retinexformer/LOL_v2_synthetic.pth --iters 3000 --schedule $S --plain-loss
run syn_abl_full3k --bench lolv2_syn --init weights/retinexformer/LOL_v2_synthetic.pth --iters 3000 --schedule $S
echo ALL_DONE >> runs/queue.log
