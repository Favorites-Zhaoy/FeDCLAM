#! /bin/bash
loss="diceFIMce"
alpha=1.0
l_fim=0.01
fim_warmup=10

python main.py \
    --config 'config.prostate_mri.fedclam' \
    --loss "$loss" \
    --l_fim "$l_fim" \
    --fim_warmup "$fim_warmup" \
    --alpha "$alpha" \
    --beta 1 \
    --run_notes  "final" \
    --exp_name "${loss}_alpha${alpha}_lfim${l_fim}_warmup${fim_warmup}"
