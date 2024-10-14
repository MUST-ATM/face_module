#!/bin/bash
# custom config

GPU_IDS='1'
DATASET=UniAttackData
DATA=/mnt/UniAttackData
PROTOCOL=p1@UniAttack@UniAttack@UniAttack
PREPROCESS=resize_crop_rotate_flip   ### resize_crop_rotate_flip_ColorJitter
OUTPUT=/root/FAS/Baseline/CLIP-VL
TRAINER=CLIP
VERSION=VL         # V or VL
PROMPT=class  # class, engineering, ensembling
CFG=vit_b16        # config file

for SEED in 1
do
    DIR=${OUTPUT}/${TRAINER}@${PROMPT}/${CFG}/${PROTOCOL}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    python train.py \
    --gpu_ids ${GPU_IDS} \
    --root ${DATA} \
    --protocol ${PROTOCOL} \
    --preprocess ${PREPROCESS} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --version ${VERSION} \
    --prompt ${PROMPT} \
    # fi
done