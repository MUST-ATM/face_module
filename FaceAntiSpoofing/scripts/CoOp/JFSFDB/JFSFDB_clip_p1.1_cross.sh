#!/bin/bash
# custom config

GPU_IDS='0'
DATASET=JFSFDB
DATA=/homedata/must/hcyuan/Datasets/JFSFDB
PROTOCOL=p1.1_spoofing_cross@FaceSpoofing@FaceSpoofing@FaceSpoofing
PREPROCESS=resize_crop_rotate_flip   ### resize_crop_rotate_flip_ColorJitter
OUTPUT=/homedata/must/hcyuan/test
TRAINER=CoOp
VERSION=VL         # V or VL
PROMPT=class  # class, engineering, ensembling
CFG=vit_b16        # config file

CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=1  # number of shots (1, 2, 4, 8, 16)
CSC=False # class-specific context (False or True)


for SEED in 1
do
    DIR=${OUTPUT}/${TRAINER}/${DATASET}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    python train.py \
    --gpu_ids ${GPU_IDS} \
    --protocol ${PROTOCOL} \
    --preprocess ${PREPROCESS} \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}
    # fi
done