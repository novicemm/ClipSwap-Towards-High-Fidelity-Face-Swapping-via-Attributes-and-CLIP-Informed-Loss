#!/bin/bash

python inference.py \
    --retina_path ./pretrained_ckpts/retinaface/RetinaFace-Res50.h5 \
    --arcface_path ./pretrained_ckpts/arcface/ArcFace-Res50.h5 \
    --model_path ./pretrained_ckpts/clipswap/clipswap.h5 \
    --target ./examples/target/6.jpg \
    --source ./examples/source/4.jpg \
    --output ./output/result.jpg
