#!/bin/sh
# usage: build.sh <name> <FOLD> <EPOCHS> <LIMIT> <WARM_EPOCHS> <BS> <ACC> <BB_LR> <HEAD_LR> <ARCH> <MODEL_SOURCE> <RES> [LABEL_FILE] [LABEL_DATASET] [PER_FINDING_HEAD 0|1] [MIRROR 0|1]
#  MODEL_SOURCE e.g. ohmahler91/torchvision-convnext/pytorch/convnext-tiny-imagenet1k-v1/1  (effective batch = BS*ACC, keep 16)
#  LABEL_FILE   llm_labels_v4_blend.csv (default) or e.g. ts_labels_a50.csv (then LABEL_DATASET must be a dataset slug that holds it)
NAME=$1; FOLD=$2; EPOCHS=$3; LIMIT=$4; WARM=$5; BS=$6; ACC=$7; BBLR=$8; HLR=$9; ARCH=${10}; MSRC=${11}; RES=${12}; LF=${13:-llm_labels_v4_blend.csv}; LDS=${14:-stevenleehans/rsna-knee-llm-report-labels}; PF=${15:-0}; MIR=${16:-0}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
# cache shards to mount: env CACHE_SRCS="slug1,slug2,..." (default = the 280 px cache); e.g. 384 px: CACHE_SRCS=$(for i in 0 1 2 3 4 5; do printf "rsna-knee-09-cache2-s%spx384," $i; done | sed 's/,$//')
CACHE_SRCS=${CACHE_SRCS:-rsna-knee-09-cache2-s0,rsna-knee-09-cache2-s1,rsna-knee-09-cache2-s2}
KS=$(echo "$CACHE_SRCS" | sed "s/[^,]*/\"$KUSER\/&\"/g")
{ printf 'FOLD, EPOCHS, K_UNFREEZE, LIMIT, WARM_EPOCHS, BS, ACC, BB_LR, HEAD_LR, ARCH, RES, LABEL_FILE, PF, MIRROR = %s, %s, 0, %s, %s, %s, %s, %s, %s, "%s", %s, "%s", %s, %s\nimport os\nimport numpy as np\nimport pandas as pd\n' "$FOLD" "$EPOCHS" "$LIMIT" "$WARM" "$BS" "$ACC" "$BBLR" "$HLR" "$ARCH" "$RES" "$LF" "$PF" "$MIR"; cat ../../src/knee_head.py; printf '\n'; cat ../../src/knee_cnn.py; cat driver.py; } > cnn3.py
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/rsna-knee-18-cnn3-$NAME", "title": "rsna-knee-18-cnn3-$NAME", "code_file": "cnn3.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": false,
 "dataset_sources": ["$LDS"], "competition_sources": ["rsna-knee-abnormality-detection"],
 "kernel_sources": [$KS],
 "model_sources": ["$MSRC"]}
JSON
