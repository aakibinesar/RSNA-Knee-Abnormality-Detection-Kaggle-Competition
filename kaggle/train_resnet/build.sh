#!/bin/sh
# usage: build.sh <name> <FOLD> <EPOCHS> <LIMIT> <WARM_EPOCHS> <ACC> <BB_LR> <HEAD_LR> <ARCH> <WEIGHTS_VARIANT> <RES>
NAME=$1; FOLD=$2; EPOCHS=$3; LIMIT=$4; WARM=$5; ACC=$6; BBLR=$7; HLR=$8; ARCH=$9; VAR=${10}; RES=${11}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
{ printf 'FOLD, EPOCHS, K_UNFREEZE, LIMIT, WARM_EPOCHS, ACC, BB_LR, HEAD_LR, ARCH, RES = %s, %s, 0, %s, %s, %s, %s, %s, "%s", %s\nimport os\nimport numpy as np\nimport pandas as pd\n' "$FOLD" "$EPOCHS" "$LIMIT" "$WARM" "$ACC" "$BBLR" "$HLR" "$ARCH" "$RES"; cat ../../src/knee_head.py; cat driver.py; } > cnn.py
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/rsna-knee-15-cnn-$NAME", "title": "rsna-knee-15-cnn-$NAME", "code_file": "cnn.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": false,
 "dataset_sources": ["stevenleehans/rsna-knee-llm-report-labels"], "competition_sources": ["rsna-knee-abnormality-detection"],
 "kernel_sources": ["$KUSER/rsna-knee-09-cache2-s0", "$KUSER/rsna-knee-09-cache2-s1", "$KUSER/rsna-knee-09-cache2-s2"],
 "model_sources": ["ohmahler91/torchvision-resnet/pytorch/$VAR/1"]}
JSON
