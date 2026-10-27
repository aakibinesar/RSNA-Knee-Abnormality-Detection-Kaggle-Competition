#!/bin/sh
# usage: build.sh <name> <default ARCH> <RES> <comma-separated CNN kernel slugs, each optionally slug:arch> [TTA 0|1] [CACHE_PX default 280]
#  e.g. ./build.sh mix1 resnet50 224 rsna-knee-15-cnn-f1,rsna-knee-18-cnn3-cx0:convnext_tiny 1
NAME=$1; ARCH=$2; RES=$3; SRCS=$4; TTA=${5:-0}; PX=${6:-280}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
AMAP=$(echo "$SRCS" | tr ',' '\n' | grep ':' | sed 's/\(.*\):\(.*\)/"\1": "\2"/' | paste -sd, -)
SLUGS=$(echo "$SRCS" | sed 's/:[^,]*//g')
{ printf 'ARCH, RES, TTA, CACHE_PX = "%s", %s, %s, %s\nARCH_MAP = {%s}\nimport os\nimport numpy as np\nimport pandas as pd\n' "$ARCH" "$RES" "$TTA" "$PX" "$AMAP"; cat ../../src/knee_prep.py | grep -v '^import numpy\|^import pandas\|^import os$'; printf '\n'; grep -v '^from knee_prep import\|^import os$\|^import numpy\|^import pandas' ../../src/knee_prep2.py; printf '\n'; cat ../../src/knee_head.py; printf '\n'; cat ../../src/knee_cnn.py; cat driver.py; } > infer_cnn.py
KS=$(echo "$SLUGS" | sed "s/[^,]*/\"$KUSER\/&\"/g")
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/rsna-knee-16-infer-cnn-$NAME", "title": "rsna-knee-16-infer-cnn-$NAME", "code_file": "infer_cnn.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": false,
 "dataset_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"],
 "kernel_sources": [$KS], "model_sources": []}
JSON
