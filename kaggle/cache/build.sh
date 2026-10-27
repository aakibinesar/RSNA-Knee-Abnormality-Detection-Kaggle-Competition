#!/bin/sh
# usage: build.sh <name> <SHARD> <NSHARDS> <LIMIT> [CACHE_PX default 280]   (name e.g. s0px384 when CACHE_PX differs, so slugs stay distinct)
NAME=$1; SHARD=$2; NSHARDS=$3; LIMIT=$4; PX=${5:-280}
KUSER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
cd "$(dirname "$0")"
{ printf 'SHARD, NSHARDS, LIMIT, CACHE_PX = %s, %s, %s, %s\n' "$SHARD" "$NSHARDS" "$LIMIT" "$PX"; cat ../../src/knee_prep.py; printf '\n'; grep -v '^from knee_prep import' ../../src/knee_prep2.py; cat driver.py; } > cache2.py
cat > kernel-metadata.json <<JSON
{"id": "$KUSER/rsna-knee-09-cache2-$NAME", "title": "rsna-knee-09-cache2-$NAME", "code_file": "cache2.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": false, "enable_internet": false,
 "dataset_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
JSON
