# RSNA Knee Abnormality Detection: a 2.5D CNN ensemble trained on weak labels

Code and write-up for my entry to the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) competition on Kaggle.
It documents what I built, what worked and what did not. It was a learning-oriented entry, not a competitive one: private score **0.874** (public 0.886); the top private score was 0.960.

**Approach:** end-to-end ResNet-50, ConvNeXt-tiny and ConvNeXt-small on 2.5D MRI stacks, 5 folds each, probability-averaged, trained on LLM-derived soft labels.
**Main lessons:** (1) end-to-end fine-tuning beat frozen features; (2) most of the gain came from ensembling different architectures; (3) resolution, mirror augmentation, per-finding heads, label blending and weight averaging gave nothing measurable; (4) a small set of held-out true labels predicted the private score to within 0.002.

## 1. Task and constraints
- Knee MRI; predict 12 findings per study (ACL, MCL, medial and lateral meniscus, medial / lateral / patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, contusion, fracture). Scoring is ROC-AUC based.
- Code competition: notebooks only, no internet, at most 9 h per run.
- Training labels were not given as ground truth. Of 4,407 training studies only 58 have true labels; for the rest I used public LLM-extracted soft labels (see Credits).
- Evaluation: 5-fold cross-validation against the soft labels, plus the 58 true-label ("gold") studies, which no model ever trained on.

## 2. Data handling
- One series per plane; slices ordered by physical position; a fixed 150 mm field-of-view crop.
- 2.5D input: 3-slice triplets as RGB channels, 11 position groups per plane, 3 planes (axial, coronal, sagittal).
- Cached at 280 px; models trained at 224 px from the cache.

## 3. Models
- End-to-end fine-tuning of ResNet-50, ConvNeXt-tiny and ConvNeXt-small (ImageNet weights), with a slice-attention head shared across the 11 groups.
- 8 epochs, effective batch 16 (gradient accumulation), backbone LR 1e-4, head LR 1e-3, 5 folds per architecture, no test-time augmentation.
- Final ensemble (19 models): 5 folds x 3 architectures, plus 2 mirror-augmented ConvNeXt-tiny and 2 ConvNeXt-tiny at 384 px; plain probability average.

## 4. Results
| submission | public | private |
|---|---|---|
| frozen DINOv2-small + attention head | 0.822 | - |
| single ResNet-50 | 0.839 | - |
| 5 x ResNet-50 | 0.864 | - |
| 5 x ResNet-50 + 5 x ConvNeXt-tiny | 0.882 | - |
| 15 models (adds 5 x ConvNeXt-small) | 0.884 | 0.873 |
| 19 models (adds mirror and 384 px members) | 0.886 | 0.874 |

**Calibration.** On the 58 gold studies the 15-model ensemble scored 0.872 (never trained on them); its private score was 0.873. The public score ran about 0.012 above the gold estimate. With 58 studies the gold set is noisy (about +/-0.02), so this match is encouraging, not proof.

## 5. What helped
- End-to-end fine-tuning over frozen features (frozen features capped at about 0.80 on the soft labels, whatever the backbone).
- ConvNeXt over ResNet: pooled out-of-fold 0.824 against 0.809; the two together 0.835.
- Ensembling gave most of the leaderboard gain: one ResNet 0.839, five 0.864, ten mixed 0.882.

## 6. What did not help
| idea | result |
|---|---|
| native 280 px instead of 224 | +0.001 on the soft labels, -0.005 on gold, 1.6x slower |
| 384 px (2 folds) | +0.014 on fold 0, -0.008 on fold 1; ensemble gain about +0.005 |
| frozen backbone size, series coverage, pooling variants, label blending | none moved the 0.80 ceiling |
| per-finding attention heads | +0.003, within noise |
| EMA / stronger augmentation | no gain |
| averaging the best 3 epochs | no consistent gain |
| mirror augmentation for laterality | single models +0.007 / -0.005 on two folds; as an ensemble member about +0.003 |
| EfficientNet-B3 | diverged (NaN) in the first epoch; dropped |

Laterality check: only about half of the studies carry a laterality tag. Out-of-fold accuracy differed by +0.010 between right and left knees (95% interval -0.005 to +0.024), so I found no measurable laterality problem.

## 7. Limits
- The soft labels score 0.893 on the gold studies, which caps what a model trained on them can reach.
- Gold-58 is too small to separate models that differ by less than about 0.02.
- I did not reproduce the strongest public recipes (larger pretrained backbones at 384 px with per-finding heads). The top public notebooks combine other participants' trained weights; I did not use those.
- Untested ideas: a larger pretrained backbone, teacher-student relabelling from out-of-fold predictions, more windows per study.
- Compute: all training ran on Kaggle's free tier (a weekly quota of about 30 GPU hours, T4 GPUs). One fold took about 65 minutes for ResNet-50, about 80 minutes for ConvNeXt-tiny and about 4 hours for ConvNeXt-tiny at 384 px, which is why the 384 px and mirror-augmentation tests were run on two folds only and why I kept the models at 224 px with no test-time augmentation. Larger backbones and longer schedules were not tried.

## 8. Repository layout
| folder | content |
|---|---|
| `src/` | DICOM loading and 2.5D preparation (`knee_prep.py`, `knee_prep2.py`), the slice-attention head (`knee_head.py`), the CNN modules (`knee_cnn.py`), and the analysis scripts behind the negative results (probes, pooling, series coverage, label scoring, out-of-fold analysis) |
| `tests/` | unit tests for the preparation code (synthetic DICOM files) |
| `kaggle/cache/` | builds the 2.5D cache notebook (280 or 384 px, in shards) |
| `kaggle/train_resnet/`, `kaggle/train_cnn3/` | build the training notebooks (ResNet-50; ConvNeXt and others, with options for per-finding heads and mirror augmentation) |
| `kaggle/infer/` | builds the inference notebook that averages every checkpoint among its sources |

## 9. Reproducing
No images, labels or trained weights are included. To reproduce you need the competition data (accept the rules and download it), the public label datasets listed below, and a Kaggle account.
1. Put the competition files and the label tables under `data/` (the analysis scripts read `data/train.csv`, `data/train_series.csv` and the label files).
2. Each `kaggle/*/build.sh` concatenates the modules from `src/` into one self-contained Kaggle script and writes a `kernel-metadata.json`. Set `KAGGLE_USER` first, then push with `kaggle kernels push -p <folder>`.
   - `kaggle/cache/build.sh <name> <SHARD> <NSHARDS> <LIMIT> [CACHE_PX]` builds the 2.5D cache.
   - `kaggle/train_cnn3/build.sh ...` trains one fold of one architecture (usage is in the script header).
   - `kaggle/infer/build.sh <name> <ARCH> <RES> <slug,slug,...>` builds the inference notebook.
3. The notebooks reference Kaggle datasets and models by slug (the cache shards you built, the label dataset, the ImageNet weights); adapt the slugs to your own account.

## 10. Data and licence note
The competition data is provided under the RSNA MIRA data licence and must not be redistributed, so this repository contains none of it, and no labels derived from the reports.

## 11. Credits
- Soft labels used for training: **stevenleehans**, `rsna-knee-llm-report-labels` (file `llm_labels_v4_blend`). Other public label sets were used only in the label-quality analysis: laymond (Qwen3-8B weak labels), Pilkwang Kim (`rsna-knee-llm-labels`) and lixin73.
- Public notebooks and datasets by Pilkwang Kim, wguesdon, mattiaangeli, nartaa and dreaddevelopment were read for ideas (text only; none of their weights or data were used).
- ImageNet weights for ResNet-50 and ConvNeXt from Kaggle Models (ohmahler91/torchvision-resnet and torchvision-convnext).
- RSNA and Kaggle for the competition and data.
