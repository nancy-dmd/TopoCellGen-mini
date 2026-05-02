# TopoCellGen Minimal Reproduction Report

## Goal

Reproduce the BRCA layout generation pipeline from `Melon-Xu/TopoCellGen` with the smallest practical resource footprint:

- single GPU
- single test sample
- pre-trained checkpoint only
- no full training

## Local Changes

The upstream repository uses hard-coded paths. For reproducibility in this workspace, the following changes were made:

1. `generate_layout_brca.py`
   - converted hard-coded paths into CLI arguments
   - added `--test-patch-path`, `--results-root-path`, and `--max-test-files`
   - preserved the original BRCA model hyperparameters

2. `tools/prepare_brca_minimal.py`
   - added a small helper to convert BRCA-M2C `labels/*.txt` annotations into the `256x256x3` `.npy` point-map format expected by TopoCellGen
   - supports `--limit` to build a minimal subset

## Minimal Dataset

Source dataset:

- `../Dataset-BRCA-M2C`

Minimal test subset prepared for reproduction:

- `data/brca_minimal/test_dataset`

Current subset size:

- 1 sample

Prepared sample:

- `TCGA-A8-A099-01Z-00-DX1_29501_22501_1000_1000_0.93.npy`

Ground-truth cell counts in this sample:

- class 0: 11
- class 1: 77
- class 2: 27

## Environment

Runtime environment used for the reproduction run:

- conda env: `11`
- Python: `3.12.11`
- PyTorch: `2.7.0+cu126`
- CUDA runtime reported by PyTorch: `12.6`
- GPU used for generation: `GPU 4`

## Commands

Dataset preparation:

```bash
python tools/prepare_brca_minimal.py \
  --labels-dir ../Dataset-BRCA-M2C/labels \
  --split-file ../Dataset-BRCA-M2C/brca_ds_test.txt \
  --output-dir data/brca_minimal/test_dataset \
  --limit 1
```

Generation:

```bash
python generate_layout_brca.py \
  --model_path checkpoints/brca_m2c.pt \
  --test_patch_path data/brca_minimal/test_dataset \
  --results_root_path results_minimal \
  --max_test_files 1
```

## Run Result

Status:

- Success

Observed runtime:

- roughly 12 seconds for the single-sample generation pass based on the logger timestamps

Cell-count comparison:

- ground truth: `[11, 77, 27]` -> total `115`
- generated: `[10, 76, 28]` -> total `114`
- absolute total-count difference: `1`
- absolute per-class differences: `[1, 1, 1]`

Outputs:

- [results_minimal/2026-05-02/15-26-04/hyperparams.json](/data/dingmeidan/xingqiu-auto-code/TopoCellGen/results_minimal/2026-05-02/15-26-04/hyperparams.json)
- [results_minimal/2026-05-02/15-26-04/cell_counts.json](/data/dingmeidan/xingqiu-auto-code/TopoCellGen/results_minimal/2026-05-02/15-26-04/cell_counts.json)
- [results_minimal/2026-05-02/15-26-04/npy/TCGA-A8-A099-01Z-00-DX1_29501_22501_1000_1000_0.93_gen_114.npy](/data/dingmeidan/xingqiu-auto-code/TopoCellGen/results_minimal/2026-05-02/15-26-04/npy/TCGA-A8-A099-01Z-00-DX1_29501_22501_1000_1000_0.93_gen_114.npy)
- [results_minimal/2026-05-02/15-26-04/img/TCGA-A8-A099-01Z-00-DX1_29501_22501_1000_1000_0.93_gen_114.png](/data/dingmeidan/xingqiu-auto-code/TopoCellGen/results_minimal/2026-05-02/15-26-04/img/TCGA-A8-A099-01Z-00-DX1_29501_22501_1000_1000_0.93_gen_114.png)

## Notes

- This is a pipeline-level smoke reproduction, not a full paper-number reproduction.
- The objective is to verify that the official pre-trained generation path runs end to end with minimal resources.
- The upstream code path imports training-only topological loss modules at import time. To make minimal inference reproduction robust, `guided_diffusion/gaussian_diffusion.py` was adjusted so those topology-loss dependencies are optional unless the training loss branch is actually used.
