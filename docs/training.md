# Training the segmentation model

OpenHeart's `Segment` step runs a model trained with standard nnU-Net v2. To train
or retrain one, build a dataset with OpenHeart and hand it to nnU-Net.

## 1. Build the dataset

From case folders that already hold reviewed segmentations:

```bash
openheart export nnunet /path/to/reviewed_cases --layout legacy \
    -o $nnUNet_raw --dataset-id 3 --name RBHSHare
```

or from CSV lists of `image_path,label_path` pairs (e.g. manual QC exports):

```bash
openheart export nnunet site_a.csv site_b.csv --base-dir /data/root \
    -o $nnUNet_raw --dataset-id 3 --name RBHSHare
```

Labels are written in the canonical convention (1 = LV, 2 = myocardium,
3 = RV; label 4 is merged into RV). Pairs whose image and label grids differ are
skipped and logged.

## 2. Train with nnU-Net

```bash
export nnUNet_raw=/path/to/nn_raw
export nnUNet_preprocessed=/path/to/nn_pre
export nnUNet_results=/path/to/nn_res

nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
nnUNetv2_train 3 2d 0          # one fold per GPU; repeat for folds 1-4
nnUNetv2_train 3 3d_fullres 0  # optional second configuration
nnUNetv2_find_best_configuration 3
```

## 3. Use it

Point `Segment` (or `openheart run --model`) at the training output folder:

```bash
openheart run cases/ -o out/ \
    --model $nnUNet_results/Dataset003_RBHSHare/nnUNetTrainer__nnUNetPlans__2d --folds 1
```

`--folds` defaults to every fold found (an ensemble). No `nnUNet_*` variables are
needed at inference time.
