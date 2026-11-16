# Dataset preparation

These scripts turn the public datasets into the nnU-Net (v1) tasks used in the PRISM paper. No data is
distributed with this repository: download each dataset from its source and accept its licence or data use
agreement first.

All scripts write an nnU-Net raw task folder (`imagesTr/`, `labelsTr/`, `dataset.json`). Write it to
`$nnUNet_raw_data_base/nnUNet_raw_data/` (see `documentation/setting_up_paths.md`), then run the planning and
preprocessing step shown for each setting.

## Hippocampus: Task097 to Task099

Domain order in the paper: DecathHip → Dryad → HarP (`-t 97 98 99`).

| Task | Dataset | Source | Licence / access |
|---|---|---|---|
| Task097_DecathHip | Medical Segmentation Decathlon, Task04_Hippocampus (Antonelli et al., 2022) | http://medicaldecathlon.com | CC BY-SA 4.0 |
| Task098_Dryad | Kulaga-Yoskovitz et al., 2015, Scientific Data, doi:10.1038/sdata.2015.59 | Dryad, https://doi.org/10.5061/dryad.gc72v | see the Dryad record |
| Task099_HarP | EADC-ADNI Harmonized Hippocampal Protocol labels (Boccardi et al., 2015) on ADNI T1 images | labels: http://www.hippocampal-protocol.net; images: https://adni.loni.usc.edu | ADNI data use agreement (application required) |

```bash
OUT=$nnUNet_raw_data_base/nnUNet_raw_data

# Decathlon folder with dataset.json, imagesTr/, labelsTr/. Anterior and posterior hippocampus are merged into one label.
python scripts/data_preparation/hippocampus/create_DecathHip.py --input /data/Task04_Hippocampus --output $OUT/Task097_DecathHip

# Folder with the subject folders s01 ... s25 (files <subject>_t1w_standard_defaced_MNI.nii.gz and
# <subject>_hippolabels_hres_{L,R}_MNI.nii.gz). Left and right labels are merged; volumes are cropped to
# 96 x 64 x 64 voxels around the hippocampus.
python scripts/data_preparation/hippocampus/create_Dryad.py --input /data/dryad/mri_dataset --output $OUT/Task098_Dryad

# ADNI download (one folder per subject, e.g. 002_S_0295) and the HarP label release (Released_data_NII_v1.3).
# Subjects are processed in sorted order and renamed HarP_000, HarP_001, ...; cropping as for Dryad.
python scripts/data_preparation/hippocampus/create_HarP.py --adni /data/ADNI --labels /data/Released_data_NII_v1.3 --output $OUT/Task099_HarP

nnUNet_plan_and_preprocess -t 97 98 99
```

## Spine: Task611 to Task614

Spinal cord gray matter challenge (Prados et al., 2017, NeuroImage), training data of the four sites. Source:
http://niftyweb.cs.ucl.ac.uk/program.php?p=CHALLENGE (download after accepting the challenge's data usage licence agreement).

Put the challenge training files into one folder as `imagesTr/site<N>-sc<NN>-image.nii.gz` and
`labelsTr/site<N>-sc<NN>-mask-r<R>.nii.gz`. The script merges the four rater masks by consensus (a voxel gets the
class at least three raters agree on, otherwise background; labels 0 background, 1 white matter, 2 gray
matter) and writes one task per site:

| Task | Challenge site | Site name in the paper |
|---|---|---|
| Task611_Spine_A | site1 | UCL |
| Task612_Spine_B | site2 | Montreal |
| Task613_Spine_C | site3 | Zurich |
| Task614_Spine_D | site4 | Vanderbilt |

Site numbers as in Prados et al. (2017): site 1 UCL, site 2 Polytechnique Montreal, site 3 University of Zurich,
site 4 Vanderbilt University. The paper's domain order Vanderbilt → Zurich → Montreal → UCL is `-t 614 613 612 611`.

```bash
python scripts/data_preparation/spine/prepare_spine.py --input /data/scgm_training --output $nnUNet_raw_data_base/nnUNet_raw_data
nnUNet_plan_and_preprocess -t 611 612 613 614
```

Use an output folder other than the input folder.

## Prostate: Task211 to Task216

We used the cross-site prostate data provided by the DeSAM authors (Gao et al., 2024): download it via the links
in the DeSAM repository, https://github.com/yifangao112/DeSAM (section "Training DeSAM on cross-site prostate
dataset"). It contains the six sites of the multi-site prostate dataset (Liu et al., 2021;
https://liuquande.github.io/SAML/), resampled to an in-plane size of 288 × 288. We do not ship a script for this step: convert
the six sites into the nnU-Net tasks below. The domain order in the paper is `-t 211 212 213 214 215 216`.

| Task | Site |
|---|---|
| Task211_Prostate-RUNMC | RUNMC |
| Task212_Prostate-BMC | BMC |
| Task213_Prostate-I2CVB | I2CVB |
| Task214_Prostate-UCL | UCL |
| Task215_Prostate-BIDMC | BIDMC |
| Task216_Prostate-HK | HK |

## Train / test split (FOCUS only)

The PRISM experiments use nnU-Net's five-fold cross-validation on the tasks above and evaluate on each fold's
validation cases. `create_3_split.py` is the script of the FOCUS paper, which additionally held out a test set: it
creates a new task that links to the preprocessed data of an existing one and moves a fraction of each fold's
training cases (fixed seed) to a `test` list in `splits_final.pkl`.

```bash
python scripts/data_preparation/create_3_split.py --source Task097_DecathHip --target Task197_DecathHip --test_fraction 0.3
```

The FOCUS experiments used `Task011_Prostate-BIDMC → Task111`, `Task012_Prostate-I2CVB → Task112`,
`Task013_Prostate-HK → Task113`, `Task015_Prostate-UCL → Task115`, `Task097_DecathHip → Task197`,
`Task098_Dryad → Task198` and `Task099_HarP → Task199`, each with `--test_fraction 0.3`.
