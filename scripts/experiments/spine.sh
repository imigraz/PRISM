#!/usr/bin/env bash
# Spine: Vanderbilt -> Zurich -> Montreal -> UCL
# Commands for the experiments of the PRISM paper, one fold per call. The paper reports 5-fold cross-validation:
#   for f in 0 1 2 3 4; do FOLD=$f bash scripts/experiments/spine.sh; done
# Set the nnU-Net paths first (documentation/setting_up_paths.md) and prepare the data
# (scripts/data_preparation/README.md). Comment out what you do not need.
set -euo pipefail

FOLD=${FOLD:-0}
GPU=${GPU:-0}
TASKS="614 613 612 611"
COMMON="-f $FOLD -num_epoch 100 -save_interval 100 -s seg_outputs --store_csv -d $GPU"

# PRISM
nnUNet_train_PRISM 2d -t $TASKS $COMMON -layer_name conv_blocks_context.7 -num_samples_in_perc 1.0 -seed 1 \
    -sparsity_aug 0.99 -sparsity_replay 0.99 -DFP 0.27 --dedup_slices --exp_name prism

# Single-domain models (needed for forward transfer, FWT; the evaluation looks for --exp_name seq)
nnUNet_train_sequential 2d -t 613 $COMMON --exp_name seq
nnUNet_train_sequential 2d -t 612 $COMMON --exp_name seq
nnUNet_train_sequential 2d -t 611 $COMMON --exp_name seq

# Baselines
nnUNet_train_rehearsal 2d -t $TASKS $COMMON -samples_in_perc 1.0 --exp_name keep_all_data
nnUNet_train_sequential_all_data 2d -t $TASKS $COMMON --exp_name seq_all_data   # Table 6, joint training
nnUNet_train_sequential 2d -t $TASKS $COMMON --exp_name seq
nnUNet_train_ewc 2d -t $TASKS $COMMON --exp_name ewc                                   # lambda 0.4 (default)
nnUNet_train_mib 2d -t $TASKS $COMMON --exp_name mib                                   # alpha 0.9, lambda 1 (defaults)
nnUNet_train_TED 2d -t $TASKS $COMMON -ted_lambda 1e-3 --exp_name ted_1e-3
nnUNet_train_sequential_GIN 2d -t $TASKS $COMMON --exp_name gin
nnUNet_train_sequential_FIESTA 2d -t $TASKS $COMMON --exp_name gin_fat
nnUNet_train_cauaug 2d -t $TASKS $COMMON -cauaug_lambda 1e-2 --exp_name cauaug_1e-2   # CKNet probability 0.5 (default)
# FR2Seg is not part of this repository; its results were produced with code provided by its authors.

# Evaluation of all folds (run once after all folds are trained)
# nnUNet_evaluate_all -trainers nnUNetTrainerPRISM nnUNetTrainerSequential nnUNetTrainerRehearsal nnUNetTrainerSequentialAllData \
#     nnUNetTrainerEWC nnUNetTrainerMiB nnUNetTrainerTED nnUNetTrainerSequentialGIN nnUNetTrainerSequentialFIESTA \
#     nnUNetTrainerCauAug -t $TASKS -t 613 -t 612 -t 611
