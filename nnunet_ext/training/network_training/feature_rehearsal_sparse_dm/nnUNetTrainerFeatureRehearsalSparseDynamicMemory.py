#########################################################################################################
#----------- Feature rehearsal with sparse features and a dynamic, patient-stratified buffer -----------#
#########################################################################################################
"""Parent of nnUNetTrainerPRISM: extraction of sparsified features after each domain into a replay buffer
that keeps a fraction of each domain's feature maps, plus the storage benchmark of that buffer."""
import math
import os
import random
import shutil

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, maybe_mkdir_p, subfiles

from nnunet_ext.inference import predict
from nnunet_ext.paths import default_plans_identifier
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalTargetType
from nnunet_ext.training.FeatureRehearsalDatasetDM import FeatureRehearsalDataset, BalancedFeatureRehearsalDataLoader, \
    FeatureRehearsalMultiDataset, FeatureSelectionMethod
from nnunet_ext.training.network_training.feature_rehearsal_sparse.nnUNetTrainerFeatureRehearsalSparse import \
    nnUNetTrainerFeatureRehearsalSparse
from nnunet_ext.training.storage_benchmark import log_storage_summary, save_storage_plot

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

FEATURE_PATH = "extracted_features"


class nnUNetTrainerFeatureRehearsalSparseDynamicMemory(nnUNetTrainerFeatureRehearsalSparse):
    """Sparse feature rehearsal with a dynamic memory: after each domain, features are extracted and the buffer
    keeps a patient-stratified random subset of them (see FeatureRehearsalDatasetDM)."""
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True,
                 stage=None, unpack_data=True, deterministic=True, fp16=False, save_interval=5,
                 already_trained_on=None, use_progress=True, identifier=default_plans_identifier,
                 extension='feature_rehearsal_dm', tasks_list_with_char=None,
                 target_type: FeatureRehearsalTargetType = FeatureRehearsalTargetType.GROUND_TRUTH,
                 num_rehearsal_samples_in_perc: float = 1.0,
                 layer_name_for_feature_extraction: str = "",
                 mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False,
                 transfer_heads=True, ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None,
                 use_param_split=False, seed: int = 42, sparsity_aug: float = 0.95,
                 sparsity_replay: float = 0.975):
        r"""Constructor; the retention_ratio of 0.05 set here is replaced by nnUNetTrainerPRISM."""
        self.target_type = target_type.DISTILLED_OUTPUT
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                         unpack_data, deterministic, fp16, save_interval, already_trained_on, use_progress,
                         identifier, extension, tasks_list_with_char, target_type, num_rehearsal_samples_in_perc,
                         layer_name_for_feature_extraction, mixed_precision, save_csv, del_log, use_vit, vit_type,
                         version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT, network,
                         use_param_split, seed, sparsity_aug, sparsity_replay)

        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)

        if layer == "conv_blocks_context":
            num_features = id + 1
        elif layer == "td":
            num_features = id + 2
        else:
            num_features = len(self.network.conv_blocks_context)

        self.dataset_dm = FeatureRehearsalDataset(
            join(self.trained_on_path, self.extension, FEATURE_PATH + self.date_time),
            self.deep_supervision_scales,
            self.target_type,
            num_features,
            selection_method=FeatureSelectionMethod.RANDOM,
            retention_ratio=0.05,
            random_state=self.seed,
        )

        # accumulate benchmark results across tasks for plotting
        self.storage_benchmarks = []   # list of dicts, one per completed task
        self.storage_task_names  = []  # matching task name strings

    def store_features(self, task: str):
        """Extract sparsified features of this domain's training cases into the buffer and rebuild the
        replay dataloader."""
        self.print_to_log_file("extract features!")

        with torch.no_grad():
            input_folder = os.path.join(
                os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', task, 'imagesTr'
            )

            expected_num_modalities = load_pickle(self.plans_file)['num_modalities']
            case_ids = predict.check_input_folder_and_return_caseIDs(
                input_folder, expected_num_modalities
            )

            assert set(self.dataset_tr.keys()).issubset(set(case_ids)), \
                "training cases missing from imagesTr: " + str(self.dataset_tr.keys()) + " " + str(case_ids)
            case_ids = list(self.dataset_tr.keys())

            case_ids = random.sample(
                case_ids, round(len(case_ids) * self.num_rehearsal_samples_in_perc)
            )
            self.num_feature_rehearsal_cases += len(case_ids)

            self.print_to_log_file(
                "the following cases will be used for feature rehearsal:" + str(case_ids)
            )

            output_folder = join(
                self.trained_on_path, self.extension, FEATURE_PATH + self.date_time
            )
            maybe_mkdir_p(output_folder)

            output_files  = [join(output_folder, i + ".nii.gz") for i in case_ids]
            all_files     = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
            list_of_lists = [
                [join(input_folder, i) for i in all_files
                 if i[:len(j)].startswith(j) and len(i) == (len(j) + 12)]
                for j in case_ids
            ]

            output_filenames = output_files[0::1]
            assert len(list_of_lists) == len(output_filenames)

            for o in output_filenames:
                dr, f = os.path.split(o)
                if len(dr) > 0:
                    maybe_mkdir_p(dr)

            ground_truth_segmentations = []
            for input_path_in_list in list_of_lists:
                input_path       = input_path_in_list[0]
                input_path_array = input_path.split('/')
                assert input_path_array[-2] == "imagesTr"
                input_path_array[-2] = "labelsTr"
                assert input_path_array[-1].endswith('_0000.nii.gz')
                input_path_array[-1] = input_path_array[-1][:-12] + '.nii.gz'
                segmentation_path = "/" + join(*input_path_array)
                ground_truth_segmentations.append(segmentation_path)

            preprocessing_generator = self._preprocess_multithreaded(
                list_of_lists[0::1], output_filenames,
                segs_from_prev_stage=ground_truth_segmentations
            )

            for preprocessed in preprocessing_generator:
                output_filename, (d, dct), gt_segmentation = preprocessed

                if isinstance(d, str):
                    assert isinstance(gt_segmentation, str)
                    data = np.load(d);  os.remove(d);  d = data
                    s    = np.load(gt_segmentation);  os.remove(gt_segmentation)
                    gt_segmentation = s

                assert np.all(d.shape[1:] == gt_segmentation.shape), \
                    str(d.shape) + " " + str(gt_segmentation.shape)

                current_mode = self.network.training
                self.network.eval()
                ds = self.network.do_ds
                self.network.do_ds = False

                self.network.predict_3D(
                    d,
                    do_mirroring      = False,
                    mirror_axes       = self.data_aug_params['mirror_axes'],
                    use_sliding_window= True,
                    step_size         = 0.5,
                    patch_size        = self.patch_size,
                    regions_class_order = self.regions_class_order,
                    use_gaussian      = True,
                    pad_border_mode   = 'constant',
                    pad_kwargs        = None,
                    all_in_gpu        = False,
                    verbose           = True,
                    mixed_precision   = True,
                    ground_truth_segmentation     = gt_segmentation,
                    feature_dir       = output_filename[:-7],
                    layer_name_for_feature_extraction = self.layer_name_for_feature_extraction,
                    feature_storage   = self.dataset_dm.feature_storage,
                )

                self.network.do_ds = ds
                self.network.train(current_mode)

            # Update dataset and dataloader
            if hasattr(self, 'feature_rehearsal_dataloader'):
                del self.feature_rehearsal_dataloader
                del self.feature_rehearsal_dataiter
            self.dataset_dm.update_features()
            self._run_and_store_benchmark(task)
            dataset    = FeatureRehearsalMultiDataset(
                self.dataset_dm,
                math.ceil((self.batch_size * 250) / len(self.dataset_dm))
            )
            dataloader = BalancedFeatureRehearsalDataLoader(
                dataset, batch_size=int(self.batch_size), num_workers=0,
                pin_memory=True,
                deep_supervision_scales=self.deep_supervision_scales,
                deterministic=self.deterministic,
            )
            self.feature_rehearsal_dataloader = dataloader
            self.feature_rehearsal_dataiter   = iter(dataloader)

    def _run_and_store_benchmark(self, task: str) -> dict:
        """Run benchmark for the most recently added dataset, store result, update plot.

        Passes self.dataset_tr.keys() as fold_cases so that NIfTI reference sizes
        are counted for exactly the cases seen during training — the true upper bound
        and the same set used by store_features above.
        """
        dataset_id = self.dataset_dm.current_dataset_id - 1

        report = self.dataset_dm.benchmark_storage(
            dataset_id=dataset_id,
            task_name=task,
            fold_cases=set(self.dataset_tr.keys()),
        )

        if report:
            model_size_MB = sum(
                p.numel() * p.element_size() for p in self.network.parameters()
            ) / 1e6
            report['model_size_MB'] = round(model_size_MB, 2)

            self.storage_benchmarks.append(report)
            self.storage_task_names.append(task)
            self._log_storage_summary(report)
            self._save_storage_plot()

        return report

    def _log_storage_summary(self, report: dict):
        """See storage_benchmark.log_storage_summary."""
        log_storage_summary(report, self.storage_benchmarks, self.print_to_log_file)

    def _save_storage_plot(self):
        """See storage_benchmark.save_storage_plot; writes into the feature folder."""
        save_storage_plot(self.storage_benchmarks, self.storage_task_names,
                          join(self.trained_on_path, self.extension, FEATURE_PATH + self.date_time),
                          self.print_to_log_file)


    def clean_up(self):
        """Clean up all feature-related folders including backup folders."""
        base_path = join(
            self.trained_on_path, self.extension, FEATURE_PATH + self.date_time
        )

        summary = self.dataset_dm.sparsity_analyzer.get_summary()
        for dataset_id, stats in summary.items():
            self.print_to_log_file(f"\nDataset {dataset_id}:")
            self.print_to_log_file(f"Average share of non-positive values: {stats['sparsity_ratio'] * 100:.2f}%")
            self.print_to_log_file(f"- Negative values: {stats['negative_ratio'] * 100:.2f}%")
            self.print_to_log_file(f"- Zero values: {stats['zero_ratio'] * 100:.2f}%")
            self.print_to_log_file(f"Samples analyzed: {stats['samples_analyzed']}")

        self._save_storage_plot()

        for folder in self._buffer_folders():
            path = join(base_path, folder)
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                for item in os.listdir(path):
                    item_path = join(path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            assert len(os.listdir(path)) == 0

        for item in os.listdir(base_path):
            if item.startswith('duplicate_backup_dataset_'):
                backup_path = join(base_path, item)
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
                    print(f"Removed backup folder: {backup_path}")

        backup_folders = [
            f for f in os.listdir(base_path)
            if f.startswith('duplicate_backup_dataset_')
            and os.path.isdir(join(base_path, f))
        ]
        assert len(backup_folders) == 0, "Not all backup folders were removed"
        print("Cleanup completed successfully")