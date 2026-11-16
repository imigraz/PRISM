#########################################################################################################
#------------------------ Feature rehearsal with sparsified features (PRISM base) ----------------------#
#########################################################################################################
"""Grandparent of nnUNetTrainerPRISM: per-domain training loop with encoder freezing and feature extraction.

After the first domain the network is frozen up to ``layer_name_for_feature_extraction`` and, after each domain
but the last, the features of the domain's training cases are extracted (``store_features``, implemented in
nnUNetTrainerFeatureRehearsalSparseDynamicMemory). After the last domain the features are extracted once more
for the storage benchmark and then deleted (``clean_up``).
"""
import os
import re
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.utilities.file_and_folder_operations import isfile, join
from nnunet.training.dataloading.dataset_loading import DataLoader3D
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer

from nnunet_ext.network_architecture.generic_UNet_prism import Generic_UNet_PRISM
from nnunet_ext.paths import default_plans_identifier
from nnunet_ext.training.DataLoader2DWithSlices import DataLoader2DWithSlices
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalTargetType
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}


FEATURE_PATH = "extracted_features"


class nnUNetTrainerFeatureRehearsalSparse(nnUNetTrainerMultiHead):
    """Sequential training over domains with a frozen encoder and replay of sparsified features."""
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='feature_rehearsal2', tasks_list_with_char=None,
                 target_type: FeatureRehearsalTargetType = FeatureRehearsalTargetType.GROUND_TRUTH,
                 num_rehearsal_samples_in_perc: float= 1.0,
                 layer_name_for_feature_extraction: str="",
                 mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False,
                 seed: int = 42, sparsity_aug: float = 0.95, sparsity_replay: float = 0.975):
        r"""Constructor. transfer_heads is always True for this trainer (the new head starts from the previous one).

        layer_name_for_feature_extraction: the network is frozen up to and including this layer after the first
            domain (the bottleneck freezes the whole encoder). Replay uses the first-level features regardless.
        sparsity_aug / sparsity_replay: CCD rates, passed to Generic_UNet_PRISM.set_sparsity.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, True, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)

        self.seed = seed
        self.target_type = target_type.DISTILLED_OUTPUT
        self.num_rehearsal_samples_in_perc = num_rehearsal_samples_in_perc
        self.layer_name_for_feature_extraction = layer_name_for_feature_extraction
        assert self.num_rehearsal_samples_in_perc > 0 and self.num_rehearsal_samples_in_perc <= 1, "samples_in_perc should be between 0 and 1: (0, 1]."
        assert self.layer_name_for_feature_extraction.startswith(("conv_blocks_context","td", "tu", "conv_blocks_localization"))
        assert self.layer_name_for_feature_extraction.count('.') == 1, "layer_name must have exactly 1 dot"
        self.num_feature_rehearsal_cases = 0        # store amount of feature_rehearsal cases. init with 0
        self.rehearse = False                       # variable for alternating schedule
        self.date_time = self.extract_datetime_and_suffix('_'.join(output_folder.split('_')))
        self.sparsity_aug = sparsity_aug
        self.sparsity_replay = sparsity_replay

        self.init_args = (split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16, save_interval, self.already_trained_on, use_progress, identifier, extension,
                          tasks_list_with_char, target_type, num_rehearsal_samples_in_perc, layer_name_for_feature_extraction, 
                          mixed_precision, save_csv, del_log, use_vit, self.vit_type,
                          version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA, do_SPT)

    def extract_datetime_and_suffix(self, path_str):
        """
        Extract datetime and everything after it from a complex path.
        For a pattern like: path/to/something_25.11.2024_08:56:53_test1
        Returns: '25.11.2024_08:56:53_test1'

        Args:
            path_str (str): The full path string

        Returns:
            str: The datetime and suffix string, or None if not found
        """
        # Convert to Path object and get the final directory name
        path = Path(path_str)
        final_dir = path.parts[-1]

        # Define the pattern for date_time (DD.MM.YYYY_HH:MM:SS) and capture everything after
        datetime_pattern = r'(\d{2}\.\d{2}\.\d{4}_\d{2}:\d{2}:\d{2}.*$)'

        # Search for the pattern
        match = re.search(datetime_pattern, final_dir)

        if match:
            return match.group(1)

        return None

    def _buffer_folders(self):
        """Sub-folders of the feature folder that the extraction writes into."""
        return ["gt", "features", "predictions", "feature_pkl"]

    def run_training(self, task, output_folder, build_folder=True):
        """Train on one domain; afterwards freeze the network and extract this domain's features."""
        # clear feature folder on first task!
        if self.tasks_list_with_char[0][0] == task:
            self.print_to_log_file("first task. deleting feature sets")
            for folder in self._buffer_folders():
                path = join(self.trained_on_path, self.extension,  FEATURE_PATH + self.date_time, folder)
                if not os.path.exists(path):
                    os.makedirs(path)
                else:
                    for f in os.listdir(path):
                        os.remove(join(path,f))
                assert len(os.listdir(path)) == 0

            assert self.was_initialized
            self.save_checkpoint(join(self.output_folder, "before_training.model"), False)
        else:
            # freeze encoder
            self.freeze_network()

        self.network.__class__ = Generic_UNet_PRISM
        self.network.set_sparsity(self.sparsity_aug, self.sparsity_replay)

        ret = super().run_training(task, output_folder, build_folder)

        if task == self.tasks_list_with_char[0][-1]:
            self.print_to_log_file("last task: extracting features only for the storage benchmark, then deleting them.")
            # No replay follows the last task; the extraction only measures the buffer size
            self.store_features(task)
            self.clean_up()
            return

        # freeze encoder
        self.freeze_network()

        # compute features, store them and update feature_rehearsal_dataloader
        self.store_features(task)
        return ret

    def freeze_network(self):
        """Freeze the network up to and including layer_name_for_feature_extraction and rebuild the optimizer.

        Frozen parameters get no gradients and frozen layers stay in eval mode (Generic_UNet_PRISM.train).
        """
        self.print_to_log_file("freeze network!")
        assert self.layer_name_for_feature_extraction.startswith(("conv_blocks_context", "td", "tu", "conv_blocks_localization"))

        self.network.freeze_layers(self.layer_name_for_feature_extraction)

        self.initialize_optimizer_and_scheduler()

    def _preprocess_multithreaded(self, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
        """Preprocess cases together with their ground-truth segmentations in worker processes (generator).
        Mostly copied from nnunet.inference.predict."""
        def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, output_files, segs_from_prev_stage, classes,
                             transpose_forward):

            errors_in = []
            for i, l in enumerate(list_of_lists):
                try:
                    output_file = output_files[i]
                    print("preprocessing", output_file)
                    d, _, dct = preprocess_fn(l)
                    if segs_from_prev_stage[i] is not None:
                        assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                            ".nii.gz"), "segs_from_prev_stage" \
                                        " must point to a " \
                                        "segmentation file " + str(segs_from_prev_stage[i])
                        seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                        # check to see if shapes match
                        img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                        assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                        "stage don't have the same pixel array " \
                                                                                        "shape! image: %s, seg_prev: %s" % \
                                                                                        (l[0], segs_from_prev_stage[i])
                        seg_prev = seg_prev.transpose(transpose_forward)
                        seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1)
                        assert(np.all(seg_reshaped.shape == d.shape[1:])), "error in segmentation shape probably"

                    """There is a problem with python process communication that prevents us from communicating objects 
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
                    filename or np.ndarray and will handle this automatically"""
                    print(d.shape)
                    seg_shape = np.prod(seg_reshaped.shape) if seg_reshaped is not None else 0
                    if np.prod(d.shape) + seg_shape > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                        print(
                            "This output is too large for python process-process communication. "
                            "Saving output temporarily to disk")
                        np.save(output_file[:-7] + ".npy", d)
                        d = output_file[:-7] + ".npy"
                        np.save(output_file[:-7] + "_seg.npy", seg_reshaped)
                        seg_reshaped = output_file[:-7] + "_seg.npy"

                    q.put((output_file, (d, dct), seg_reshaped))
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as e:
                    print("error in", l)
                    print(e)
                    errors_in.append(l)
            q.put("end")
            if len(errors_in) > 0:
                print("There were some errors in the following cases:", errors_in)
                print("These cases were ignored.")
            else:
                print("This worker has ended successfully, no errors to report")
        
        
        if segs_from_prev_stage is None:
            segs_from_prev_stage = [None] * len(list_of_lists)

        num_processes = min(len(list_of_lists), num_processes)

        classes = list(range(1, self.num_classes))
        assert isinstance(self, nnUNetTrainer)
        q = Queue(1)
        processes = []
        for i in range(num_processes):
            pr = Process(target=preprocess_save_to_queue, args=(self.preprocess_patient, q,
                                                                list_of_lists[i::num_processes],
                                                                output_files[i::num_processes],
                                                                segs_from_prev_stage[i::num_processes],
                                                                classes, self.plans['transpose_forward']))
            pr.start()
            processes.append(pr)

        try:
            end_ctr = 0
            while end_ctr != num_processes:
                item = q.get()
                if item == "end":
                    end_ctr += 1
                    continue
                else:
                    yield item

        finally:
            for p in processes:
                if p.is_alive():
                    p.terminate()  # this should not happen but better safe than sorry right
                p.join()

            q.close()


    def get_basic_generators(self, use_all_data=False):
        """nnU-Net data loaders; in 2D, DataLoader2DWithSlices also returns the slice index of each patch."""
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr,
                                 self.basic_generator_patch_size if not use_all_data else self.patch_size,
                                 self.patch_size, self.batch_size, False,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2DWithSlices(self.dataset_tr,
                                 self.basic_generator_patch_size if not use_all_data else self.patch_size,
                                 self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2DWithSlices(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val


