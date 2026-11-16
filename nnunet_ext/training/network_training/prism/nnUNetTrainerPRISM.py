#########################################################################################################
#------------------------------ nnU-Net trainer for PRISM (continual segmentation) ---------------------#
#########################################################################################################
"""PRISM trainer.

Trains a 2D nnU-Net on a sequence of domains. After the first domain the encoder is frozen up to
``-layer_name``, and from each finished domain a small buffer of sparsified first-level feature maps is kept.
Training then alternates between batches of the current domain and replay batches from the buffer:

* current-domain batches: GIN and FAT augmentation of the images (each with probability 25%,
  independently), and input sparsification of the features as augmentation (rate ``sparsity_aug``,
  applied in the network);
* replay batches: the stored feature maps (sparsified with ``sparsity_replay`` when they were extracted)
  are passed through the rest of the network, and the loss is a label-free distillation (MSE over the
  deep-supervision outputs) against the model from the end of the previous domain. With ``use_kd=False``
  the segmentation loss on the stored predictions is used instead (only then are predictions stored).

Buffer handling (feature extraction, patient-stratified selection, storage benchmark) lives in the parent
classes nnUNetTrainerFeatureRehearsalSparseDynamicMemory and nnUNetTrainerFeatureRehearsalSparse.
"""
import copy
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch import nn
from torch.cuda.amp import autocast

from nnunet_ext.network_architecture.GenericUNet_ import Generic_UNet_
from nnunet_ext.network_architecture.generic_UNet_prism import Generic_UNet_PRISM
from nnunet_ext.paths import default_plans_identifier
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalTargetType
from nnunet_ext.training.FeatureRehearsalDatasetDM import FeatureRehearsalDataset, FeatureSelectionMethod
from nnunet_ext.training.feature_io import FeatureStorage
from nnunet_ext.training.network_training.feature_rehearsal_sparse_dm.nnUNetTrainerFeatureRehearsalSparseDynamicMemory import \
    nnUNetTrainerFeatureRehearsalSparseDynamicMemory
from nnunet_ext.utilities.helpful_functions import GINGroupConv2D, FIESTA

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}


FEATURE_PATH = "extracted_features"


class nnUNetTrainerPRISM(nnUNetTrainerFeatureRehearsalSparseDynamicMemory):
    """PRISM: replay of sparsified first-level features with label-free distillation, plus GIN and FAT."""

    # Probabilities of the two image augmentations on a current-domain batch (fixed at 25% in the paper).
    GIN_PROBABILITY = 0.25
    FAT_PROBABILITY = 0.25

    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='prism', tasks_list_with_char=None,
                 target_type: FeatureRehearsalTargetType = FeatureRehearsalTargetType.GROUND_TRUTH,
                 num_rehearsal_samples_in_perc: float = 1.0,
                 layer_name_for_feature_extraction: str = "",
                 mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False, seed: int = 42,
                 sparsity_aug: float = 0.95, sparsity_replay: float = 0.975, num_features_in_perc: float = 0.05,
                 use_kd: bool = True, dedup_slices: bool = False):
        r"""Constructor of the PRISM trainer (2D nnU-Net).

        sparsity_aug / sparsity_replay: CCD rates of input sparsification as augmentation / of the stored features.
        num_features_in_perc: fraction of a domain's extracted feature maps kept in the buffer (-DFP); at least
            30 per domain are kept.
        use_kd: replay with label-free distillation (default); False uses the segmentation loss instead.
        dedup_slices: keep at most one patch per image slice in the replay buffer.
        """
        self.target_type = target_type.DISTILLED_OUTPUT
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char,
                         target_type, num_rehearsal_samples_in_perc, layer_name_for_feature_extraction, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, transfer_heads, ViT_task_specific_ln, do_LSA,
                         do_SPT, network, use_param_split, seed, sparsity_aug, sparsity_replay)

        layer, id = self.layer_name_for_feature_extraction.split('.')
        id = int(id)

        if layer == "conv_blocks_context":
            num_features = id + 1
        elif layer == "td":
            num_features = id + 2
        else:
            num_features = len(self.network.conv_blocks_context)

        # Replay with distillation needs no stored targets (the previous-domain model provides them); only
        # use_kd=False replays with a segmentation loss on the stored predictions.
        self.dataset_dm = FeatureRehearsalDataset(
            join(self.trained_on_path, self.extension, FEATURE_PATH + self.date_time),
            self.deep_supervision_scales,
            FeatureRehearsalTargetType.NONE if use_kd else FeatureRehearsalTargetType.DISTILLED_OUTPUT,
            num_features,
            selection_method=FeatureSelectionMethod.RANDOM,
            retention_ratio=num_features_in_perc,
            random_state=self.seed,
            dedup_slices=dedup_slices,
            # Only level 0 is replayed (feature_forward); deeper levels are not written.
            feature_storage=FeatureStorage(levels=1, predictions=not use_kd),
        )

        # FAT (Fourier/Bezier intensity augmentation, class FIESTA)
        self.fiesta = FIESTA()
        self.prev_network = None
        self.use_kd = use_kd

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        """One iteration. During training on the second and later domains, current-domain and replay batches
        alternate. Only mixed precision (fp16) is supported."""
        if not isinstance(self.network, Generic_UNet_PRISM):
            self.network.__class__ = Generic_UNet_PRISM
            self.network.set_sparsity(self.sparsity_aug, self.sparsity_replay)

        rehearse = False

        # Replay only while training (not during evaluation) and not on the first domain
        if do_backprop and len(self.mh_network.heads.keys()) > 1:
            rehearse = self.rehearse
            self.rehearse = not self.rehearse

        if rehearse:
            try:
                data_dict = next(self.feature_rehearsal_dataiter)
            except StopIteration:
                self.feature_rehearsal_dataiter = iter(self.feature_rehearsal_dataloader)
                data_dict = next(self.feature_rehearsal_dataiter)

        else:
            data_dict = next(data_generator)

        data = data_dict['data']  # torch.Tensor (normal),    list[torch.Tensor] (rehearsal)
        target = data_dict.get('target')  # list[torch.Tensor]; None for replay batches with distillation

        if not rehearse:
            properties = data_dict['properties']
            slice_indices = data_dict['slice_indices']
        else:
            properties = None

        data = maybe_to_torch(data)
        if target is not None:
            target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            if target is not None:
                target = to_cuda(target)

        # Image augmentation of current-domain batches: GIN, then FAT, each with its own draw
        if not rehearse and random.random() < self.GIN_PROBABILITY:
            with torch.no_grad():
                augmentor = GINGroupConv2D()
                data = augmentor(data)
                del augmentor

        if not rehearse and random.random() < self.FAT_PROBABILITY:
            with torch.no_grad():
                data = self.fiesta.augment_batch(data)

        self.optimizer.zero_grad()

        kl = 0.0

        if not self.fp16:
            raise NotImplementedError("nnUNetTrainerPRISM supports mixed precision training only (do not use --fp32)")

        with autocast():
            if rehearse:
                output = self.network.feature_forward(data)
                if self.use_kd:
                    kl = self.compute_knowledge_distillation_loss(data, output)
            else:
                output = self.network(data)

            del data
            if not no_loss:
                l = 0.0
                if rehearse and self.use_kd:
                    l = kl
                else:
                    l = self.loss(output, target)

        if do_backprop:
            self.amp_grad_scaler.scale(l).backward()
            self.amp_grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.amp_grad_scaler.step(self.optimizer)
            self.amp_grad_scaler.update()

        if run_online_evaluation:
            self.run_online_evaluation(output, target, properties, slice_indices)

        del target

        # -- Update the Multi Head Network after one iteration only if backprop is performed (during training) -- #
        if do_backprop:
            self.mh_network.update_after_iteration()

        # -- Return the loss -- #
        if not no_loss:
            if detach:
                l = l.detach().cpu().numpy()
            return l

    def compute_knowledge_distillation_loss(self, data, current_output, alpha_kd: float = 1.0) -> torch.Tensor:
        """Label-free distillation: MSE between the deep-supervision outputs of the current network and of the
        model from the end of the previous domain, both run from the same stored features, averaged over the
        deep-supervision levels.
        """
        prev_network = self.prev_network
        prev_network.eval()

        with torch.no_grad():
            prev_output = prev_network.feature_forward(data)

        mse_loss = 0.0
        for curr_out, prev_out in zip(current_output, prev_output):
            mse_loss += F.mse_loss(curr_out, prev_out)

        mse_loss /= len(current_output)

        return alpha_kd * mse_loss

    def _buffer_folders(self):
        """Blosc2 layout: no ground-truth patches, only a foreground flag per patch (see feature_io)."""
        return list(self.dataset_dm.feature_storage.folders())

    def run_training(self, task, output_folder, build_folder=True):
        """Train on one domain, then keep a copy of the network as the distillation teacher for the next one."""
        self.print_to_log_file("PRISM input sparsification (CCD): augmentation %s, replay %s"
                               % (self.sparsity_aug, self.sparsity_replay))
        ret = super().run_training(task, output_folder, build_folder)

        # Model from the end of this domain: teacher for label-free distillation on the next domain
        self.prev_network = copy.deepcopy(self.network)

        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        """Inference without deep supervision. Input sparsification is not applied at inference."""
        if not isinstance(self.network, Generic_UNet_PRISM):
            self.network.__class__ = Generic_UNet_PRISM
            self.network.set_sparsity(self.sparsity_aug, self.sparsity_replay)

        ds = self.network.do_ds
        self.network.do_ds = False
        ret = self.predict_preprocessed_data_return_seg_and_softmax_2(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax_2(self, data: np.ndarray, do_mirroring: bool = True,
                                                             mirror_axes: Tuple[int] = None,
                                                             use_sliding_window: bool = True, step_size: float = 0.5,
                                                             use_gaussian: bool = True,
                                                             pad_border_mode: str = 'constant',
                                                             pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                             verbose: bool = True, mixed_precision: bool = True) -> \
        Tuple[np.ndarray, np.ndarray]:
            """Sliding-window prediction with the network in eval mode (restored afterwards)."""

            if pad_border_mode == 'constant' and pad_kwargs is None:
                pad_kwargs = {'constant_values': 0}

            if do_mirroring and mirror_axes is None:
                mirror_axes = self.data_aug_params['mirror_axes']

            if do_mirroring:
                assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                          "was done without mirroring"

            valid = list((SegmentationNetwork, Generic_UNet_, nn.DataParallel))
            assert isinstance(self.network, tuple(valid))

            current_mode = self.network.training
            self.network.eval()
            ret = self.network.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                          use_sliding_window=use_sliding_window, step_size=step_size,
                                          patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                          use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                          pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                          mixed_precision=mixed_precision)

            self.network.train(current_mode)
            return ret
