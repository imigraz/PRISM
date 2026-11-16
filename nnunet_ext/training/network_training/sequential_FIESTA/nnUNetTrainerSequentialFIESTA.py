#########################################################################################################
#------------- nnUNet trainer for sequential training with GIN and FAT augmentation (GIN+FAT) ----------#
#########################################################################################################

import random

import torch
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast

from nnunet_ext.paths import default_plans_identifier
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead
from nnunet_ext.utilities.helpful_functions import FIESTA, GINGroupConv2D

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {}

# Each augmentation is applied to a training batch independently with this probability (as in PRISM).
GIN_PROBABILITY = 0.25
FAT_PROBABILITY = 0.25

class nnUNetTrainerSequentialFIESTA(nnUNetTrainerMultiHead):
    # -- Trains n tasks sequentially using transfer learning -- #
    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='sequential_FIESTA', tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False):
        r"""Sequential training with GIN and FAT augmentation of the input images (the GIN+FAT baseline).
            GIN (Ouyang et al., 2023) and the Fourier augmentative transformer of FIESTA (Oh et al., 2024) are each
            applied to a batch with probability 0.25. As in the Sequential trainer, transfer_heads is always True.
        """
        # -- Initialize using parent class -- #
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, True, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)

        self.fiesta = FIESTA()

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, detach=True, no_loss=False):
        r"""This function runs an iteration based on the underlying model. It returns the detached or undetached loss.
            The undetached loss might be important for methods that have to extract gradients without always copying
            the run_iteration function.
            NOTE: The calling class needs to set self.network according to the desired task, this is not done in this
                  function but expected by the user.
        """
        # -- Run iteration as usual --> copied and modified from nnUNetTrainerV2 -- #
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        with torch.no_grad():
            if random.random() < GIN_PROBABILITY:
                data = GINGroupConv2D()(data)
            if random.random() < FAT_PROBABILITY:
                data = self.fiesta.augment_batch(data)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                if not no_loss:
                    l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            if not no_loss:
                l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        # -- Update the Multi Head Network after one iteration only if backprop is performed (during training) -- #
        if do_backprop:
            self.mh_network.update_after_iteration()

        # -- Return the loss -- #
        if not no_loss:
            if detach:
                l = l.detach().cpu().numpy()
            return l