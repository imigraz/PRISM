#########################################################################################################
# ------------------This class represents the nnUNet trainer for CauAug training.--------------------#
#########################################################################################################
from matplotlib import pyplot as plt

from nnunet_ext.paths import default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet_ext.training.network_training.multihead.nnUNetTrainerMultiHead import nnUNetTrainerMultiHead
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.utilities.nd_softmax import softmax_helper
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
import random
import os
import numpy as np
from pathlib import Path
from matplotlib.transforms import blended_transform_factory

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {
    'use_cknet': (bool, True, 'Enable Cross Kernel Network for anatomy causality'),
    'use_ftgen': (bool, True, 'Enable Fourier Transform Generator for domain causality'),
    'lambda_d': (float, 0.001, 'Weight for distillation loss'),
    'cknet_prob': (float, 0.5, 'Probability of applying CKNet augmentation'),
}

DOMAIN_KNOWLEDGE_PATH = "domain_knowledge_archive"


class nnUNetTrainerCauAug(nnUNetTrainerMultiHead):
    """
    Integrates CauAug (Causality-Adjusted Data Augmentation) into nnUNet lifelong learning.

    Key components:
    - CKNet: Cross Kernel Network for anatomy causality (reduces absorption bias)
    - FTGen: Fourier Transform Generator for domain causality (reduces retention bias)
    """

    def __init__(self, split, task, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, save_interval=5, already_trained_on=None, use_progress=True,
                 identifier=default_plans_identifier, extension='cauaug', tasks_list_with_char=None, mixed_precision=True,
                 save_csv=True, del_log=False, use_vit=False, vit_type='base', version=1, split_gpu=False, transfer_heads=True,
                 ViT_task_specific_ln=False, do_LSA=False, do_SPT=False, network=None, use_param_split=False,
                 use_cknet=True, use_ftgen=True, lambda_d=0.001, cknet_prob=0.5):
        r"""
        Constructor for CauAug trainer.

        Args:
            use_cknet (bool): Enable Cross Kernel Network for anatomy causality
            use_ftgen (bool): Enable Fourier Transform Generator for domain causality
            lambda_d (float): Weight for old knowledge distillation loss
            cknet_prob (float): Probability of applying CKNet augmentation
        """
        super().__init__(split, task, plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic,
                         fp16, save_interval, already_trained_on, use_progress, identifier, extension, tasks_list_with_char, mixed_precision,
                         save_csv, del_log, use_vit, vit_type, version, split_gpu, True, ViT_task_specific_ln, do_LSA, do_SPT,
                         network, use_param_split)

        # CauAug configuration
        self.use_cknet  = use_cknet
        self.use_ftgen  = use_ftgen
        self.lambda_d   = lambda_d
        self.cknet_prob = cknet_prob

        # Domain knowledge archive for FTGen
        self.magnitude_dataset = []
        self.phase_dataset     = []
        self.std_dataset       = []
        self.mean_dataset      = []

        # Storage tracking
        self.storage_benchmarks: list = []
        self.storage_task_names: list = []

    # ─────────────────────────────────────────────────────────────────────────
    def run_training(self, task, output_folder, build_folder=True):
        """
        Main training loop with domain knowledge archiving after each task.
        """
        if self.tasks_list_with_char[0][0] == task:
            self.print_to_log_file("First task - initializing domain knowledge archive")
            self._initialize_domain_archive()
            assert self.was_initialized
            self.save_checkpoint(join(self.output_folder, "before_training.model"), False)

        ret = super().run_training(task, output_folder, build_folder)

        # Archive domain knowledge for ALL tasks (including the last one)
        if self.use_ftgen:
            self.print_to_log_file(f"Archiving domain knowledge for task {task}")
            self._archive_domain_knowledge(task)

        return ret

    def _initialize_domain_archive(self):
        """Initialize/clear the domain knowledge archive directory."""
        archive_path = join(self.trained_on_path, self.extension, DOMAIN_KNOWLEDGE_PATH)
        maybe_mkdir_p(archive_path)
        for f in os.listdir(archive_path):
            os.remove(join(archive_path, f))

    def _archive_domain_knowledge(self, task: str):
        """
        Store domain-specific features (magnitude, phase, mean, std) from current training data.
        This implements the domain knowledge archive described in the CauAug paper.
        """
        self.network.eval()

        magnitude_accumulator = None
        phase_accumulator     = None
        std_accumulator       = None
        mean_accumulator      = None
        count = 0

        tr_gen = self.get_basic_generators()[0]

        with torch.no_grad():
            for _ in range(1, len(self.dataset_tr)):
                try:
                    data_dict = next(tr_gen)
                except StopIteration:
                    tr_gen = self.get_basic_generators()[0]
                    data_dict = next(tr_gen)

                inputs = maybe_to_torch(data_dict['data'])
                if torch.cuda.is_available():
                    inputs = to_cuda(inputs)

                std  = torch.std(inputs,  dim=(2, 3), keepdim=True)
                mean = torch.mean(inputs, dim=(2, 3), keepdim=True)
                std  = torch.mean(std,  dim=0)
                mean = torch.mean(mean, dim=0)

                fft       = torch.fft.fft2(inputs)
                magnitude = torch.abs(fft)
                phase     = torch.angle(fft)
                magnitude = torch.mean(magnitude, dim=0)
                phase     = torch.mean(phase,     dim=0)

                if count == 0:
                    magnitude_accumulator = magnitude
                    phase_accumulator     = phase
                    std_accumulator       = std
                    mean_accumulator      = mean
                else:
                    magnitude_accumulator = magnitude_accumulator * (count / (count + 1)) + magnitude / (count + 1)
                    phase_accumulator     = phase_accumulator     * (count / (count + 1)) + phase     / (count + 1)
                    std_accumulator       = std_accumulator       * (count / (count + 1)) + std       / (count + 1)
                    mean_accumulator      = mean_accumulator      * (count / (count + 1)) + mean      / (count + 1)
                count += 1

        if magnitude_accumulator is None:
            return

        self.magnitude_dataset.append(magnitude_accumulator.cpu())
        self.phase_dataset.append(phase_accumulator.cpu())
        self.std_dataset.append(std_accumulator.cpu())
        self.mean_dataset.append(mean_accumulator.cpu())

        self.print_to_log_file(f"Domain knowledge archived. Total domains: {len(self.magnitude_dataset)}")
        self.network.train()

        # Benchmark and plot after each domain is archived
        self._run_and_store_benchmark(task)

    # ─────────────────────────────────────────────────────────────────────────
    def _run_and_store_benchmark(self, task: str) -> dict:
        """Compute buffer sizes for the most recently archived domain, store and plot."""
        domain_idx = len(self.magnitude_dataset) - 1

        report = self._compute_buffer_benchmark(domain_idx, task)
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

    def _compute_buffer_benchmark(self, domain_idx: int, task: str) -> dict:
        """
        Measure the in-memory (and theoretical on-disk) size of the four archived
        tensors for a single domain.

        NIfTI reference counts are restricted to the cases in the current
        cross-validation fold (train + val) so that the numbers are comparable
        across trainers that use different fold subsets.

        Returns a dict with per-component MB values and an optional nifti_stats block.
        """
        def tensor_mb(t: torch.Tensor) -> float:
            return t.numel() * t.element_size() / 1e6

        mag_MB  = tensor_mb(self.magnitude_dataset[domain_idx])
        ph_MB   = tensor_mb(self.phase_dataset[domain_idx])
        std_MB  = tensor_mb(self.std_dataset[domain_idx])
        mean_MB = tensor_mb(self.mean_dataset[domain_idx])
        total_MB = mag_MB + ph_MB + std_MB + mean_MB

        report = {
            'domain_idx':      domain_idx,
            'magnitude_MB':    round(mag_MB,   4),
            'phase_MB':        round(ph_MB,    4),
            'std_MB':          round(std_MB,   6),
            'mean_MB':         round(mean_MB,  6),
            'total_buffer_MB': round(total_MB, 4),
        }

        # ── NIfTI reference ───────────────────────────────────────────────
        nifti_stats = None
        raw_base = os.environ.get("nnUNet_raw_data_base", None)
        if raw_base is not None:
            images_dir = join(raw_base, "nnUNet_raw_data", task, "imagesTr")
            if os.path.isdir(images_dir):
                all_nii_files = sorted(
                    f for f in os.listdir(images_dir) if f.endswith(".nii.gz")
                )

                # ── restrict to current fold (train + val) ────────────────
                fold_cases = set(self.dataset_tr.keys())
                if fold_cases is not None:
                    nii_files = [
                        f for f in all_nii_files
                        if f.rsplit("_", 1)[0] in fold_cases
                    ]
                    self.print_to_log_file(
                        f"_compute_buffer_benchmark: fold {self.fold} — "
                        f"using {len(nii_files)}/{len(all_nii_files)} NIfTI files"
                    )
                else:
                    nii_files = all_nii_files

                if nii_files:
                    total_nifti_MB = sum(
                        os.path.getsize(join(images_dir, f)) / 1e6 for f in nii_files
                    )
                    n_cases = len({f.rsplit("_", 1)[0] for f in nii_files})

                    nifti_dtype_str        = "unknown"
                    total_uncompressed_f32 = None
                    total_uncompressed_f16 = None
                    try:
                        import nibabel as nib
                        nifti_dtype_str = str(nib.load(join(images_dir, nii_files[0])).header.get_data_dtype())
                        total_uncompressed_f32 = 0.0
                        total_uncompressed_f16 = 0.0
                        for fname in nii_files:
                            n_voxels = int(np.prod(nib.load(join(images_dir, fname)).header.get_data_shape()))
                            total_uncompressed_f32 += n_voxels * 4 / 1e6
                            total_uncompressed_f16 += n_voxels * 2 / 1e6
                    except Exception as e:
                        self.print_to_log_file(f"_compute_buffer_benchmark: nibabel error: {e}")

                    nifti_stats = {
                        'n_cases':                         n_cases,
                        'n_modality_files':                len(nii_files),
                        'nifti_dtype':                     nifti_dtype_str,
                        'total_nifti_MB':                  round(total_nifti_MB, 3),
                        'total_uncompressed_nifti_MB_f32': round(total_uncompressed_f32, 3)
                                                           if total_uncompressed_f32 is not None else None,
                        'total_uncompressed_nifti_MB_f16': round(total_uncompressed_f16, 3)
                                                           if total_uncompressed_f16 is not None else None,
                    }

        report['nifti_stats'] = nifti_stats
        return report

    def _log_storage_summary(self, report: dict):
        """Log per-domain buffer size and running cumulative totals."""
        if not report:
            return

        total_MB = report.get('total_buffer_MB', 0.0)
        mag_MB   = report.get('magnitude_MB',    0.0)
        ph_MB    = report.get('phase_MB',        0.0)
        model_MB = report.get('model_size_MB',   None)
        nifti_MB = (report.get('nifti_stats') or {}).get('total_nifti_MB', None)

        parts = [
            f"domain={report.get('domain_idx', '?')}",
            f"buffer={total_MB:.3f} MB  (mag={mag_MB:.3f}  phase={ph_MB:.3f}  std+mean≈0)",
        ]
        if model_MB is not None:
            parts.append(f"model={model_MB:.1f} MB")
        if nifti_MB is not None and total_MB > 0:
            parts.append(f"NIfTI/buffer={nifti_MB / total_MB:.1f}×")

        self.print_to_log_file("CauAug Storage  —  " + "  |  ".join(parts))

        # ── cumulative ────────────────────────────────────────────────────
        cum_buffer = sum(r.get('total_buffer_MB', 0.0) for r in self.storage_benchmarks)
        cum_nifti  = sum((r.get('nifti_stats') or {}).get('total_nifti_MB', 0.0)
                         for r in self.storage_benchmarks)
        n_domains  = len(self.storage_benchmarks)
        model_line = f"model={self.storage_benchmarks[-1].get('model_size_MB', '?')} MB  |  " \
                     if self.storage_benchmarks else ""

        self.print_to_log_file(
            f"Cumulative ({n_domains} domain(s))  —  "
            f"{model_line}"
            f"Buffer_total={cum_buffer:.3f} MB  |  "
            f"NIfTI_gz={cum_nifti:.1f} MB  |  "
            f"NIfTI/buffer={cum_nifti / max(cum_buffer, 1e-9):.1f}×"
        )

    def _save_storage_plot(self):
        """Save per-domain buffer breakdown chart for CauAug domain knowledge archive."""
        if not self.storage_benchmarks:
            return

        n       = len(self.storage_benchmarks)
        x       = np.arange(n, dtype=float)
        xlabels = [
            t.replace("Task", "T").replace("_", "\n")
            for t in self.storage_task_names
        ]

        mag_arr   = np.array([r.get('magnitude_MB',    0.0) for r in self.storage_benchmarks])
        ph_arr    = np.array([r.get('phase_MB',        0.0) for r in self.storage_benchmarks])
        sm_arr    = np.array([r.get('std_MB', 0.0) + r.get('mean_MB', 0.0)
                              for r in self.storage_benchmarks])
        nifti_arr = np.array([(r.get('nifti_stats') or {}).get('total_nifti_MB', 0.0)
                               for r in self.storage_benchmarks])
        has_nifti = np.any(nifti_arr > 0)

        model_mb = self.storage_benchmarks[-1].get('model_size_MB', None)

        c_mag   = "#e07b54"  # warm orange  — magnitude
        c_ph    = "#c94e7a"  # pink-red     — phase
        c_sm    = "#a0c4a0"  # muted green  — std + mean (tiny)
        c_nifti = "#4878cf"  # blue         — NIfTI reference
        c_model = "#6abf69"  # green        — model weights line

        bar_w     = 0.30
        inner_gap = 0.06
        offsets   = np.array([-0.5, 0.5]) * (bar_w + inner_gap)

        fig, ax = plt.subplots(figsize=(max(8, n * 2.5 + 3), 5.5))
        fig.suptitle(
            "CauAug Domain Knowledge Archive: Buffer Size vs. NIfTI imagesTr",
            fontsize=11, fontweight="bold",
        )

        # ── stacked buffer bars ───────────────────────────────────────────
        for i in range(n):
            ax.bar(i + offsets[0], mag_arr[i], bar_w,
                   color=c_mag, alpha=0.88, edgecolor="white", linewidth=0.4,
                   label="Magnitude" if i == 0 else "_nolegend_")
            ax.bar(i + offsets[0], ph_arr[i], bar_w, bottom=mag_arr[i],
                   color=c_ph, alpha=0.88, edgecolor="white", linewidth=0.4,
                   label="Phase" if i == 0 else "_nolegend_")
            ax.bar(i + offsets[0], sm_arr[i], bar_w, bottom=mag_arr[i] + ph_arr[i],
                   color=c_sm, alpha=0.88, edgecolor="white", linewidth=0.4,
                   label="Std + Mean" if i == 0 else "_nolegend_")

            total = mag_arr[i] + ph_arr[i] + sm_arr[i]
            if total > 0:
                ax.text(i + offsets[0], total * 1.08, f"{total:.2f}",
                        ha="center", va="bottom", fontsize=6.5,
                        color="#333333", rotation=45)

        # ── NIfTI reference bars ──────────────────────────────────────────
        if has_nifti:
            for i, val in enumerate(nifti_arr):
                ax.bar(i + offsets[1], val, bar_w,
                       color=c_nifti, alpha=0.88, edgecolor="white", linewidth=0.4,
                       label="NIfTI compressed (.nii.gz)" if i == 0 else "_nolegend_")
                if val > 0:
                    ax.text(i + offsets[1], val * 1.08, f"{val:.1f}",
                            ha="center", va="bottom", fontsize=6.5,
                            color=c_nifti, rotation=45)

        # ── model size: single horizontal reference line ──────────────────
        if model_mb is not None:
            ax.axhline(model_mb, color=c_model, linewidth=1.5, linestyle="--",
                       label=f"Model weights (params only) = {model_mb:.1f} MB")
            ax.text(n - 0.5, model_mb * 1.06, f"{model_mb:.1f} MB",
                    color=c_model, fontsize=7.5, va="bottom", ha="right")

        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel("Domain (task)", fontsize=9)
        ax.set_ylabel("Size (MB, log scale)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9, handlelength=1.4)
        ax.grid(axis="y", which="both", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # group labels below x-axis
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        sub_labels = [(offsets[0], "Buffer\n(stacked)")]
        if has_nifti:
            sub_labels.append((offsets[1], "NIfTI\ngz"))
        for off, lbl in sub_labels:
            for xi in x:
                ax.text(xi + off, -0.16, lbl, transform=trans,
                        ha="center", va="top", fontsize=6,
                        color="#555555", style="italic")

        # ── summary annotation ────────────────────────────────────────────
        cum_buffer = float(np.sum(mag_arr + ph_arr + sm_arr))
        cum_nifti  = float(np.sum(nifti_arr))
        model_line = f"  Model weights (params only): {model_mb:.1f} MB\n" if model_mb else ""

        ann = (
            f"Cumulative across {n} domain(s)\n"
            f"{model_line}"
            f"  Buffer total:    {cum_buffer:.3f} MB\n"
            f"    magnitude:     {float(np.sum(mag_arr)):.3f} MB\n"
            f"    phase:         {float(np.sum(ph_arr)):.3f} MB\n"
            f"    std + mean:    {float(np.sum(sm_arr)):.4f} MB\n"
            + (f"  NIfTI gz total:  {cum_nifti:.1f} MB\n"
               f"  NIfTI / buffer:  {cum_nifti / max(cum_buffer, 1e-9):.1f}×"
               if has_nifti else "")
        )
        fig.text(
            0.99, 0.01, ann,
            ha="right", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                      edgecolor="#cccccc", alpha=0.92),
        )

        plt.tight_layout(rect=[0, 0.15, 1, 1])

        archive_path = join(self.trained_on_path, self.extension, DOMAIN_KNOWLEDGE_PATH)
        os.makedirs(archive_path, exist_ok=True)

        for ext in ("png", "pdf"):
            fig.savefig(join(archive_path, f"memory_benchmark.{ext}"),
                        dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.print_to_log_file(
            f"CauAug storage plot saved → {join(archive_path, 'memory_benchmark.png')}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def cross_kernel_network(self, inputs):
        """
        CKNet: Cross Kernel Network for anatomy causality.
        Blurs irrelevant local textures using random convolutions to force the model
        to focus on anatomical structures, reducing absorption bias.

        Args:
            inputs: Input tensor [B, C, H, W]

        Returns:
            Augmented tensor with blurred local textures
        """
        nb, nc, nx, ny = inputs.shape
        device = inputs.device

        original_mean = inputs.mean(dim=(2, 3), keepdim=True)
        original_std  = inputs.std(dim=(2, 3),  keepdim=True)

        w_1 = torch.rand(nc, nc, 3, 3, device=device)
        b_1 = torch.rand(nc, device=device)
        x_1 = F.conv2d(inputs, w_1, b_1, stride=1, padding=1)
        x_1 = F.leaky_relu(x_1)

        w_2 = torch.rand(nc, nc, 3, 3, device=device)
        b_2 = torch.rand(nc, device=device)
        x_2 = F.conv2d(x_1, w_2, b_2, stride=1, padding=1)

        current_mean = x_2.mean(dim=(2, 3), keepdim=True)
        current_std  = x_2.std(dim=(2, 3),  keepdim=True)
        x_2 = (x_2 - current_mean) / (current_std + 1e-8) * original_std + original_mean

        return x_2

    def fourier_transformer_generator(self, inputs):
        """
        FTGen: Fourier Transform Generator for domain causality.
        Reconstructs domain-specific features from previous domains using Fourier domain
        manipulation, reducing retention bias.

        Args:
            inputs: Input tensor [B, C, H, W]

        Returns:
            Augmented tensor with old domain characteristics
        """
        if len(self.magnitude_dataset) == 0:
            return inputs

        fft = torch.fft.fft2(inputs)

        index        = torch.randint(len(self.magnitude_dataset), []).item()
        magnitude_old = self.magnitude_dataset[index].to(inputs.device)
        std_old       = self.std_dataset[index].view(1, -1, 1, 1).to(inputs.device)
        mean_old      = self.mean_dataset[index].view(1, -1, 1, 1).to(inputs.device)

        batch_size, channels, height, width = inputs.shape

        if magnitude_old.shape[-2:] != (height, width):
            magnitude_old = magnitude_old.unsqueeze(0)
            magnitude_old = F.interpolate(
                magnitude_old, size=(height, width),
                mode='bilinear', align_corners=False,
            )
            magnitude_old = magnitude_old.squeeze(0)

        magnitude_old = magnitude_old.unsqueeze(0).expand(batch_size, -1, -1, -1)

        new_fft    = magnitude_old * torch.exp(1j * torch.angle(fft))
        new_inputs = torch.fft.ifft2(new_fft).real

        cur_std  = torch.std(new_inputs,  dim=(2, 3), keepdim=True)
        cur_mean = torch.mean(new_inputs, dim=(2, 3), keepdim=True)
        new_inputs = (new_inputs - cur_mean) / (cur_std + 1e-8) * std_old + mean_old

        return new_inputs

    def distillation_loss(self, pred, target):
        """
        Pixel-wise distillation loss for knowledge retention.

        Args:
            pred: Predicted output (softmax applied)
            target: Target from old model (softmax applied)

        Returns:
            Cross-entropy loss
        """
        return -(target * torch.log(pred.clamp(min=1e-8, max=1.0 - 1e-8))).sum(dim=1).mean()

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False,
                      detach=True, no_loss=False):
        """
        Modified training iteration with CauAug augmentations and deep supervision support.

        Implements:
        1. CKNet augmentation for new knowledge (anatomy causality)
        2. Standard and FTGen-based distillation for old knowledge (domain causality)
        3. Deep supervision for all loss components
        """
        data_dict    = next(data_generator)
        data         = data_dict['data']
        target       = data_dict['target']
        properties   = data_dict['properties']
        slice_indices = data_dict['slice_indices']

        data   = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data   = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                # ── new knowledge (CKNet) ─────────────────────────────────
                if self.use_cknet and random.random() < self.cknet_prob and do_backprop:
                    data_augmented = self.cross_kernel_network(data)
                else:
                    data_augmented = data

                output_new = self.network(data_augmented)
                del data_augmented

                if not no_loss:
                    loss_new = self.loss(output_new, target)

                    # ── old knowledge distillation ────────────────────────
                    loss_old = torch.tensor(0.0).cuda()

                    if hasattr(self, 'network_old') and self.network_old is not None:
                        ds_loss_weights = self.loss.weight_factors if hasattr(self.loss, 'weight_factors') \
                            else [1.0] * len(output_new)

                        with torch.no_grad():
                            output_old = self.network_old(data)

                        loss_kd = 0.0
                        for output_new_level, output_old_level, weight in zip(output_new, output_old, ds_loss_weights):
                            if weight == 0:
                                continue
                            output_new_soft = softmax_helper(output_new_level)
                            output_old_soft = softmax_helper(output_old_level)
                            loss_kd += weight * self.distillation_loss(output_new_soft, output_old_soft)

                        loss_old = loss_kd

                        # ── FTGen distillation ────────────────────────────
                        if self.use_ftgen and len(self.magnitude_dataset) > 0 and do_backprop:
                            data_domain   = self.fourier_transformer_generator(data)
                            output_domain = self.network(data_domain)

                            with torch.no_grad():
                                target_domain = self.network_old(data_domain)

                            loss_dom = 0.0
                            for output_dom_level, target_dom_level, weight in zip(
                                    output_domain, target_domain, ds_loss_weights):
                                if weight == 0:
                                    continue
                                output_domain_soft = softmax_helper(output_dom_level)
                                target_domain_soft = softmax_helper(target_dom_level)
                                loss_dom += weight * self.distillation_loss(output_domain_soft, target_domain_soft)

                            loss_old = loss_kd + loss_dom
                            del data_domain, output_domain, target_domain

                    loss = loss_new + self.lambda_d * loss_old

            if do_backprop:
                self.amp_grad_scaler.scale(loss).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()

        else:
            # ── new knowledge (CKNet) ─────────────────────────────────────
            if self.use_cknet and random.random() < self.cknet_prob and do_backprop:
                data_augmented = self.cross_kernel_network(data)
            else:
                data_augmented = data

            output_new = self.network(data_augmented)
            del data_augmented

            if not no_loss:
                loss_new = self.loss(output_new, target)

                loss_old = torch.tensor(0.0).cuda() if torch.cuda.is_available() else torch.tensor(0.0)

                if hasattr(self, 'network_old') and self.network_old is not None:
                    ds_loss_weights = self.loss.weight_factors if hasattr(self.loss, 'weight_factors') \
                        else [1.0] * len(output_new)

                    with torch.no_grad():
                        output_old = self.network_old(data)

                    loss_kd = 0.0
                    for output_new_level, output_old_level, weight in zip(output_new, output_old, ds_loss_weights):
                        if weight == 0:
                            continue
                        output_new_soft = softmax_helper(output_new_level)
                        output_old_soft = softmax_helper(output_old_level)
                        loss_kd += weight * self.distillation_loss(output_new_soft, output_old_soft)

                    loss_old = loss_kd

                    # ── FTGen distillation ────────────────────────────────
                    if self.use_ftgen and len(self.magnitude_dataset) > 0 and do_backprop:
                        data_domain   = self.fourier_transformer_generator(data)
                        output_domain = self.network(data_domain)

                        with torch.no_grad():
                            target_domain = self.network_old(data_domain)

                        loss_dom = 0.0
                        for output_dom_level, target_dom_level, weight in zip(
                                output_domain, target_domain, ds_loss_weights):
                            if weight == 0:
                                continue
                            output_domain_soft = softmax_helper(output_dom_level)
                            target_domain_soft = softmax_helper(target_dom_level)
                            loss_dom += weight * self.distillation_loss(output_domain_soft, target_domain_soft)

                        loss_old = loss_kd + loss_dom
                        del data_domain, output_domain, target_domain

                loss = loss_new + self.lambda_d * loss_old

            if do_backprop:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output_new, target, properties, slice_indices)

        del target, data

        if do_backprop:
            self.mh_network.update_after_iteration()

        if not no_loss:
            if detach:
                return loss.detach().cpu().numpy()
            else:
                return loss