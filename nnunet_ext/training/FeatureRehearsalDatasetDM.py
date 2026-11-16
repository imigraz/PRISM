from typing import Iterable, Union, Any
import math
import torch, os, random
from torch import Tensor
from torch.utils.data import Sampler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from nnunet.training.data_augmentation import downsampling
from nnunet_ext.training.FeatureRehearsalDataset import FeatureRehearsalTargetType
from nnunet_ext.training.feature_selector import FeatureSelector
from nnunet_ext.training import feature_io
from nnunet_ext.training.feature_io import FeatureStorage

from enum import Enum
from os.path import join, exists
from typing import Optional, Dict, List, Set, Tuple
import pickle
from tqdm import tqdm
import shutil
import logging
from datetime import datetime
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileTracker:
    def __init__(self, base_path: str):
        """
        Track and log file operations and issues.

        Args:
            base_path: Base path for log files
        """
        self.base_path = base_path
        self.missing_files: Set[str] = set()
        self.corrupted_files: Set[str] = set()
        self.log_dir = os.path.join(base_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Set up file logger
        self.log_file = os.path.join(
            self.log_dir,
            f"feature_rehearsal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setLevel(logging.INFO)
        logger.addHandler(self.file_handler)

    def log_missing_file(self, file_path: str):
        """Log missing file and add to tracking set."""
        self.missing_files.add(file_path)
        logger.error(f"Missing file: {file_path}")

    def log_corrupted_file(self, file_path: str, error: Exception):
        """Log corrupted file and add to tracking set."""
        self.corrupted_files.add(file_path)
        logger.error(f"Corrupted file {file_path}: {str(error)}")

    def save_report(self):
        """Save summary report of file issues."""
        report_path = os.path.join(self.log_dir, "file_issues_report.txt")
        with open(report_path, 'w') as f:
            f.write("=== Missing Files ===\n")
            for file in sorted(self.missing_files):
                f.write(f"{file}\n")
            f.write("\n=== Corrupted Files ===\n")
            for file in sorted(self.corrupted_files):
                f.write(f"{file}\n")
        logger.info(f"File issues report saved to {report_path}")


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class FeatureSelectionMethod(Enum):
    RANDOM = "random"


class DatasetBuffer:
    def __init__(self,
                 retention_ratio: float = 0.05,
                 batch_size: int = 32):
        self.retention_ratio = retention_ratio
        self.batch_size = batch_size
        self.valid_indices: List[int] = []
        self.processed_files: Set[str] = set()
        self.data_patches: List[str] = []  # Store the actual files for this dataset
        self.file_to_index: Dict[str, int] = {}  # Map files to their indices


class SparsityAnalyzer:
    def __init__(self):
        self.dataset_stats = {}

    def update_stats(self, feature_map, dataset_id):
        """Update running average statistics for a given feature map and dataset."""
        if dataset_id not in self.dataset_stats:
            self.dataset_stats[dataset_id] = {
                'avg_negative_ratio': 0.0,
                'avg_zero_ratio': 0.0,
                'num_samples': 0
            }

        stats = self.dataset_stats[dataset_id]

        # Compute current ratios
        total_elements = feature_map.size
        curr_negative_ratio = (feature_map < 0).sum().item() / total_elements
        curr_zero_ratio = (feature_map == 0).sum().item() / total_elements

        # Update running averages
        n = stats['num_samples']
        stats['avg_negative_ratio'] = (n * stats['avg_negative_ratio'] + curr_negative_ratio) / (n + 1)
        stats['avg_zero_ratio'] = (n * stats['avg_zero_ratio'] + curr_zero_ratio) / (n + 1)
        stats['num_samples'] += 1

    def get_summary(self):
        """Get summary statistics for all datasets."""
        summary = {}
        for dataset_id, stats in self.dataset_stats.items():
            summary[dataset_id] = {
                'sparsity_ratio': stats['avg_negative_ratio'] + stats['avg_zero_ratio'],
                'negative_ratio': stats['avg_negative_ratio'],
                'zero_ratio': stats['avg_zero_ratio'],
                'samples_analyzed': stats['num_samples']
            }
        return summary

    def reset_dataset(self, dataset_id):
        """Reset statistics for a specific dataset."""
        if dataset_id in self.dataset_stats:
            del self.dataset_stats[dataset_id]


class FeatureRehearsalDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 deep_supervision_scales: list[list[float]],
                 target_type: FeatureRehearsalTargetType,
                 num_features: int,
                 retention_ratio: float = 0.05,
                 selection_method: FeatureSelectionMethod = FeatureSelectionMethod.RANDOM,
                 batch_size: int = 32,
                 cleanup_unused: bool = True,
                 new_task_idx: Optional[int] = None,
                 old_dict_from_file_name_to_task_idx: Optional[dict] = None,
                 load_skips: bool = True,
                 constant_skips: Optional[np.ndarray] = None,
                 random_state: int = 42,
                 dedup_slices: bool = False,
                 feature_storage: Optional[FeatureStorage] = None) -> None:
        """feature_storage: None reads the original format (.npy files, all levels in feature_pkl/*.pkl);
        a FeatureStorage reads the Blosc2 format of nnunet_ext.training.feature_io. The trainer passes the same
        object to the feature extraction, so writer and reader agree. The Blosc2 format stores no ground-truth
        label patches (only whether a patch contains foreground), so it cannot replay with ground-truth targets."""
        super().__init__()
        if feature_storage is not None and not load_skips:
            raise ValueError("Blosc2 feature storage requires load_skips=True")
        if feature_storage is not None and target_type == FeatureRehearsalTargetType.GROUND_TRUTH:
            raise ValueError("Blosc2 feature storage keeps no ground truth; use FeatureRehearsalTargetType.DISTILLED_OUTPUT")
        if (feature_storage is not None and not feature_storage.predictions
                and target_type == FeatureRehearsalTargetType.DISTILLED_OUTPUT):
            raise ValueError("DISTILLED_OUTPUT targets need FeatureStorage(predictions=True)")
        self.feature_storage = feature_storage
        self.data_path = data_path
        self.deep_supervision_scales = deep_supervision_scales
        self.target_type = target_type
        self.num_features = num_features
        self.load_skips = load_skips
        self.constant_skips = constant_skips
        self.load_meta = False
        self.batch_size = batch_size
        self.cleanup_unused = cleanup_unused
        self.selection_method = selection_method
        self.retention_ratio = retention_ratio
        self.random_state = random_state
        self.dedup_slices = dedup_slices
        self.sparsity_analyzer = SparsityAnalyzer()

        self.new_task_idx = new_task_idx
        self.old_dict_from_file_name_to_task_idx = old_dict_from_file_name_to_task_idx
        self.task_idx_array = []

        # Initialize file tracker
        self.file_tracker = FileTracker(data_path)

        # Initialize feature selector
        self._init_feature_selector()

        # Store separate buffers for each dataset
        self.dataset_buffers: Dict[int, DatasetBuffer] = {}
        self.current_dataset_id = 0

        # Global mapping of files to their dataset and index
        self.global_file_mapping: Dict[str, Tuple[int, int]] = {}

    def _verify_file_exists(self, file_path: str) -> bool:
        """Verify file exists and log if missing."""
        if not exists(file_path):
            self.file_tracker.log_missing_file(file_path)
            return False
        return True

    def _init_feature_selector(self):
        """Initialize feature selector based on selected method."""
        self.feature_selector = FeatureSelector(random_state=self.random_state, dedup_slices=self.dedup_slices)

    @property
    def _patch_folder(self) -> str:
        """Folder whose file names list the stored patches: foreground/ (Blosc2) or gt/ (original format)."""
        return "foreground" if self.feature_storage is not None else "gt"

    def _patch_name(self, file_name: str) -> str:
        """Patch name '<case>_<x>_<y>_<z>' from a file name in the patch folder."""
        if self.feature_storage is not None:
            return feature_io.stem(file_name)
        return file_name[:-4]  # Remove .npy extension

    def _load_gt(self, file_name: str) -> np.ndarray:
        """Ground-truth patch for a file name in gt/ (original format only)."""
        if self.feature_storage is not None:
            raise ValueError("the Blosc2 feature storage keeps no ground-truth patches")
        return np.load(join(self.data_path, "gt", file_name))

    def _has_foreground(self, file_name: str) -> bool:
        """Whether the patch's ground truth contains foreground (any label > 0)."""
        if self.feature_storage is not None:
            return feature_io.load_foreground(self.data_path, self._patch_name(file_name))
        return feature_io.has_foreground(self._load_gt(file_name))

    def _get_all_related_files(self, patch_name):
        """Get all files related to a given patch."""
        if self.feature_storage is not None:
            return feature_io.patch_files(self.data_path, self._patch_name(patch_name), self.feature_storage.levels)

        files = []
        base_name = patch_name[:-4]  # Remove .npy extension

        # Feature files
        if not self.load_skips:
            for i in range(self.num_features):
                feature_file = join(self.data_path, "features", f"{base_name}_{i}.npy")
                if os.path.exists(feature_file):
                    files.append(feature_file)

        # Feature pickle files
        pkl_file = join(self.data_path, "feature_pkl", f"{base_name}.pkl")
        if os.path.exists(pkl_file):
            files.append(pkl_file)

        # GT files
        gt_file = join(self.data_path, "gt", patch_name)
        if os.path.exists(gt_file):
            files.append(gt_file)

        # Prediction files
        pred_file = join(self.data_path, "predictions", f"{base_name}_0.npy")
        if os.path.exists(pred_file):
            files.append(pred_file)

        return files

    def update_features(self, specific_files: Optional[List[str]] = None):
        """Update features and predictions with a new dataset."""

        try:
            data_patches = os.listdir(join(self.data_path, self._patch_folder))
            data_patches.sort()
            if not self.load_skips:
                if self.constant_skips is None:
                    assert len(data_patches) > 0
                    # Load the pickle file first
                    loaded_data = load_pickle(join(self.data_path, "feature_pkl", data_patches[0][:-4] + ".pkl"))
                    features = loaded_data[:-1]  # except bottleneck
                    self.constant_skips = []
                    for f in features:
                        zero_array = np.zeros_like(f)
                        self.constant_skips.append(zero_array)

            # Create new buffer for this dataset
            buffer = DatasetBuffer(
                retention_ratio=self.retention_ratio,
                batch_size=self.batch_size
            )

            # Get list of files to process
            if specific_files is None:
                current_files = os.listdir(join(self.data_path, self._patch_folder))
                current_files.sort()
                files_to_process = [f for f in current_files
                                    if f not in self.global_file_mapping]
            else:
                files_to_process = [f for f in specific_files
                                    if f not in self.global_file_mapping]

            if not files_to_process:
                logger.info("No new features to process.")
                return

            # Load and process features and predictions
            valid_indices, valid_filenames = self._load_and_preprocess_features(files_to_process)

            # Update files_to_process to only include valid files
            valid_files = [files_to_process[i] for i in valid_indices]

            if len(valid_files) < 30:
                logger.warning(f"Only {len(valid_files)} feature maps for this domain; all of them are kept.")

            # Calculate number of features to keep, at least 30
            num_features_to_keep = max(30, int(len(valid_files) * self.retention_ratio))

            # Create sample IDs for scores that match training records
            sample_ids = []
            for file_name in valid_filenames:
                base_name = self._patch_name(file_name)
                parts = base_name.split('_')
                case_id = '_'.join(parts[:-3])
                z_coord = parts[-1]
                sample_id = f"{case_id}_{z_coord}"
                sample_ids.append(sample_id)

            selected_indices, metrics = self.feature_selector.select_samples(
                n_select=num_features_to_keep,
                method=self.selection_method.value,
                sample_ids=sample_ids,
                all_files=valid_files
            )

            logger.info(f"Selected {len(selected_indices)} features")

            # Update buffer with selected files
            buffer.valid_indices = selected_indices
            buffer.data_patches = valid_files

            # Update file mappings
            for idx, file_name in enumerate(valid_files):
                buffer.file_to_index[file_name] = idx

            # Store the buffer
            self.dataset_buffers[self.current_dataset_id] = buffer

            # Update global file mapping for selected files
            for idx in selected_indices:
                file_name = valid_files[idx]
                self.global_file_mapping[file_name] = (self.current_dataset_id, idx)

            logger.info(
                f"Dataset {self.current_dataset_id}: Processed {len(valid_files)} files, "
                f"kept {len(selected_indices)} features"
            )

            # Clean up unused files only from the current dataset
            if self.cleanup_unused:
                self._cleanup_unused_files(valid_files, selected_indices)

            # Save current file issues report
            self.file_tracker.save_report()

            # Increment dataset counter
            self.current_dataset_id += 1

        except Exception as e:
            logger.error(f"Error updating features: {str(e)}")
            raise

    def _cleanup_unused_files(self, all_files: List[str], valid_indices: List[int]):
        """Clean up files that weren't selected during deduplication for the current dataset only."""
        valid_files = {all_files[i] for i in valid_indices}
        files_to_backup = set(all_files) - valid_files

        if not files_to_backup:
            return

        backup_dir = join(self.data_path, f"duplicate_backup_dataset_{self.current_dataset_id}")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        total_size_saved = 0

        for file_name in files_to_backup:
            files_to_move = self._get_all_related_files(file_name)

            for file_path in files_to_move:
                if os.path.exists(file_path):
                    rel_path = os.path.relpath(file_path, self.data_path)
                    backup_path = join(backup_dir, rel_path)

                    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                    total_size_saved += os.path.getsize(file_path)
                    shutil.move(file_path, backup_path)

        total_mb_saved = total_size_saved / (1024 * 1024)
        logger.info(f"Cleanup complete for dataset {self.current_dataset_id}. "
                    f"Moved {total_mb_saved:.2f} MB to {backup_dir}")

    def _load_and_preprocess_features(self, files_to_process: List[str], load_deepest_feature=False) -> Union[
        tuple[Tensor, list[Any], list[Any]], tuple[list[Any], list[Any]]]:
        """Load and preprocess features."""
        bottleneck_features = []
        valid_indices = []
        valid_filenames = []

        print("Loading features...")
        for idx, file_name in enumerate(tqdm(files_to_process)):
            if self.feature_storage is not None:
                # check that the deepest stored level can be read
                feature_path = feature_io.feature_path(self.data_path, self._patch_name(file_name),
                                                       self.feature_storage.levels - 1)
            else:
                feature_path = join(self.data_path, "feature_pkl", file_name[:-4] + ".pkl")
            if not self._verify_file_exists(feature_path):
                continue

            try:
                if self.feature_storage is not None:
                    bottleneck = feature_io.load_array(feature_path)
                else:
                    with open(feature_path, 'rb') as f:
                        features_dict = pickle.load(f)
                        bottleneck = features_dict[self.num_features - 1]
            except Exception as e:
                self.file_tracker.log_corrupted_file(feature_path, e)
                continue

            if load_deepest_feature:
                bottleneck_features.append(bottleneck)
            valid_indices.append(idx)
            valid_filenames.append(file_name)

        if load_deepest_feature:
            bottleneck_tensor = torch.from_numpy(np.stack(bottleneck_features)).float()
            orig_shape = bottleneck_tensor.shape
            bottleneck_flat = bottleneck_tensor.reshape(orig_shape[0], -1)
            return bottleneck_flat, valid_indices, valid_filenames

        return valid_indices, valid_filenames

    def benchmark_storage(
            self,
            dataset_id: int,
            max_samples: Optional[int] = None,
            task_name: Optional[str] = None,
            fold_cases: Optional[set] = None,
    ) -> Dict:
        """
        Measure theoretical storage costs for one dataset's selected features.

        Only level 0 (first encoder output / first skip) is actively replayed
        by feature_forward. All other levels are shown for reference only.
        Reported summary costs (log lines, plot) use level 0 only.

        Per-format costs:
          raw            – float32 dense
          coo            – float16 values + int16 COO indices (non-zeros only)
          gz_dense       – zlib-compressed float32 dense  (level 6)
          gz_sparse_f16  – zlib-compressed float16 values + int16 indices (level 6)
          gz_sparse_f32  – zlib-compressed float32 values + int16 indices (level 6)

        All sizes in MB. Returns a structured summary dict.

        Args:
            dataset_id:  Index into self.dataset_buffers.
            max_samples: Cap the number of patches sampled for per-format stats.
            task_name:   Task name used to locate imagesTr for NIfTI reference sizes.
            fold_cases:  Set of case IDs (without modality suffix) that were actually
                         used during training (i.e. dataset_tr.keys()).  When provided,
                         NIfTI reference sizes are restricted to these cases — the true
                         upper bound and consistent across trainers.  When None, case IDs
                         are inferred from stored patch filenames as a fallback.
        """
        import zlib

        buffer = self.dataset_buffers.get(dataset_id)
        if buffer is None:
            logger.warning(f"benchmark_storage: no buffer for dataset_id={dataset_id}")
            return {}

        selected = buffer.valid_indices
        files = buffer.data_patches
        sample_n = len(selected) if max_samples is None else min(max_samples, len(selected))
        rng = np.random.default_rng(self.random_state)
        sampled_indices = rng.choice(selected, size=sample_n, replace=False)

        rows = []

        for idx in sampled_indices:
            file_name = files[idx]
            if self.feature_storage is not None:
                patch = self._patch_name(file_name)
                pkl_path = feature_io.feature_path(self.data_path, patch, 0)
                if not exists(pkl_path):
                    continue
                try:
                    features: list = feature_io.load_feature_list(self.data_path, patch, self.feature_storage.levels)
                except Exception as e:
                    logger.warning(f"benchmark_storage: could not load {pkl_path}: {e}")
                    continue
            else:
                pkl_path = join(self.data_path, "feature_pkl", file_name[:-4] + ".pkl")
                if not exists(pkl_path):
                    continue
                try:
                    with open(pkl_path, "rb") as f:
                        features: list = pickle.load(f)
                except Exception as e:
                    logger.warning(f"benchmark_storage: could not load {pkl_path}: {e}")
                    continue

            for level, arr in enumerate(features):
                arr_f32 = arr.astype(np.float32)
                nz_mask = arr_f32 != 0.0
                nz_vals_f16 = arr_f32[nz_mask].astype(np.float16)
                nz_vals_f32 = arr_f32[nz_mask]
                nz_coords = np.array(np.nonzero(nz_mask), dtype=np.int16)

                rows.append({
                    "level": level,
                    "shape": arr_f32.shape,
                    "sparsity": float(1.0 - nz_mask.mean()),
                    "raw_MB": arr_f32.nbytes / 1e6,
                    "coo_MB": (nz_vals_f16.nbytes + nz_coords.nbytes) / 1e6,
                    "gz_dense_MB": len(zlib.compress(arr_f32.tobytes(), level=6)) / 1e6,
                    "gz_sparse_f16_MB": len(
                        zlib.compress(nz_vals_f16.tobytes() + nz_coords.tobytes(), level=6)
                    ) / 1e6,
                    "gz_sparse_f32_MB": len(
                        zlib.compress(nz_vals_f32.tobytes() + nz_coords.tobytes(), level=6)
                    ) / 1e6,
                })

        if not rows:
            logger.warning("benchmark_storage: no rows collected — check dataset_id / paths")
            return {}

        # Per-level averages
        by_level: Dict[int, list] = defaultdict(list)
        for r in rows:
            by_level[r["level"]].append(r)

        def avg(lvl_rows, key):
            return float(np.mean([r[key] for r in lvl_rows]))

        level_summaries = {}
        for level in sorted(by_level):
            lr = by_level[level]
            level_summaries[level] = {
                "shape": lr[0]["shape"],
                "sparsity_pct": round(avg(lr, "sparsity") * 100, 1),
                "raw_MB": round(avg(lr, "raw_MB"), 4),
                "coo_MB": round(avg(lr, "coo_MB"), 4),
                "gz_dense_MB": round(avg(lr, "gz_dense_MB"), 4),
                "gz_sparse_f16_MB": round(avg(lr, "gz_sparse_f16_MB"), 4),
                "gz_sparse_f32_MB": round(avg(lr, "gz_sparse_f32_MB"), 4),
            }

        # Actual on-disk sizes
        # Original format: feature_pkl/*.pkl. Blosc2 format: the features/*.b2nd files of the patch, and
        # separately the whole buffer on disk (features, foreground flag and, if stored, prediction).
        pkl_dir = join(self.data_path, "feature_pkl")
        actual_pkl_sizes = {}
        total_actual_pkl_MB = 0.0
        total_buffer_MB = 0.0
        for idx in selected:
            file_name = files[idx]
            if self.feature_storage is not None:
                patch = self._patch_name(file_name)
                feature_files = [feature_io.feature_path(self.data_path, patch, lv)
                                 for lv in range(self.feature_storage.levels)]
                if all(exists(f) for f in feature_files):
                    sz = sum(os.path.getsize(f) for f in feature_files) / 1e6
                    actual_pkl_sizes[file_name] = sz
                    total_actual_pkl_MB += sz
                total_buffer_MB += sum(os.path.getsize(f) for f in
                                       feature_io.patch_files(self.data_path, patch, self.feature_storage.levels)) / 1e6
                continue
            pkl_path = join(pkl_dir, file_name[:-4] + ".pkl")
            if exists(pkl_path):
                sz = os.path.getsize(pkl_path) / 1e6
                actual_pkl_sizes[file_name] = sz
                total_actual_pkl_MB += sz
        avg_actual_pkl_MB = total_actual_pkl_MB / max(len(actual_pkl_sizes), 1)

        # NIfTI reference sizes
        nifti_stats = None
        nifti_per_file = []
        if task_name is not None:
            raw_base = os.environ.get("nnUNet_raw_data_base", None)
            if raw_base is not None:
                images_dir = join(raw_base, "nnUNet_raw_data", task_name, "imagesTr")
                if os.path.isdir(images_dir):
                    all_nii = sorted(
                        f for f in os.listdir(images_dir) if f.endswith(".nii.gz")
                    )

                    if fold_cases is not None:
                        nii_files = sorted(
                            f for f in all_nii
                            if f.rsplit("_", 1)[0] in fold_cases
                        )
                        logger.info(
                            f"benchmark_storage: dataset_tr filter — "
                            f"{len(nii_files)}/{len(all_nii)} NIfTI files "
                            f"({len({f.rsplit('_', 1)[0] for f in nii_files})} cases)"
                        )
                    else:
                        training_case_ids: set = set()
                        for fname in files:
                            base = self._patch_name(fname)
                            parts = base.rsplit('_', 3)
                            if len(parts) == 4:
                                training_case_ids.add(parts[0])
                        nii_files = sorted(
                            f for f in all_nii
                            if f.rsplit("_", 1)[0] in training_case_ids
                        )
                        logger.info(
                            f"benchmark_storage: patch-derived filter (fold_cases not provided) — "
                            f"{len(nii_files)}/{len(all_nii)} NIfTI files"
                        )

                    case_ids = sorted({f.rsplit("_", 1)[0] for f in nii_files})

                    if nii_files:
                        total_nifti_MB = sum(
                            os.path.getsize(join(images_dir, f)) / 1e6
                            for f in nii_files
                        )
                        n_cases = len(case_ids)
                        total_slices = len(files)
                        avg_nifti_per_slice = total_nifti_MB / max(total_slices, 1)
                        avg_l0_gz_sparse = level_summaries[0]["gz_sparse_f16_MB"]

                        total_uncompressed_nifti_MB_f32 = None
                        total_uncompressed_nifti_MB_f16 = None
                        nifti_dtype_str = "unknown"
                        try:
                            import nibabel as nib
                            dtype = nib.load(join(images_dir, nii_files[0])).header.get_data_dtype()
                            nifti_dtype_str = str(dtype)
                            total_uncompressed_nifti_MB_f32 = 0.0
                            total_uncompressed_nifti_MB_f16 = 0.0
                            for fname in nii_files:
                                fpath = join(images_dir, fname)
                                hdr = nib.load(fpath).header
                                n_voxels = np.prod(hdr.get_data_shape())
                                uncomp_f32_mb = n_voxels * 4 / 1e6
                                uncomp_f16_mb = n_voxels * 2 / 1e6
                                total_uncompressed_nifti_MB_f32 += uncomp_f32_mb
                                total_uncompressed_nifti_MB_f16 += uncomp_f16_mb
                                nifti_per_file.append(
                                    (fname, os.path.getsize(fpath) / 1e6, uncomp_f32_mb)
                                )
                        except Exception as e:
                            logger.warning(
                                f"benchmark_storage: could not compute uncompressed NIfTI size: {e}"
                            )

                        nifti_stats = {
                            "n_cases": n_cases,
                            "n_modality_files": len(nii_files),
                            "nifti_dtype": nifti_dtype_str,
                            "total_nifti_MB": round(total_nifti_MB, 3),
                            "total_uncompressed_nifti_MB_f32": round(total_uncompressed_nifti_MB_f32, 3)
                            if total_uncompressed_nifti_MB_f32 is not None else None,
                            "total_uncompressed_nifti_MB_f16": round(total_uncompressed_nifti_MB_f16, 3)
                            if total_uncompressed_nifti_MB_f16 is not None else None,
                            "total_slices": total_slices,
                            "avg_nifti_MB_per_slice": round(avg_nifti_per_slice, 4),
                            "avg_nifti_vs_prism": round(
                                avg_nifti_per_slice / max(avg_l0_gz_sparse, 1e-9), 1
                            ),
                        }
                    else:
                        logger.warning(
                            f"benchmark_storage: no matching .nii.gz files found in {images_dir}"
                        )
                else:
                    logger.warning(
                        f"benchmark_storage: imagesTr not found at {images_dir}"
                    )

        n_selected = len(selected)
        n_total = len(files)
        col_w = 9

        def fmt_row(label, shape, sp, raw, coo, gd, gsf16, gsf32, marker="  "):
            return (
                f"{marker}{label:<6}  {str(shape):>22}  {sp:>7.1f}%  "
                f"{raw:>{col_w}.3f}  {coo:>{col_w}.3f}  "
                f"{gd:>{col_w}.3f}  {gsf16:>{col_w}.3f}  {gsf32:>{col_w}.3f}"
            )

        def fmt_proj(label, raw, coo, gd, gsf16, gsf32):
            return (
                f"    {label:<14}"
                f"  raw={raw:>8.1f} MB"
                f"  coo={coo:>8.1f} MB"
                f"  gz_dense={gd:>8.1f} MB"
                f"  gz_sparse_f16={gsf16:>8.1f} MB"
                f"  gz_sparse_f32={gsf32:>8.1f} MB"
            )

        SEP = "  " + "─" * 82

        lines = [
            "",
            "  " + "═" * 82,
            (
                f"  FEATURE STORAGE BENCHMARK  —  dataset_id={dataset_id}"
                f"  ({n_selected} selected / {n_total} total patches,"
                f" {sample_n} sampled for stats)"
            ),
            "  " + "═" * 82,
            f"  {'Level':>6}  {'Shape':>22}  {'Sparse%':>8}  "
            f"{'Raw':>{col_w}}  {'COO':>{col_w}}  {'GZ-dense':>{col_w}}"
            f"  {'GZ-sp-f16':>{col_w}}  {'GZ-sp-f32':>{col_w}}",
            f"  {'':->6}  {'':->22}  {'':->8}  "
            f"{'(MB)':>{col_w}}  {'(MB)':>{col_w}}  {'(MB)':>{col_w}}"
            f"  {'(MB)':>{col_w}}  {'(MB)':>{col_w}}",
        ]

        s0 = level_summaries[0]
        lines += [
            SEP,
            "  ▶  ACTIVE REPLAY  (level 0 only — used by feature_forward)",
            SEP,
            fmt_row("0", s0["shape"], s0["sparsity_pct"],
                    s0["raw_MB"], s0["coo_MB"], s0["gz_dense_MB"],
                    s0["gz_sparse_f16_MB"], s0["gz_sparse_f32_MB"],
                    marker="▶ "),
            "",
            f"  Projected cost × {n_selected} patches (level 0 — replay cost only):",
            fmt_proj("level 0",
                     s0["raw_MB"] * n_selected,
                     s0["coo_MB"] * n_selected,
                     s0["gz_dense_MB"] * n_selected,
                     s0["gz_sparse_f16_MB"] * n_selected,
                     s0["gz_sparse_f32_MB"] * n_selected),
            (
                f"  Compression ratios vs raw dense:"
                f"  gz_sparse_f16={s0['gz_sparse_f16_MB'] / max(s0['raw_MB'], 1e-9):.3f}x"
                f"  gz_sparse_f32={s0['gz_sparse_f32_MB'] / max(s0['raw_MB'], 1e-9):.3f}x"
            ),
        ]

        other_levels = [lv for lv in sorted(level_summaries) if lv != 0]
        if other_levels:
            lines += [
                SEP,
                "  ℹ  ALL OTHER LEVELS  (stored on disk, not used by feature_forward)",
                SEP,
            ]
            tot_raw = tot_coo = tot_gd = tot_gsf16 = tot_gsf32 = 0.0
            for lv in other_levels:
                s = level_summaries[lv]
                lines.append(fmt_row(str(lv), s["shape"], s["sparsity_pct"],
                                     s["raw_MB"], s["coo_MB"], s["gz_dense_MB"],
                                     s["gz_sparse_f16_MB"], s["gz_sparse_f32_MB"]))
                tot_raw += s["raw_MB"]
                tot_coo += s["coo_MB"]
                tot_gd += s["gz_dense_MB"]
                tot_gsf16 += s["gz_sparse_f16_MB"]
                tot_gsf32 += s["gz_sparse_f32_MB"]

            lines += [
                SEP,
                fmt_row("TOTAL", "(levels 1+)", 0.0,
                        tot_raw, tot_coo, tot_gd, tot_gsf16, tot_gsf32),
                "",
                f"  Projected cost × {n_selected} patches  (levels 1+ combined — not needed for replay):",
                fmt_proj("levels 1+",
                         tot_raw * n_selected, tot_coo * n_selected,
                         tot_gd * n_selected,
                         tot_gsf16 * n_selected, tot_gsf32 * n_selected),
            ]

        all_raw = sum(s["raw_MB"] for s in level_summaries.values())
        all_coo = sum(s["coo_MB"] for s in level_summaries.values())
        all_gd = sum(s["gz_dense_MB"] for s in level_summaries.values())
        all_gsf16 = sum(s["gz_sparse_f16_MB"] for s in level_summaries.values())
        all_gsf32 = sum(s["gz_sparse_f32_MB"] for s in level_summaries.values())

        lines += [
            SEP,
            "  GRAND TOTAL  (all levels per patch — current on-disk footprint)",
            fmt_row("ALL", "(all levels)", 0.0,
                    all_raw, all_coo, all_gd, all_gsf16, all_gsf32),
            "",
            f"  Projected cost × {n_selected} patches  (all levels combined):",
            fmt_proj("all levels",
                     all_raw * n_selected, all_coo * n_selected,
                     all_gd * n_selected,
                     all_gsf16 * n_selected, all_gsf32 * n_selected),
            "  " + "═" * 82,
            "",
        ]

        on_disk_label = "features/ (.b2nd)" if self.feature_storage is not None else "feature_pkl/"
        lines += [
            SEP,
            f"  ACTUAL ON-DISK  {on_disk_label}  ({len(actual_pkl_sizes)} files found of {n_selected} selected)",
            SEP,
        ]
        for file_name in sorted(actual_pkl_sizes):
            lines.append(f"    {file_name:<60s}  {actual_pkl_sizes[file_name]:>8.3f} MB")
        lines += [
            f"    {'─' * 60}  {'─' * 8}",
            f"    {'TOTAL':<60s}  {total_actual_pkl_MB:>8.3f} MB",
            f"    {'AVG per patch':<60s}  {avg_actual_pkl_MB:>8.3f} MB",
            "",
        ]
        if self.feature_storage is not None:
            lines += [
                f"    {'TOTAL buffer on disk (.b2nd)':<60s}  {total_buffer_MB:>8.3f} MB",
                "",
            ]

        if nifti_stats is not None:
            ns = nifti_stats
            filter_note = " [dataset_tr]" if fold_cases is not None else " [patch-derived]"
            lines += [
                SEP,
                f"  Real .nii.gz sizes  ({ns['n_cases']} cases,"
                f" {ns['n_modality_files']} files,"
                f" {ns['total_slices']} slices, dtype={ns['nifti_dtype']}){filter_note}:",
                SEP,
                f"    {'Filename':<50s}  {'gz (MB)':>9}  {'uncomp f32 (MB)':>16}",
                f"    {'─' * 50}  {'─' * 9}  {'─' * 16}",
            ]
            for fname, gz_mb, uncomp_mb in sorted(nifti_per_file):
                lines.append(f"    {fname:<50s}  {gz_mb:>9.3f}  {uncomp_mb:>16.3f}")
            lines += [
                f"    {'─' * 50}  {'─' * 9}  {'─' * 16}",
                f"    {'TOTAL':<50s}  {ns['total_nifti_MB']:>9.3f}  "
                f"{ns['total_uncompressed_nifti_MB_f32']:>16.3f}"
                if ns['total_uncompressed_nifti_MB_f32'] is not None
                else f"    {'TOTAL':<50s}  {ns['total_nifti_MB']:>9.3f}  {'n/a':>16}",
                "",
                f"  {'  NIfTI uncompressed float32':<46}  "
                f"{ns['total_uncompressed_nifti_MB_f32']:>9.3f} MB"
                if ns['total_uncompressed_nifti_MB_f32'] is not None
                else f"  {'  NIfTI uncompressed float32':<46}  {'n/a':>9}",
                f"  {'  NIfTI uncompressed float16':<46}  "
                f"{ns['total_uncompressed_nifti_MB_f16']:>9.3f} MB"
                if ns['total_uncompressed_nifti_MB_f16'] is not None
                else f"  {'  NIfTI uncompressed float16':<46}  {'n/a':>9}",
                f"  {'  NIfTI compressed (.nii.gz)':<46}  "
                f"{ns['total_nifti_MB']:>9.3f} MB",
                f"  {'  NIfTI per slice (compressed)':<46}  "
                f"{ns['avg_nifti_MB_per_slice']:>9.4f}  "
                f"{ns['avg_nifti_vs_prism']:>13.1f}×",
                f"  PRISM gz_sparse_f16 (level 0) is {ns['avg_nifti_vs_prism']:.1f}× smaller per slice "
                f"than already-compressed .nii.gz",
            ]

        for line in lines:
            logger.info(line)
            print(line)

        return {
            "nifti_stats": nifti_stats,
            "level_0_active": level_summaries[0],
            "other_levels": {lv: level_summaries[lv] for lv in other_levels},
            "n_selected": n_selected,
            "n_total": n_total,
            "n_sampled": sample_n,
            "actual_pkl_total_MB": round(total_actual_pkl_MB, 3),
            "actual_pkl_avg_MB": round(avg_actual_pkl_MB, 4),
            "storage_format": "blosc2" if self.feature_storage is not None else "pkl",
            "actual_buffer_total_MB": round(total_buffer_MB, 3) if self.feature_storage is not None else None,
            "projected": {
                "level_0": {
                    "raw_MB": round(s0["raw_MB"] * n_selected, 2),
                    "coo_MB": round(s0["coo_MB"] * n_selected, 2),
                    "gz_dense_MB": round(s0["gz_dense_MB"] * n_selected, 2),
                    "gz_sparse_f16_MB": round(s0["gz_sparse_f16_MB"] * n_selected, 2),
                    "gz_sparse_f32_MB": round(s0["gz_sparse_f32_MB"] * n_selected, 2),
                },
                "all_levels": {
                    "raw_MB": round(all_raw * n_selected, 2),
                    "coo_MB": round(all_coo * n_selected, 2),
                    "gz_dense_MB": round(all_gd * n_selected, 2),
                    "gz_sparse_f16_MB": round(all_gsf16 * n_selected, 2),
                    "gz_sparse_f32_MB": round(all_gsf32 * n_selected, 2),
                },
            },
        }

    def __len__(self):
        return sum(len(buffer.valid_indices) for buffer in self.dataset_buffers.values())

    def __getitem__(self, index):
        current_count = 0
        for dataset_id, buffer in self.dataset_buffers.items():
            buffer_size = len(buffer.valid_indices)
            if current_count + buffer_size > index:
                local_index = buffer.valid_indices[index - current_count]
                file_name = buffer.data_patches[local_index]
                data_dict = self._load_item(file_name, dataset_id)
                data_dict['dataset_id'] = dataset_id
                return data_dict
            current_count += buffer_size
        raise IndexError("Index out of range")

    def _load_item(self, file_name: str, dataset_id: int):
        """Load a single item with proper error handling."""
        try:
            data_dict = {}

            if self.feature_storage is not None:
                patch = self._patch_name(file_name)
                for level in range(self.feature_storage.levels):
                    path = feature_io.feature_path(self.data_path, patch, level)
                    if not self._verify_file_exists(path):
                        raise FileNotFoundError(f"Missing feature file: {path}")
                data_dict['features_and_skips'] = feature_io.load_feature_list(self.data_path, patch,
                                                                               self.feature_storage.levels)
            elif self.load_skips:
                pkl_path = join(self.data_path, "feature_pkl", file_name[:-4] + ".pkl")
                if not self._verify_file_exists(pkl_path):
                    raise FileNotFoundError(f"Missing feature file: {pkl_path}")
                data_dict['features_and_skips'] = load_pickle(pkl_path)
            else:
                feature_path = join(
                    self.data_path,
                    "features",
                    file_name[:-4] + f"_{self.num_features - 1}.npy"
                )
                if not self._verify_file_exists(feature_path):
                    raise FileNotFoundError(f"Missing feature file: {feature_path}")
                data_dict['features_and_skips'] = self.constant_skips + [np.load(feature_path)]

            first_feature_map = data_dict['features_and_skips'][0]
            self.sparsity_analyzer.update_stats(first_feature_map, dataset_id)

            if self.target_type == FeatureRehearsalTargetType.GROUND_TRUTH:
                gt_path = join(self.data_path, "gt", file_name)
                if not self._verify_file_exists(gt_path):
                    raise FileNotFoundError(f"Missing ground truth file: {gt_path}")
                gt_patch = self._load_gt(file_name)
                data_dict['target'] = gt_patch[None, None]
            elif self.target_type == FeatureRehearsalTargetType.DISTILLED_OUTPUT:
                if self.feature_storage is not None:
                    pred_path = feature_io.prediction_path(self.data_path, self._patch_name(file_name))
                else:
                    pred_path = join(self.data_path, "predictions", file_name[:-4] + "_0.npy")
                if not self._verify_file_exists(pred_path):
                    raise FileNotFoundError(f"Missing prediction file: {pred_path}")
                gt_patch = feature_io.load_array(pred_path) if self.feature_storage is not None else np.load(pred_path)
                data_dict['target'] = gt_patch[None, None]

            return data_dict

        except Exception as e:
            logger.error(f"Error loading file {file_name} from dataset {dataset_id}: {str(e)}")
            raise


class FeatureRehearsalDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None,
                 num_workers: int = 0, pin_memory: bool = False, drop_last: bool = False, timeout: float = 0,
                 worker_init_fn=None, multiprocessing_context=None, generator=None, *,
                 prefetch_factor=None, persistent_workers: bool = False, pin_memory_device: str = "",
                 deep_supervision_scales=None,
                 deterministic=False):
        self.deep_supervision_scales = deep_supervision_scales

        assert len(dataset) >= batch_size, f"{len(dataset)}, {batch_size}"

        def my_collate_function(list_of_samples: list[dict]):
            B = len(list_of_samples)
            output_batch = dict()

            if dataset.target_type in [FeatureRehearsalTargetType.GROUND_TRUTH,
                                       FeatureRehearsalTargetType.DISTILLED_OUTPUT]:
                targets = []
                for b in range(B):
                    targets.append(list_of_samples[b]['target'])
                targets = np.vstack(targets)
                output_batch['target'] = downsampling.downsample_seg_for_ds_transform2(targets,
                                                                                       self.deep_supervision_scales)
            elif dataset.target_type in [FeatureRehearsalTargetType.DISTILLED_DEEP_SUPERVISION]:
                assert False, "not implemented"
            elif dataset.target_type in [FeatureRehearsalTargetType.NONE]:
                pass
            else:
                assert False

            features_and_skips = []
            for res in range(len(list_of_samples[0]['features_and_skips'])):
                l = []
                for b in range(B):
                    l.append(torch.from_numpy(list_of_samples[b]['features_and_skips'][res]))
                features_and_skips.append(torch.vstack(l))
            output_batch['data'] = features_and_skips

            output_batch['dataset_id'] = torch.IntTensor([sample['dataset_id'] for sample in list_of_samples])

            if hasattr(dataset, 'load_meta') and dataset.load_meta:
                output_batch['slice_idx_normalized'] = torch.FloatTensor(
                    [sample['slice_idx_normalized'] for sample in list_of_samples])

            return output_batch

        def seed_worker(worker_id):
            if deterministic:
                worker_seed = torch.initial_seed() % 2 ** 32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            if worker_init_fn != None:
                worker_init_fn(worker_id)

        super().__init__(dataset, batch_size, shuffle,
                         sampler,
                         batch_sampler, num_workers, my_collate_function, pin_memory, drop_last, timeout, seed_worker,
                         multiprocessing_context, generator, prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)


class BalancedBatchSampler(Sampler):
    """Replay sampler: domain-balanced batches with foreground oversampling.

    Every batch of ``batch_size`` samples is split as evenly as possible over the domains in the buffer (the
    domains that get one sample more are drawn at random), so each previous domain is replayed with the same
    probability regardless of its buffer size. ceil(batch_size / 3) of the batch's slots, chosen at random, are
    foreground slots: they are filled with patches whose ground truth contained foreground (nnU-Net's foreground
    oversampling; only the stored foreground flag is used). The other slots are drawn uniformly from the domain.
    A domain without foreground patches fills its foreground slots uniformly as well. Within a batch, samples are
    drawn without replacement as long as the domain has enough patches.

    Indices refer to ``dataset`` (a FeatureRehearsalDataset or a FeatureRehearsalMultiDataset around one, whose
    copies are only used to set the number of batches per pass): domain d occupies the consecutive indices
    after the buffers of domains 0 .. d-1, in the order of ``valid_indices``.
    """

    FOREGROUND_FRACTION = 1 / 3

    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        self.num_copies = dataset.num_copies if hasattr(dataset, 'num_copies') else 1

        # Index range of every domain in the dataset
        self.domain_ranges: Dict[int, range] = {}
        start = 0
        for dataset_id, buffer in self.base_dataset.dataset_buffers.items():
            self.domain_ranges[dataset_id] = range(start, start + len(buffer.valid_indices))
            start += len(buffer.valid_indices)
        self.domain_ids = [d for d, r in self.domain_ranges.items() if len(r) > 0]
        assert len(self.domain_ids) > 0, "the replay buffer is empty"

        # One pass over the (virtually copied) dataset, as before
        self.total_batches = max(1, math.ceil(len(dataset) / batch_size))

        self.foreground_indices = self._get_foreground_indices()
        for dataset_id in self.domain_ids:
            if not self.foreground_indices.get(dataset_id):
                logger.warning(f"Domain {dataset_id} has no foreground patch in the replay buffer; "
                               "its share of each batch is sampled uniformly.")

    def _get_foreground_indices(self) -> Dict[int, List[int]]:
        """Dataset indices of the patches with foreground, per domain."""
        foreground_indices = defaultdict(list)
        print("Analyzing patches for foreground content...")
        for dataset_id, buffer in self.base_dataset.dataset_buffers.items():
            first = self.domain_ranges[dataset_id].start
            for position, local_index in enumerate(buffer.valid_indices):
                if self.base_dataset._has_foreground(buffer.data_patches[local_index]):
                    foreground_indices[dataset_id].append(first + position)
        return dict(foreground_indices)

    def _domain_counts(self) -> Dict[int, int]:
        """Samples per domain for one batch: as equal as possible, random domains get the remainder."""
        base, extra = divmod(self.batch_size, len(self.domain_ids))
        counts = {d: base for d in self.domain_ids}
        for d in random.sample(self.domain_ids, extra):
            counts[d] += 1
        return counts

    @staticmethod
    def _draw(pool: List[int], k: int) -> List[int]:
        """k items from pool, without replacement while possible."""
        if k <= 0 or not pool:
            return []
        if k <= len(pool):
            return random.sample(pool, k)
        return random.sample(pool, len(pool)) + random.choices(pool, k=k - len(pool))

    def _foreground_counts(self, counts: Dict[int, int]) -> Dict[int, int]:
        """Foreground slots per domain: ceil(batch_size / 3) of the batch's slots, chosen at random."""
        slots = [d for d, k in counts.items() for _ in range(k)]
        n_foreground = math.ceil(len(slots) * self.FOREGROUND_FRACTION)
        foreground_counts = {d: 0 for d in counts}
        for d in random.sample(slots, n_foreground):
            foreground_counts[d] += 1
        return foreground_counts

    def _domain_indices(self, dataset_id: int, k: int, n_foreground: int) -> List[int]:
        all_indices = list(self.domain_ranges[dataset_id])
        foreground = self.foreground_indices.get(dataset_id, [])
        if not foreground:
            n_foreground = 0
        chosen = self._draw(foreground, n_foreground)
        chosen_set = set(chosen)
        rest = [i for i in all_indices if i not in chosen_set] or all_indices
        return chosen + self._draw(rest, k - n_foreground)

    def __iter__(self):
        for _ in range(self.total_batches):
            batch = []
            counts = self._domain_counts()
            foreground_counts = self._foreground_counts(counts)
            for dataset_id, k in counts.items():
                batch.extend(self._domain_indices(dataset_id, k, foreground_counts[dataset_id]))
            random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.total_batches * self.batch_size


class BalancedFeatureRehearsalDataLoader(FeatureRehearsalDataLoader):
    """DataLoader that ensures balanced sampling across different datasets."""

    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                 multiprocessing_context=None, generator=None, *, prefetch_factor=None,
                 persistent_workers=False, pin_memory_device="", deep_supervision_scales=None,
                 deterministic=False):
        base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset

        dataset_sizes = {}
        for dataset_id, buffer in base_dataset.dataset_buffers.items():
            dataset_sizes[dataset_id] = len(buffer.valid_indices)

        sampler = BalancedBatchSampler(dataset, batch_size, drop_last)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            deep_supervision_scales=deep_supervision_scales,
            deterministic=deterministic
        )


class InfiniteIterator():
    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
        self.dataiter = iter(self.dataloader)

    def __next__(self):
        try:
            return next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            return next(self.dataiter)


class FeatureRehearsalConcatDataset(ConcatDataset):
    def __init__(self, main_dataset, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)
        self.dataset = main_dataset
        self.target_type = main_dataset.target_type
        self.load_meta = main_dataset.load_meta


class FeatureRehearsalMultiDataset(Dataset):
    # emulates multiple copies of the given dataset
    def __init__(self, dataset: FeatureRehearsalDataset, num_copies: float = 2) -> None:
        self.dataset = dataset
        self.target_type = dataset.target_type
        self.num_copies = num_copies
        self.load_meta = dataset.load_meta

    def __len__(self):
        return int(len(self.dataset) * self.num_copies)

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]