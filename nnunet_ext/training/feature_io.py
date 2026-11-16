"""Blosc2 storage of replay-buffer patches (used by PRISM).

Every array is stored losslessly as a Blosc2 NDArray (``.b2nd``, zstd with byte shuffle); dtype and shape are
preserved and the values read back are bit-identical to the values written.

Layout of a feature folder in this format (``<patch>`` is ``<case>_<x>_<y>_<z>``)::

    foreground/<patch>.b2nd          one bool: whether the ground-truth label patch contains foreground (used for
                                     foreground oversampling of the replay batches); the label patch is not stored
    predictions/<patch>_0.b2nd       predicted segmentation of the patch; only with FeatureStorage.predictions
                                     (replay with a segmentation loss on stored targets instead of distillation)
    features/<patch>_<level>.b2nd    sparsified feature maps, levels 0 .. FeatureStorage.levels - 1

foreground/ also serves as the list of the stored patches. Trainers that do not pass a FeatureStorage keep the
original format (``.npy`` files including the label patch in ``gt/``, and a ``feature_pkl/`` pickle with all
levels).
"""
import os
from dataclasses import dataclass
from os.path import join
from typing import List

import blosc2
import numpy as np

SUFFIX = ".b2nd"
FOLDERS = ("foreground", "features", "predictions")   # possible sub-folders of a feature folder; the first lists the patches
CPARAMS = {'codec': blosc2.Codec.ZSTD, 'clevel': 5, 'filters': [blosc2.Filter.SHUFFLE]}


@dataclass(frozen=True)
class FeatureStorage:
    """What the feature extraction writes in Blosc2 format.

    levels: number of feature levels stored per patch, counted from the first (level 0 = output of the first
        encoder block, the only level that PRISM replays).
    predictions: also store the model's prediction of each patch as a replay target. Not needed for replay with
        knowledge distillation, where the previous-domain model provides the targets.
    """
    levels: int = 1
    predictions: bool = True

    def folders(self) -> tuple:
        """Sub-folders that the extraction writes into."""
        return FOLDERS if self.predictions else tuple(f for f in FOLDERS if f != "predictions")

    def __post_init__(self):
        if self.levels < 1:
            raise ValueError("levels must be >= 1, got %r" % self.levels)


def save_array(path: str, array: np.ndarray) -> None:
    """Write ``array`` losslessly to ``path`` (which must end in .b2nd)."""
    if not path.endswith(SUFFIX):
        raise ValueError("Blosc2 array files must end in %s: %s" % (SUFFIX, path))
    blosc2.asarray(np.ascontiguousarray(array), urlpath=path, mode='w', cparams=CPARAMS)


def load_array(path: str) -> np.ndarray:
    """Read an array written by save_array, as a numpy array with its original dtype and shape."""
    return blosc2.open(path, mode='r')[:]


def stem(file_name: str) -> str:
    """Patch name without the .b2nd suffix."""
    if not file_name.endswith(SUFFIX):
        raise ValueError("not a %s file: %s" % (SUFFIX, file_name))
    return file_name[:-len(SUFFIX)]


def foreground_path(data_path: str, patch: str) -> str:
    return join(data_path, "foreground", patch + SUFFIX)


def has_foreground(ground_truth_patch: np.ndarray) -> bool:
    """The criterion of the replay sampler's foreground oversampling: any label > 0."""
    return bool(np.any(ground_truth_patch > 0))


def load_foreground(data_path: str, patch: str) -> bool:
    return bool(load_array(foreground_path(data_path, patch))[0])


def prediction_path(data_path: str, patch: str) -> str:
    return join(data_path, "predictions", patch + "_0" + SUFFIX)


def feature_path(data_path: str, patch: str, level: int) -> str:
    return join(data_path, "features", "%s_%d%s" % (patch, level, SUFFIX))


def save_feature_list(data_path: str, patch: str, features: List[np.ndarray]) -> None:
    """Store feature maps as features/<patch>_<level>.b2nd, one file per level."""
    for level, f in enumerate(features):
        save_array(feature_path(data_path, patch, level), f)


def load_feature_list(data_path: str, patch: str, levels: int) -> List[np.ndarray]:
    return [load_array(feature_path(data_path, patch, level)) for level in range(levels)]


def write_patch(data_path: str, patch: str, storage: FeatureStorage, ground_truth_patch: np.ndarray,
                features_and_skips: list, predicted_segmentations: list) -> None:
    """Write one extracted patch: its foreground flag, the first ``storage.levels`` feature maps and, if
    ``storage.predictions``, the first prediction. The ground-truth patch is only used to compute the foreground
    flag and is not stored.

    features_and_skips and predicted_segmentations are lists of torch tensors (as produced by the network).
    """
    if len(features_and_skips) < storage.levels:
        raise ValueError("%d feature levels requested, the network returned %d"
                         % (storage.levels, len(features_and_skips)))
    if len(predicted_segmentations) != 1:
        raise ValueError("expected one predicted segmentation (deep supervision off), got %d"
                         % len(predicted_segmentations))
    save_array(foreground_path(data_path, patch), np.array([has_foreground(ground_truth_patch)]))
    if storage.predictions:
        save_array(prediction_path(data_path, patch), predicted_segmentations[0].cpu().numpy())
    save_feature_list(data_path, patch, [f.cpu().numpy() for f in features_and_skips[:storage.levels]])


def patch_files(data_path: str, patch: str, levels: int) -> List[str]:
    """All files that belong to one patch (existing ones only)."""
    files = [feature_path(data_path, patch, level) for level in range(levels)]
    files += [foreground_path(data_path, patch), prediction_path(data_path, patch)]
    return [f for f in files if os.path.exists(f)]
