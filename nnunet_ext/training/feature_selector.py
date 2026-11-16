import logging
from collections import defaultdict
from typing import Tuple, Optional, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def patient_id(sample_id: str) -> str:
    """Patient (case) of a sample id '<case>_<z>': everything before the slice index."""
    return sample_id.rsplit('_', 1)[0]


class FeatureSelector:
    """Selects which extracted feature maps are kept in the replay buffer.

    Only patient-stratified random selection is used: every patient of a domain
    contributes about the same number of feature maps. With dedup_slices=True, at most
    one patch per image slice is kept (one patch is drawn at random from each slice first).
    """

    def __init__(self, random_state: int = 42, dedup_slices: bool = False):
        self.random_state = random_state
        self.dedup_slices = dedup_slices

    def select_samples(self,
                       n_select: int,
                       method: str,
                       sample_ids: Optional[List[str]] = None,
                       all_files: Optional[List[str]] = None
                       ) -> Tuple[np.ndarray, Dict]:
        """Select n_select samples; method must be 'random'."""
        if method == 'random':
            return self._select_random_samples(n_select, sample_ids, all_files)
        raise ValueError(f"Unknown selection method: {method}")

    def _select_random_samples(self, n_select: int,
                               sample_ids: Optional[List[str]] = None,
                               all_files: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Select random samples with equal distribution across patients.

        Args:
            n_select: Number of samples to select
            sample_ids: List of sample IDs '<case>_<z>' (case id and slice index)
            all_files: Patch file names ('<case>_<x>_<y>_<z>' plus a 4-character suffix), parallel to
                sample_ids; used only with dedup_slices.

        Returns:
            Tuple of (selected indices array, metrics dictionary)
        """
        if self.dedup_slices:
            if all_files is None:
                raise ValueError("dedup_slices needs the patch file names (all_files)")
            return self._select_random_samples_dedup_slices(n_select, sample_ids, all_files)
        return self._select_random_samples_stratified(n_select, sample_ids)

    def _select_random_samples_stratified(self, n_select: int,
                                          sample_ids: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
        """Patient-stratified random selection (the selection used for the paper's main results)."""
        n_samples = len(sample_ids)

        if n_samples <= n_select:
            # Fall back to selecting all samples
            rng = np.random.RandomState(self.random_state)
            selected_indices = np.sort(rng.choice(n_samples, size=min(n_select, n_samples), replace=False))
            return selected_indices, {
                'method': 'random',
                'selected_ratio': len(selected_indices) / n_samples
            }

        # Patient of each sample; sorted so that the selection depends only on random_state
        patient_ids = [patient_id(sample_id) for sample_id in sample_ids]
        unique_patients = sorted(set(patient_ids))
        n_patients = len(unique_patients)

        # Calculate samples per patient
        base_samples_per_patient = n_select // n_patients
        extra_samples = n_select % n_patients

        rng = np.random.RandomState(self.random_state)
        selected_indices = []
        samples_per_patient = {}

        # Distribute extra samples randomly among patients
        if extra_samples > 0:
            extra_sample_patients = rng.choice(unique_patients, size=extra_samples, replace=False)
            samples_per_patient = {patient: base_samples_per_patient + (1 if patient in extra_sample_patients else 0)
                                   for patient in unique_patients}
        else:
            samples_per_patient = {patient: base_samples_per_patient for patient in unique_patients}

        # Select samples for each patient
        for patient in unique_patients:
            # Get indices for this patient
            patient_mask = [pid == patient for pid in patient_ids]
            patient_indices = np.where(patient_mask)[0]

            # Handle case where patient has fewer samples than allocated
            n_select_patient = min(samples_per_patient[patient], len(patient_indices))

            # Randomly select samples for this patient
            if len(patient_indices) > 0:
                selected_patient_indices = rng.choice(
                    patient_indices,
                    size=n_select_patient,
                    replace=False
                )
                selected_indices.extend(selected_patient_indices)

        # Patients with fewer samples than their share make the buffer smaller than n_select
        if len(selected_indices) < n_select:
            logger.warning(f"Patient-stratified selection: {n_select} patches requested, "
                           f"{len(selected_indices)} selected.")

        metrics = {
            'method': 'stratified_random',
            'selected_ratio': len(selected_indices) / n_samples,
            'n_patients': n_patients,
            'target_samples_per_patient': samples_per_patient,
            'actual_samples_per_patient': {
                patient: sum(1 for idx in selected_indices if patient_ids[idx] == patient)
                for patient in unique_patients
            }
        }

        return np.sort(np.array(selected_indices)), metrics

    def _select_random_samples_dedup_slices(self, n_select: int, sample_ids: List[str],
                                            all_files: List[str]) -> Tuple[np.ndarray, Dict]:
        """Patient-stratified random selection with at most one patch per (case, slice).

        For each patient, the patches are grouped by case and slice index (z), one patch is drawn per group,
        and the patient's share is drawn from these. If a patient has fewer slices than its share, the buffer
        ends up smaller than n_select (a warning is logged).
        """
        n_samples = len(sample_ids)
        rng = np.random.RandomState(self.random_state)

        if n_samples <= n_select:
            selected_indices = np.sort(rng.choice(n_samples, size=min(n_select, n_samples), replace=False))
            return selected_indices, {'method': 'random', 'selected_ratio': len(selected_indices) / n_samples}

        patient_ids = [patient_id(sid) for sid in sample_ids]
        unique_patients = sorted(set(patient_ids))
        n_patients = len(unique_patients)

        base_samples_per_patient = n_select // n_patients
        extra_samples = n_select % n_patients
        extra_sample_patients = set(
            rng.choice(unique_patients, size=extra_samples, replace=False)) if extra_samples > 0 else set()
        samples_per_patient = {
            p: base_samples_per_patient + (1 if p in extra_sample_patients else 0)
            for p in unique_patients
        }

        selected_indices = []
        shortfall = 0
        for patient in unique_patients:
            patient_indices = np.where([pid == patient for pid in patient_ids])[0]

            if all_files is not None:
                slice_buckets: Dict[str, List[int]] = defaultdict(list)

                parsed = {
                    idx: all_files[idx][:-4].rsplit("_", 3)
                    for idx in patient_indices
                }

                for idx in patient_indices:
                    parts = parsed[idx]
                    if len(parts) == 4:
                        slice_key = f"{parts[0]}_{parts[3]}"
                    else:
                        slice_key = all_files[idx]
                    slice_buckets[slice_key].append(int(idx))

                patient_indices = np.array([
                    rng.choice(candidates) for candidates in slice_buckets.values()
                ])

            n_select_patient = min(samples_per_patient[patient], len(patient_indices))
            shortfall += samples_per_patient[patient] - n_select_patient
            if len(patient_indices) > 0:
                selected_indices.extend(rng.choice(patient_indices, size=n_select_patient, replace=False))

        if shortfall > 0:
            logger.warning(
                f"Slice de-duplication: {n_select} patches requested, but only {n_select - shortfall} "
                f"unique slices are available. Final buffer size: {len(selected_indices)}.")

        metrics = {
            'method': 'stratified_random',
            'selected_ratio': len(selected_indices) / n_samples,
            'n_patients': n_patients,
            'samples_per_patient': samples_per_patient,
        }
        return np.sort(np.array(selected_indices)), metrics
