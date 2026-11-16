#!/usr/bin/env python3
"""
nnUNet Dataset Creator for Spine Data
Splits the spine data by acquisition site into four nnUNet datasets (Task611_Spine_A ... Task614_Spine_D).
The input folder holds imagesTr/<site>-<subject>-image.nii.gz and labelsTr/<site>-<subject>-mask-r<rater>.nii.gz;
the rater masks are merged by consensus voting.
"""

import argparse
import os
import json
import shutil
import nibabel as nib
import numpy as np
from pathlib import Path
import re
from collections import defaultdict


def merge_label_masks(mask_files, majority_threshold=0.75):
    """
    Merge multiple label masks using majority voting with consensus requirement.
    Following official strategy: consensus defined as voxels receiving three or more rater votes.
    Handles multi-class labels: 0=background, 1=white matter, 2=gray matter.

    Args:
        mask_files: List of paths to mask files for the same case
        majority_threshold: Fraction of annotators that must agree (default: 0.75 for 3/4 consensus)

    Returns:
        Merged mask as nibabel image
    """
    if not mask_files:
        raise ValueError("No mask files provided")

    num_annotators = len(mask_files)
    min_votes_needed = int(np.ceil(num_annotators * majority_threshold))

    print(f"    Using consensus voting: {min_votes_needed}/{num_annotators} raters must agree (≥3 votes required)")

    # Load the first mask to get dimensions and affine
    first_mask = nib.load(mask_files[0])
    mask_shape = first_mask.get_fdata().shape

    # Initialize vote counters for each class (0=background, 1=white matter, 2=gray matter)
    class_votes = {0: np.zeros(mask_shape, dtype=np.int32),
                   1: np.zeros(mask_shape, dtype=np.int32),
                   2: np.zeros(mask_shape, dtype=np.int32)}

    # Count votes from all annotators for each class
    for mask_file in mask_files:
        mask = nib.load(mask_file)
        mask_data = mask.get_fdata().astype(np.int32)

        # Count votes for each class
        for class_label in [0, 1, 2]:
            class_votes[class_label] += (mask_data == class_label).astype(np.int32)

    # Create consensus mask using majority voting for each voxel
    merged_data = np.zeros(mask_shape, dtype=np.uint8)

    for i in range(mask_shape[0]):
        for j in range(mask_shape[1]):
            for k in range(mask_shape[2]):
                # Find the class with the most votes for this voxel
                votes = [class_votes[0][i, j, k], class_votes[1][i, j, k], class_votes[2][i, j, k]]
                max_votes = max(votes)

                # Only assign a class if it meets the consensus threshold
                if max_votes >= min_votes_needed:
                    # If multiple classes tie with max votes, prioritize: gray matter > white matter > background
                    if class_votes[2][i, j, k] == max_votes:
                        merged_data[i, j, k] = 2  # Gray matter
                    elif class_votes[1][i, j, k] == max_votes:
                        merged_data[i, j, k] = 1  # White matter
                    else:
                        merged_data[i, j, k] = 0  # Background
                else:
                    # If no consensus, default to background
                    merged_data[i, j, k] = 0

    # Calculate statistics for each class
    total_background = np.sum(class_votes[0] > 0)
    total_white_matter = np.sum(class_votes[1] > 0)
    total_gray_matter = np.sum(class_votes[2] > 0)

    consensus_background = np.sum(merged_data == 0)
    consensus_white_matter = np.sum(merged_data == 1)
    consensus_gray_matter = np.sum(merged_data == 2)

    print(f"    Consensus results:")
    print(f"      Background: {consensus_background} voxels")
    print(f"      White matter: {consensus_white_matter} voxels (from {total_white_matter} candidate voxels)")
    print(f"      Gray matter: {consensus_gray_matter} voxels (from {total_gray_matter} candidate voxels)")

    if total_white_matter > 0:
        wm_consensus_ratio = consensus_white_matter / total_white_matter
        print(f"      White matter consensus: {wm_consensus_ratio:.1%}")

    if total_gray_matter > 0:
        gm_consensus_ratio = consensus_gray_matter / total_gray_matter
        print(f"      Gray matter consensus: {gm_consensus_ratio:.1%}")

    # Create new nibabel image with merged data
    merged_img = nib.Nifti1Image(merged_data, first_mask.affine, first_mask.header)
    return merged_img


def parse_filename(filename):
    """
    Parse filename to extract site, patient ID, and other components.

    Examples:
        site1-sc01-image.nii.gz -> ('site1', 'sc01', 'image')
        site1-sc01-mask-r1.nii.gz -> ('site1', 'sc01', 'mask', 'r1')
    """
    # Remove .nii.gz extension
    name = filename.replace('.nii.gz', '')
    parts = name.split('-')

    if len(parts) >= 3:
        site = parts[0]
        patient_id = parts[1]
        if len(parts) == 3:
            return site, patient_id, parts[2], None
        else:
            return site, patient_id, parts[2], parts[3]
    else:
        raise ValueError(f"Unexpected filename format: {filename}")


def create_nnunet_datasets(source_folder, output_root="."):
    """
    Create four nnUNet datasets from the source folder.

    Args:
        source_folder: Folder with imagesTr/ and labelsTr/ of all sites
        output_root: Folder in which the four task folders are created
    """
    source_path = Path(source_folder)
    images_path = source_path / "imagesTr"
    labels_path = source_path / "labelsTr"

    if not images_path.exists() or not labels_path.exists():
        raise FileNotFoundError("imagesTr or labelsTr folder not found in source directory")

    # Site mapping
    site_mapping = {
        'site1': {'task_id': '611', 'name': 'A'},
        'site2': {'task_id': '612', 'name': 'B'},
        'site3': {'task_id': '613', 'name': 'C'},
        'site4': {'task_id': '614', 'name': 'D'}
    }

    # Collect all files and organize by site and patient
    image_files = defaultdict(list)  # site -> [(patient_id, filepath)]
    label_files = defaultdict(lambda: defaultdict(list))  # site -> patient_id -> [mask_files]

    # Process image files
    for img_file in images_path.glob("*.nii.gz"):
        try:
            site, patient_id, file_type, _ = parse_filename(img_file.name)
            if file_type == 'image':
                image_files[site].append((patient_id, img_file))
        except ValueError as e:
            print(f"Warning: Skipping {img_file.name} - {e}")

    # Process label files
    for label_file in labels_path.glob("*.nii.gz"):
        try:
            site, patient_id, file_type, annotator = parse_filename(label_file.name)
            if file_type == 'mask' and annotator:
                label_files[site][patient_id].append(label_file)
        except ValueError as e:
            print(f"Warning: Skipping {label_file.name} - {e}")

    # Create datasets for each site
    for site, site_info in site_mapping.items():
        if site not in image_files:
            print(f"Warning: No images found for {site}")
            continue

        task_name = f"Task{site_info['task_id']}_Spine_{site_info['name']}"
        output_dir = Path(output_root) / task_name

        # Create directory structure
        output_images_dir = output_dir / "imagesTr"
        output_labels_dir = output_dir / "labelsTr"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {task_name}...")

        # Sort images by patient ID for consistent naming
        site_images = sorted(image_files[site], key=lambda x: x[0])
        training_data = []

        for case_idx, (patient_id, image_path) in enumerate(site_images):
            # Generate new filenames
            new_case_id = f"Site{site_info['name']}_Case{case_idx:02d}"
            new_image_name = f"{new_case_id}_0000.nii.gz"
            new_label_name = f"{new_case_id}.nii.gz"

            # Copy and rename image
            output_image_path = output_images_dir / new_image_name
            shutil.copy2(image_path, output_image_path)

            # Process and merge labels
            if patient_id in label_files[site]:
                mask_files = label_files[site][patient_id]
                if mask_files:
                    try:
                        # Merge all annotator masks using consensus voting (≥3/4 raters)
                        merged_mask = merge_label_masks(mask_files)

                        # Save merged mask
                        output_label_path = output_labels_dir / new_label_name
                        nib.save(merged_mask, output_label_path)

                        # Add to training data
                        training_data.append({
                            "image": f"./imagesTr/{new_image_name}",
                            "label": f"./labelsTr/{new_label_name}"
                        })

                        print(f"  Processed {patient_id} -> {new_case_id}")
                    except Exception as e:
                        print(f"  Error processing {patient_id}: {e}")
                else:
                    print(f"  Warning: No label files found for {patient_id}")
            else:
                print(f"  Warning: No labels found for {patient_id}")

        # Create dataset.json
        dataset_config = {
            "description": f"Spine dataset for {site}",
            "labels": {
                "0": "background",
                "1": "white matter",
                "2": "gray matter"
            },
            "licence": "see the data usage licence agreement of the spinal cord grey matter challenge",
            "modality": {
                "0": "T1"
            },
            "name": task_name,
            "numTest": 0,
            "numTraining": len(training_data),
            "reference": "",
            "release": "0.0",
            "tensorImageSize": "4D",
            "test": [],
            "training": training_data
        }

        # Save dataset.json
        dataset_json_path = output_dir / "dataset.json"
        with open(dataset_json_path, 'w') as f:
            json.dump(dataset_config, f, indent=4)

        print(f"  Created {task_name} with {len(training_data)} training cases")
        print(f"  Dataset configuration saved to {dataset_json_path}")
        print()


def main():
    """Main function to run the dataset creation process."""
    parser = argparse.ArgumentParser(description="Create Task611_Spine_A ... Task614_Spine_D, one task per site.")
    parser.add_argument("--input", required=True, help="Folder with imagesTr/ and labelsTr/ of all four sites.")
    parser.add_argument("--output", required=True,
                        help="Folder in which the four task folders are created, e.g. "
                             "$nnUNet_raw_data_base/nnUNet_raw_data. Use a folder other than --input.")
    args = parser.parse_args()
    source_folder = args.input

    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' not found!")
        return

    try:
        create_nnunet_datasets(source_folder, args.output)
        print("Dataset creation completed successfully!")
        print("\nCreated datasets:")
        print("- Task611_Spine_A")
        print("- Task612_Spine_B")
        print("- Task613_Spine_C")
        print("- Task614_Spine_D")

    except Exception as e:
        print(f"Error during dataset creation: {e}")
        raise


if __name__ == "__main__":
    main()