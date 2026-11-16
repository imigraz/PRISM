import argparse
import os
import json
import shutil
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, List
import glob


def create_dataset_json(output_dir: str, num_training: int) -> None:
    """
    Create the dataset.json file required by nnUNet.
    """
    dataset = {
        "name": "HarP",
        "description": "ADNI hippocampus segmentations",
        "reference": "ADNI Dataset",
        "licence": "see ADNI",
        "release": "1.0",
        "tensorImageSize": "3D",
        "modality": {
            "0": "MRI"
        },
        "labels": {
            "0": "background",
            "1": "hippocampus"
        },
        "numTraining": num_training,
        "numTest": 0,
        "training": [],
        "test": []
    }

    with open(os.path.join(output_dir, "dataset.json"), 'w') as f:
        json.dump(dataset, f, indent=4)


def get_label_centroid(label_image: sitk.Image) -> Tuple[int, int, int]:
    """
    Calculate the centroid of the labeled region.
    """
    # First check if there are any non-zero voxels
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(label_image)

    if stats_filter.GetMaximum() == 0:
        raise ValueError("No hippocampus label found - all voxels are zero")

    # Get label statistics
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(label_image)

    # Check if label 1 exists
    if not stats.HasLabel(1):
        # Sometimes the label might not be exactly 1, let's check what labels exist
        labels = stats.GetLabels()
        print(f"Available labels: {labels}")

        if len(labels) == 1 and 0 in labels:
            raise ValueError("Only background label (0) found in the image")
        if len(labels) == 1 and 1 in labels:
            raise ValueError("Only foreground label (1) found in the image")
        elif len(labels) >= 2:
            # Use the first non-zero label
            non_zero_labels = [l for l in labels if l != 0]
            if non_zero_labels:
                target_label = non_zero_labels[0]
                print(f"Using label {target_label} instead of 1")
            else:
                raise ValueError("No non-zero labels found")
        else:
            raise ValueError("No valid hippocampus label found in the image")
    else:
        target_label = 1

    # Get centroid of the target label
    centroid_coords = stats.GetCentroid(target_label)

    # Convert physical coordinates to voxel coordinates
    centroid_voxel = label_image.TransformPhysicalPointToIndex(centroid_coords)

    return centroid_voxel


def crop_around_center(image: sitk.Image,
                       center_point: Tuple[int, int, int],
                       crop_size: Tuple[int, int, int]) -> sitk.Image:
    """
    Crop the image around a specified center point.
    """
    size = image.GetSize()

    # Calculate start index for cropping
    start = [
        max(0, center_point[0] - crop_size[0] // 2),
        max(0, center_point[1] - crop_size[1] // 2),
        max(0, center_point[2] - crop_size[2] // 2)
    ]

    # Adjust start index if it would exceed image boundaries
    for i in range(3):
        if start[i] + crop_size[i] > size[i]:
            start[i] = max(0, size[i] - crop_size[i])

    # Set up the cropping filter
    cropper = sitk.ExtractImageFilter()
    cropper.SetSize(crop_size)
    cropper.SetIndex(start)

    return cropper.Execute(image)


def get_label_statistics(label_image: sitk.Image) -> dict:
    """
    Get statistics about the label image to understand its intensity distribution.
    """
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(label_image)

    return {
        'min': stats_filter.GetMinimum(),
        'max': stats_filter.GetMaximum(),
        'mean': stats_filter.GetMean(),
        'std': stats_filter.GetSigma()
    }


def normalize_label(label_image: sitk.Image) -> sitk.Image:
    """
    Normalize label image to binary values (0 and 1).
    Handles cases where labels have high intensity values.
    """
    # Get statistics to understand the label values
    stats = get_label_statistics(label_image)
    print(f"Label stats - Min: {stats['min']}, Max: {stats['max']}, Mean: {stats['mean']:.2f}")

    # If the image is already binary (0 and 1), return as is
    if stats['min'] == 0 and stats['max'] == 1:
        return label_image

    # If max value is very high (like 255 or higher), it's likely a standard segmentation
    # where background is 0 and foreground is high value
    if stats['max'] > 10:
        # Threshold: voxels at the maximum label value become 1, all others 0
        binary_label = sitk.BinaryThreshold(label_image,  stats['max']-0.00001, stats['max'], 1, 0)

    return binary_label


def combine_labels(left_label: sitk.Image, right_label: sitk.Image) -> sitk.Image:
    """
    Combine left and right hippocampus labels into a single label.
    """
    print("Processing left hippocampus label...")
    left_normalized = normalize_label(left_label)

    print("Processing right hippocampus label...")
    right_normalized = normalize_label(right_label)

    # Combine masks using OR operation
    combined = sitk.Or(left_normalized, right_normalized)

    # Ensure the output is binary (0 and 1)
    combined = sitk.Cast(combined, sitk.sitkUInt8)

    return combined


def find_mri_file(subject_dir: str) -> str:
    """
    Find the MRI file in the complex ADNI directory structure.
    """
    # Look for .nii files in the nested directory structure
    nii_files = glob.glob(os.path.join(subject_dir, "**", "*.nii"), recursive=True)

    if not nii_files:
        raise FileNotFoundError(f"No .nii file found in {subject_dir}")

    # Filter out any files that might be labels or other non-T1 files
    # Usually the main T1 file has a long name with the acquisition details
    mri_files = [f for f in nii_files if "ADNI_" in os.path.basename(f) and len(os.path.basename(f)) > 50]

    if not mri_files:
        # If no long filename found, take the first .nii file
        mri_files = nii_files

    return mri_files[0]


def find_label_files(subject_name: str, labels_dir: str) -> Tuple[str, str]:
    """
    Find the left and right hippocampus label files for a subject.
    """
    # Look for label files with the subject name, excluding CSF files
    left_pattern = os.path.join(labels_dir, f"ADNI_{subject_name}_*_L.nii")
    right_pattern = os.path.join(labels_dir, f"ADNI_{subject_name}_*_R.nii")

    left_files = [f for f in glob.glob(left_pattern) if "CSF" not in os.path.basename(f)]
    right_files = [f for f in glob.glob(right_pattern) if "CSF" not in os.path.basename(f)]

    if not left_files:
        raise FileNotFoundError(f"No left hippocampus label found for {subject_name}")
    if not right_files:
        raise FileNotFoundError(f"No right hippocampus label found for {subject_name}")

    # Take the first match (in case of duplicates)
    return left_files[0], right_files[0]


def convert_nii_to_nii_gz(input_path: str, output_path: str) -> None:
    """
    Convert .nii file to .nii.gz format.
    """
    image = sitk.ReadImage(input_path)
    sitk.WriteImage(image, output_path)


def process_subject(subject_name: str,
                    subject_dir: str,
                    labels_dir: str,
                    output_dir: str,
                    output_index: int,
                    crop_size: Tuple[int, int, int]) -> bool:
    """
    Process a single subject's data: load T1w and labels, combine labels, crop, and save.
    Returns True if successful, False otherwise.
    """
    try:
        # Find MRI file
        mri_file = find_mri_file(subject_dir)
        print(f"Found MRI file: {mri_file}")

        # Find label files
        left_label_file, right_label_file = find_label_files(subject_name, labels_dir)
        print(f"Found label files: {left_label_file}, {right_label_file}")

        # Load images using SimpleITK
        t1w_img = sitk.ReadImage(mri_file)
        left_label = sitk.ReadImage(left_label_file)
        right_label = sitk.ReadImage(right_label_file)

        # Ensure all images have the same size and spacing
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(t1w_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for labels

        left_label = resampler.Execute(left_label)
        right_label = resampler.Execute(right_label)

        # Combine labels
        print("Combining left and right hippocampus labels...")
        combined_labels = combine_labels(left_label, right_label)

        # Verify the combined labels have content
        combined_stats = get_label_statistics(combined_labels)
        print(f"Combined label stats - Min: {combined_stats['min']}, Max: {combined_stats['max']}")

        if combined_stats['max'] == 0:
            print(f"Warning: Combined labels are empty for {subject_name}")
            return False

        # Get the centroid of the combined labels
        try:
            centroid = get_label_centroid(combined_labels)
        except Exception as e:
            print(f"Error getting centroid for {subject_name}: {str(e)}")
            return False

        # Crop both T1w and labels around the hippocampus center
        cropped_t1w = crop_around_center(t1w_img, centroid, crop_size)
        cropped_labels = crop_around_center(combined_labels, centroid, crop_size)

        # Create output directories
        imagesTr_dir = os.path.join(output_dir, "imagesTr")
        labelsTr_dir = os.path.join(output_dir, "labelsTr")
        os.makedirs(imagesTr_dir, exist_ok=True)
        os.makedirs(labelsTr_dir, exist_ok=True)

        # Generate output filenames with zero-padded indices
        output_image_name = f"HarP_{output_index:03d}_0000.nii.gz"
        output_label_name = f"HarP_{output_index:03d}.nii.gz"

        # Save cropped images
        sitk.WriteImage(cropped_t1w, os.path.join(imagesTr_dir, output_image_name))
        sitk.WriteImage(cropped_labels, os.path.join(labelsTr_dir, output_label_name))

        # Update dataset.json with new training case
        dataset_json = os.path.join(output_dir, "dataset.json")
        if os.path.exists(dataset_json):
            with open(dataset_json, 'r') as f:
                dataset = json.load(f)

            dataset["training"].append({
                "image": f"./imagesTr/{output_label_name}",
                "label": f"./labelsTr/{output_label_name}"
            })

            with open(dataset_json, 'w') as f:
                json.dump(dataset, f, indent=4)

        print(f"Successfully processed subject {subject_name} -> {output_image_name}")
        return True

    except Exception as e:
        print(f"Error processing subject {subject_name}: {str(e)}")
        return False


def get_sorted_subject_folders(adni_dir: str) -> List[str]:
    """
    Get all subject folders sorted numerically and lexically.
    """
    subject_folders = []

    # Get all directories in ADNI folder
    for item in os.listdir(adni_dir):
        item_path = os.path.join(adni_dir, item)
        if os.path.isdir(item_path):
            subject_folders.append(item)

    # Sort the folders (they should already be in the right format like 002_S_0295)
    subject_folders.sort()

    return subject_folders


def main():
    parser = argparse.ArgumentParser(description="Create Task099_HarP from the ADNI T1 images and the EADC-ADNI "
                                                 "harmonized hippocampal protocol (HarP) labels.")
    parser.add_argument("--adni", required=True,
                        help="ADNI download folder with one sub-folder per subject (e.g. 002_S_0295).")
    parser.add_argument("--labels", required=True,
                        help="HarP label folder (Released_data_NII_v1.3) with ADNI_<subject>_*_L.nii / _R.nii.")
    parser.add_argument("--output", required=True,
                        help="Output nnU-Net task folder, e.g. $nnUNet_raw_data_base/nnUNet_raw_data/Task099_HarP.")
    args = parser.parse_args()
    adni_base_dir = args.adni
    labels_base_dir = args.labels
    output_base_dir = args.output
    crop_size = (96, 64, 64)  # Adjust as needed

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Get sorted list of subject folders
    subject_folders = get_sorted_subject_folders(adni_base_dir)
    print(f"Found {len(subject_folders)} subject folders")

    # Create initial dataset.json (we'll update the count after processing)
    create_dataset_json(output_base_dir, len(subject_folders))

    # Process each subject
    successful_count = 0
    for i, subject_folder in enumerate(subject_folders):
        subject_dir = os.path.join(adni_base_dir, subject_folder)
        print(f"Processing subject {i + 1}/{len(subject_folders)}: {subject_folder}...")

        success = process_subject(
            subject_name=subject_folder,
            subject_dir=subject_dir,
            labels_dir=labels_base_dir,
            output_dir=output_base_dir,
            output_index=successful_count,  # Use successful_count for zero-based indexing
            crop_size=crop_size
        )

        if success:
            successful_count += 1

    # Update final dataset.json with actual count
    dataset_json = os.path.join(output_base_dir, "dataset.json")
    if os.path.exists(dataset_json):
        with open(dataset_json, 'r') as f:
            dataset = json.load(f)

        dataset["numTraining"] = successful_count

        with open(dataset_json, 'w') as f:
            json.dump(dataset, f, indent=4)

    print(f"Dataset creation completed! Successfully processed {successful_count}/{len(subject_folders)} subjects")


if __name__ == "__main__":
    main()