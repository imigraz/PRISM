import argparse
import os
import json
import shutil
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, List


def create_dataset_json(output_dir: str, num_training: int) -> None:
    """
    Create the dataset.json file required by nnUNet.
    """
    dataset = {
        "name": "Hippocampus",
        "description": "Hippocampus segmentation from T1w MRI",
        "reference": "Custom Dataset",
        "licence": "NA",
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
    # Get label statistics
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(label_image)

    # Get centroid of label 1 (our combined hippocampus)
    centroid_coords = stats.GetCentroid(1)

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


def combine_labels(left_label: sitk.Image, right_label: sitk.Image) -> sitk.Image:
    """
    Combine left and right hippocampus labels into a single label.
    """
    # Convert to binary masks
    left_binary = sitk.BinaryThreshold(left_label, 1, 255, 1, 0)
    right_binary = sitk.BinaryThreshold(right_label, 1, 255, 1, 0)

    # Combine masks using OR operation
    combined = sitk.Or(left_binary, right_binary)

    # Ensure the output is binary (0 and 1)
    combined = sitk.BinaryThreshold(combined, 1, 255, 1, 0)

    return combined


def ensure_RAI_orientation(image: sitk.Image) -> sitk.Image:
    """
    Ensure the image is in RAI orientation
    RAI means: X increases toward Right, Y toward Anterior, Z toward Inferior

    Args:
        image (sitk.Image): Input image

    Returns:
        sitk.Image: Image oriented in RAI direction
    """

    # Set the direction matrix to RAI orientation
    # This is a 9-element tuple representing a 3x3 direction matrix
    # [1,0,0, 0,1,0, 0,0,1] represents RAI orientation
    rai_direction = (0.0, 1.0, 0.0,
                     1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0)

    # Get current direction
    current_direction = image.GetDirection()

    # If the direction is already RAI, return the original image
    if current_direction == rai_direction:
        return image

    # Otherwise, create a new image with RAI direction
    # First, get the array data
    array_data = sitk.GetArrayFromImage(image)
    image_out = sitk.GetImageFromArray(array_data)

    # Set the metadata
    image_out.SetOrigin(image.GetOrigin())
    image_out.SetSpacing(image.GetSpacing())
    image_out.SetDirection(rai_direction)

    return image_out



def process_subject(subject_dir: str,
                    output_dir: str,
                    subject_id: str,
                    crop_size: Tuple[int, int, int]) -> None:
    """
    Process a single subject's data: load T1w and labels, combine labels, crop, and save.
    """
    # Construct file paths
    t1w_file = os.path.join(subject_dir, f"{subject_id}_t1w_standard_defaced_MNI.nii.gz")
    left_hipp = os.path.join(subject_dir, f"{subject_id}_hippolabels_hres_L_MNI.nii.gz")
    right_hipp = os.path.join(subject_dir, f"{subject_id}_hippolabels_hres_R_MNI.nii.gz")

    try:
        # Load images using SimpleITK
        t1w_img = sitk.ReadImage(t1w_file)
        left_label = sitk.ReadImage(left_hipp)
        right_label = sitk.ReadImage(right_hipp)

        # Ensure all images have the same size and spacing
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(t1w_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for labels

        left_label = resampler.Execute(left_label)
        right_label = resampler.Execute(right_label)

        # Combine labels
        combined_labels = combine_labels(left_label, right_label)

        # Get the centroid of the combined labels
        try:
            centroid = get_label_centroid(combined_labels)
        except Exception as e:
            print(f"Error getting centroid for {subject_id}: {str(e)}")
            return

        # Crop both T1w and labels around the hippocampus center
        cropped_t1w = crop_around_center(t1w_img, centroid, crop_size)
        cropped_labels = crop_around_center(combined_labels, centroid, crop_size)

        # Create output directories
        imagesTr_dir = os.path.join(output_dir, "imagesTr")
        labelsTr_dir = os.path.join(output_dir, "labelsTr")
        os.makedirs(imagesTr_dir, exist_ok=True)
        os.makedirs(labelsTr_dir, exist_ok=True)

        # Save cropped images
        sitk.WriteImage(cropped_t1w, os.path.join(imagesTr_dir, f"{subject_id}_0000.nii.gz"))
        sitk.WriteImage(cropped_labels, os.path.join(labelsTr_dir, f"{subject_id}.nii.gz"))

        # Update dataset.json with new training case
        dataset_json = os.path.join(output_dir, "dataset.json")
        if os.path.exists(dataset_json):
            with open(dataset_json, 'r') as f:
                dataset = json.load(f)

            dataset["training"].append({
                "image": f"./imagesTr/{subject_id}.nii.gz",
                "label": f"./labelsTr/{subject_id}.nii.gz"
            })

            with open(dataset_json, 'w') as f:
                json.dump(dataset, f, indent=4)

        print(f"Successfully processed subject {subject_id}")

    except Exception as e:
        print(f"Error processing subject {subject_id}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Create Task098_Dryad from the Dryad hippocampus dataset "
                                                 "(subject folders s01 ... s25).")
    parser.add_argument("--input", required=True, help="Folder containing the subject folders s01 ... s25.")
    parser.add_argument("--output", required=True,
                        help="Output nnU-Net task folder, e.g. $nnUNet_raw_data_base/nnUNet_raw_data/Task098_Dryad.")
    args = parser.parse_args()
    input_base_dir = args.input
    output_base_dir = args.output
    crop_size = (96, 64, 64)
    num_subjects = 25

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Create initial dataset.json
    create_dataset_json(output_base_dir, num_subjects)

    # Process each subject
    for i in range(1, num_subjects + 1):
        subject_id = f"s{i:02d}"
        subject_dir = os.path.join(input_base_dir, subject_id)

        if os.path.exists(subject_dir):
            print(f"Processing subject {subject_id}...")
            process_subject(subject_dir, output_base_dir, subject_id, crop_size)
        else:
            print(f"Warning: Directory not found for subject {subject_id}")

    print("Dataset creation completed!")


if __name__ == "__main__":
    main()