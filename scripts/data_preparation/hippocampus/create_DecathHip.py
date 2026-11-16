import argparse
import os
import json
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Dict, List
import shutil


def ensure_LAS_orientation(image: sitk.Image) -> sitk.Image:
    """
    Ensure the image is in LAS orientation
    LAS means: X increases toward Left, Y toward Anterior, Z toward Superior

    Args:
        image (sitk.Image): Input image

    Returns:
        sitk.Image: Image oriented in LAS direction
    """

    # Set the direction matrix to LAS orientation
    # This is a 9-element tuple representing a 3x3 direction matrix
    las_direction = (-1.0, 0.0, 0.0,  # L: negative of R axis
                     0.0, 1.0, 0.0,  # A: same as anterior
                     0.0, 0.0, -1.0)  # S: same as superior

    # Get current direction
    current_direction = image.GetDirection()

    # If the direction is already LAS, return the original image
    if current_direction == las_direction:
        return image

    # Otherwise, create a new image with LAS direction
    array_data = sitk.GetArrayFromImage(image)
    image_out = sitk.GetImageFromArray(array_data)

    # Set the metadata
    image_out.SetOrigin(image.GetOrigin())
    image_out.SetSpacing(image.GetSpacing())
    image_out.SetDirection(las_direction)

    return image_out


def transform_to_look_like_LAS_in_LPI(image: sitk.Image) -> sitk.Image:
    """
    Transform the underlying data so that when displayed in LPI orientation,
    it looks like what was previously shown as LAS
    """
    # Get the array data
    array = sitk.GetArrayFromImage(image)

    # To make LPI look like LAS:
    # - Flip Y axis (anterior-posterior)
    # - Flip Z axis (superior-inferior)
    transformed_array = array[::-1, ::-1, :]  # Flip first two axes

    # Create new image with transformed data
    new_image = sitk.GetImageFromArray(transformed_array)

    # Set LPI direction matrix
    lpi_direction = (-1.0, 0.0, 0.0,  # L: positive X axis
                     0.0, -1.0, 0.0,  # P: negative Y axis
                     0.0, 0.0, 1.0)  # I: negative Z axis

    # Set the metadata
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing(image.GetSpacing())
    new_image.SetDirection(lpi_direction)

    return new_image


def load_dataset_json(task_folder: str) -> Dict:
    """
    Load and parse the dataset.json file
    """
    json_path = os.path.join(task_folder, "dataset.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"dataset.json not found in {task_folder}")

    with open(json_path, 'r') as f:
        return json.load(f)


def swap_xz_axes(image, axis=[2, 1, 0]):
    # Get the array data and transpose it
    array = sitk.GetArrayFromImage(image)
    array_swapped = array.transpose(*axis)

    # Create new image with swapped data
    new_image = sitk.GetImageFromArray(array_swapped)

    return new_image


def get_actual_image_path(base_path: str) -> str:
    """
    Check for the image file with and without '_0000' suffix
    Returns the actual path if found, otherwise returns the original path
    """
    # First check if the original path exists
    if os.path.exists(base_path):
        return base_path

    # Try with _0000 suffix
    # Handle double extension (.nii.gz) correctly
    filename = os.path.basename(base_path)  # gets 'hippocampus_367.nii.gz'
    base_stem = filename.split('.')[0]  # gets 'hippocampus_367'
    parent = Path(base_path).parent  # gets the directory path
    path_with_suffix = str(parent / f"{base_stem}_0000.nii.gz")
    if os.path.exists(path_with_suffix):
        return path_with_suffix

    # If neither exists, return original path (will lead to error handling in process_case)
    return base_path


def process_case(image_path: str, label_path: str, output_dir: str, case_identifier: str) -> None:
    """
    Process a single case: load image and label, resample the label to the image grid if needed, merge the
    labels into one binary hippocampus label, and save both in nii.gz format (orientation is kept)
    """
    # Create output directories if they don't exist
    output_image_dir = os.path.join(output_dir, "imagesTr")
    output_label_dir = os.path.join(output_dir, "labelsTr")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    try:
        # Get actual image path (handling _0000 suffix)
        actual_image_path = get_actual_image_path(image_path)

        # Load image and label
        image = sitk.ReadImage(actual_image_path)
        label = sitk.ReadImage(label_path)

        # Ensure same spacing and size
        reference_size = image.GetSize()
        reference_spacing = image.GetSpacing()

        # Resample label to match image if needed
        if label.GetSize() != reference_size or label.GetSpacing() != reference_spacing:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            label = resampler.Execute(label)

        # Convert to binary label
        label = sitk.BinaryThreshold(label, 1, 255, 1, 0)

        # Save processed images
        output_image_path = os.path.join(output_image_dir, f"{case_identifier}_0000.nii.gz")
        output_label_path = os.path.join(output_label_dir, f"{case_identifier}.nii.gz")

        sitk.WriteImage(image, output_image_path)
        sitk.WriteImage(label, output_label_path)

        print(f"Successfully processed case {case_identifier}")

        return {
            "image": f"./imagesTr/{case_identifier}.nii.gz",  # No _0000 in dataset.json
            "label": f"./labelsTr/{case_identifier}.nii.gz"
        }

    except Exception as e:
        print(f"Error processing case {case_identifier}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Create Task097_DecathHip from the Medical Segmentation Decathlon "
                                                 "hippocampus task (Task04_Hippocampus).")
    parser.add_argument("--input", required=True,
                        help="Decathlon task folder containing dataset.json, imagesTr/ and labelsTr/.")
    parser.add_argument("--output", required=True,
                        help="Output nnU-Net task folder, e.g. $nnUNet_raw_data_base/nnUNet_raw_data/Task097_DecathHip.")
    args = parser.parse_args()
    input_task_folder = args.input  # Original nnUNet task folder
    output_task_folder = args.output  # Processed output folder

    # Create output directory
    os.makedirs(output_task_folder, exist_ok=True)

    # Load dataset.json
    dataset_info = load_dataset_json(input_task_folder)

    # Keep original metadata
    output_dataset = {
        "name": dataset_info["name"],
        "description": dataset_info["description"],
        "reference": dataset_info["reference"],
        "licence": dataset_info["licence"],
        "relase": dataset_info["relase"],
        "tensorImageSize": dataset_info["tensorImageSize"],
        "modality": dataset_info["modality"],
        "labels": {
            "0": "background",
            "1": "hippocampus"
        },
        "numTraining": dataset_info["numTraining"],
        "numTest": 0,
        "training": [],
        "test": []
    }

    # Process all training cases
    processed_cases = []
    for case in dataset_info["training"]:
        image_path = os.path.join(input_task_folder, case["image"].lstrip("./"))
        label_path = os.path.join(input_task_folder, case["label"].lstrip("./"))

        # Case identifier: file name without extensions
        base_name = os.path.basename(case["image"]).split(".")[0]  # gets 'hippocampus_367' from 'hippocampus_367.nii.gz'
        case_identifier = base_name

        result = process_case(image_path, label_path, output_task_folder, case_identifier)
        if result:
            processed_cases.append({
                "image": f"./imagesTr/{case_identifier}.nii.gz",
                "label": f"./labelsTr/{case_identifier}.nii.gz"
            })

    # Update dataset info
    output_dataset["training"] = processed_cases
    output_dataset["numTraining"] = len(processed_cases)

    with open(os.path.join(output_task_folder, "dataset.json"), 'w') as f:
        json.dump(output_dataset, f, indent=4)

    print(f"\nProcessing complete!")
    print(f"Total cases processed: {len(processed_cases)}")
    print(f"Output saved to: {output_task_folder}")


if __name__ == "__main__":
    main()