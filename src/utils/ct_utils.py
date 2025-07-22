import os
import glob
import pathlib
from typing import List, Tuple, Optional, Union

import numpy as np
import pydicom
import scipy.ndimage
import imageio
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.spatial import distance_matrix
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_scan(path: str) -> List[pydicom.Dataset]:
    """Load DICOM scan from directory and sort by slice position.
    
    Args:
        path: Directory path containing DICOM files
        
    Returns:
        List of DICOM slices sorted by ImagePositionPatient[2]
    """
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Calculate slice thickness
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except (IndexError, AttributeError):
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    # Set slice thickness for all slices
    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def convert_to_hounsfield(slices: List[pydicom.Dataset]) -> np.ndarray:
    """Convert DICOM pixel arrays to Hounsfield units.
    
    Args:
        slices: List of DICOM slices
        
    Returns:
        3D numpy array in Hounsfield units (HU)
    """
    # Stack pixel arrays into 3D volume
    image = np.stack([s.pixel_array for s in slices])
    
    # Convert to int16 for Hounsfield units
    image = image.astype(np.int16)
    
    # Set outside-of-scan pixels to 0 (air is approximately 0 HU)
    image[image == -2000] = 0
    
    # Convert to Hounsfield units using rescale slope and intercept
    for slice_number, slice_data in enumerate(slices):
        intercept = slice_data.RescaleIntercept
        slope = slice_data.RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def get_dicom_spacing(scan: List[pydicom.Dataset]) -> np.ndarray:
    """Extract voxel spacing from DICOM scan.
    
    Args:
        scan: List of DICOM slices
        
    Returns:
        Array of [slice_thickness, pixel_spacing_x, pixel_spacing_y]
    """
    slice_thickness = float(scan[0].SliceThickness)
    pixel_spacing = [float(x) for x in scan[0].PixelSpacing]
    return np.array([slice_thickness] + pixel_spacing, dtype=np.float32)


def resample(image: np.ndarray, spacing: np.ndarray, new_spacing: List[float] = [1, 1, 1]) -> Tuple[np.ndarray, np.ndarray]:
    """Resample 3D image to new voxel spacing.
    
    Args:
        image: 3D numpy array
        spacing: Current voxel spacing [z, y, x]
        new_spacing: Target voxel spacing [z, y, x]
        
    Returns:
        Tuple of (resampled_image, actual_new_spacing)
    """
    new_spacing = np.array(new_spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    actual_new_spacing = spacing / real_resize_factor
    
    resampled_image = scipy.ndimage.zoom(
        image, real_resize_factor, mode='nearest', order=0
    )
    
    return resampled_image, actual_new_spacing


def plot_3d(image: np.ndarray, threshold: float = 0, transpose: List[int] = [0, 1, 2], step_size: int = 2) -> None:
    """Create 3D visualization of volume using marching cubes.
    
    Args:
        image: 3D numpy array to visualize
        threshold: Isosurface threshold for marching cubes
        transpose: Axis order for display [default: [0,1,2], upright: [2,1,0]]
        step_size: Step size for marching cubes algorithm
    """
    # Apply transpose for visualization
    p = image.transpose(*transpose)
    
    # Generate mesh using marching cubes
    verts, faces, _, _ = measure.marching_cubes(p, threshold, step_size=step_size)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh collection
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    mesh.set_facecolor([0.45, 0.45, 0.75])
    ax.add_collection3d(mesh)
    
    # Set axis limits
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    plt.show()


def plot_3d_with_labels(image: np.ndarray, labels: np.ndarray, threshold: float = 0, 
                        transpose: List[int] = [0, 1, 2], step_size: int = 2, show: bool = True):
    """Create 3D visualization with image and segmentation labels.
    
    Args:
        image: 3D numpy array (CT volume)
        labels: 3D numpy array (segmentation labels)
        threshold: Isosurface threshold for image
        transpose: Axis order for display [default: [0,1,2], upright: [2,1,0]]
        step_size: Step size for marching cubes algorithm
        show: Whether to display the plot
        
    Returns:
        Matplotlib 3D axis object
    """
    # Apply transpose for visualization
    p = image.transpose(*transpose)
    p2 = labels.transpose(*transpose)
    
    # Generate meshes using marching cubes
    verts, faces, _, _ = measure.marching_cubes(p, threshold, step_size=step_size)
    l_verts, l_faces, _, _ = measure.marching_cubes(p2, 0, step_size=step_size)  # Binary labels

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh collections
    image_mesh = Poly3DCollection(verts[faces], alpha=0.50)
    label_mesh = Poly3DCollection(l_verts[l_faces], alpha=0.85)
    
    # Set colors
    image_mesh.set_facecolor([0.45, 0.45, 0.75])  # Blue for image
    label_mesh.set_facecolor([1.0, 0, 0])         # Red for labels
    
    # Add meshes to plot
    ax.add_collection3d(image_mesh)
    ax.add_collection3d(label_mesh)
    
    # Set axis limits
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    if show:
        plt.show()
        
    return ax


def make_gifs(ctvol: np.ndarray, outprefix: str, chosen_views: List[str]) -> None:
    """Create animated GIFs of CT volume in different anatomical planes.
    
    Args:
        ctvol: 3D CT volume array
        outprefix: Output filename prefix (without extension)
        chosen_views: List of views to create ['axial', 'coronal', 'sagittal']
        
    Note:
        Volume should be in orientation [slices, height, width]
    """
    # Normalize to 8-bit range to prevent imageio artifacts
    # Clip to typical CT window for soft tissue
    ctvol = np.clip(ctvol, a_min=-800, a_max=400)
    ctvol = (((ctvol + 800) * 255) / (400 + 800)).astype('uint8')

    # Create GIFs for each requested view
    if 'axial' in chosen_views:
        images = [ctvol[i, :, :] for i in range(ctvol.shape[0])]
        images.reverse()  # Start from top of head
        imageio.mimsave(f'{outprefix}_axial.gif', images)
        print('\t\tCreated axial GIF')

    if 'coronal' in chosen_views:
        images = [ctvol[:, i, :] for i in range(ctvol.shape[1])]
        imageio.mimsave(f'{outprefix}_coronal.gif', images)
        print('\t\tCreated coronal GIF')

    if 'sagittal' in chosen_views:
        images = [ctvol[:, :, i] for i in range(ctvol.shape[2])]
        imageio.mimsave(f'{outprefix}_sagittal.gif', images)
        print('\t\tCreated sagittal GIF')


def filter_hounsfield_bounds(image: np.ndarray, min_bound: float = -2000, max_bound: float = 4000) -> np.ndarray:
    """Filter CT image to specific Hounsfield unit range.
    
    Args:
        image: CT volume in Hounsfield units
        min_bound: Minimum HU value to keep
        max_bound: Maximum HU value to keep
        
    Returns:
        Filtered image with out-of-range values set to -2000
    """
    filtered_image = np.copy(image)
    filtered_image[image < min_bound] = -2000
    filtered_image[image > max_bound] = -2000
    return filtered_image


def binarize(image: np.ndarray) -> np.ndarray:
    """Convert CT image to binary mask (non-air regions).
    
    Args:
        image: CT volume in Hounsfield units
        
    Returns:
        Binary mask where 1 indicates non-air regions (HU > -2000)
    """
    binary_image = np.zeros_like(image)
    binary_image[image > -2000] = 1
    return binary_image


def jaw_isolation(volume: np.ndarray, hu_threshold: Tuple[float, float] = (1600, 2000), 
                 iterations: int = 2, cut_off: float = 1.5, growth_rate: float = 1, 
                 size: Optional[List[int]] = None) -> Tuple[List[int], List[int]]:
    """Isolate jaw region from CT volume using bone density thresholding.
    
    Args:
        volume: CT volume in Hounsfield units
        hu_threshold: Min and max HU values for bone detection
        iterations: Number of outlier removal iterations
        cut_off: Distance cutoff multiplier for outlier removal
        growth_rate: Factor to increase cutoff each iteration
        size: Fixed bounding box size [z, y, x] (optional)
        
    Returns:
        Tuple of (min_coords, max_coords) defining bounding box
    """
    hu_min, hu_max = hu_threshold
    
    # Filter to bone density range and create binary mask
    filtered_image = filter_hounsfield_bounds(volume, hu_min, hu_max)
    binary_image = binarize(filtered_image)

    # Get coordinates of bone voxels
    bone_coords = np.array(list(zip(*binary_image.nonzero())))
    
    # Iteratively remove outliers to focus on jaw region
    for _ in range(iterations):
        centroid = bone_coords.mean(axis=0)
        distances = distance_matrix([centroid], bone_coords)[0]
        mean_distance = np.mean(distances)
        
        # Keep points within cutoff distance
        keep_indices = distances < (mean_distance * cut_off)
        bone_coords = bone_coords[keep_indices]
        cut_off *= growth_rate

    # Calculate bounding box
    if size is None:
        # Use actual bone distribution
        min_box = np.amin(bone_coords, axis=0).tolist()
        max_box = np.amax(bone_coords, axis=0).tolist()
    else:
        # Use fixed size centered on bone centroid
        centroid = bone_coords.mean(axis=0)
        half_size = [s // 2 for s in size]
        
        min_box = [max(0, int(centroid[i] - half_size[i])) for i in range(3)]
        max_box = [int(centroid[i] + half_size[i]) for i in range(3)]
        
        # Adjust if box goes outside volume bounds
        for i, (min_val, max_val) in enumerate(zip(min_box, max_box)):
            if min_val < 0:
                adjustment = abs(min_val)
                min_box[i] = 0
                max_box[i] = min(volume.shape[i], max_val + adjustment)
    
    return min_box, max_box


def extract_roi(volume: np.ndarray, min_region: List[int], max_region: List[int]) -> np.ndarray:
    """Extract region of interest from 3D volume.
    
    Args:
        volume: 3D numpy array
        min_region: Minimum coordinates [z, y, x]
        max_region: Maximum coordinates [z, y, x]
        
    Returns:
        Extracted subvolume
    """
    return volume[min_region[0]:max_region[0], 
                 min_region[1]:max_region[1], 
                 min_region[2]:max_region[2]]


def plot_volume_histogram(volume: np.ndarray, bins: Optional[int] = None, 
                         range: Optional[Tuple[float, float]] = None, color: str = 'c') -> None:
    """Plot histogram of volume intensities.
    
    Args:
        volume: 3D numpy array
        bins: Number of histogram bins
        range: Value range for histogram (min, max)
        color: Histogram color
    """
    plt.hist(volume.flatten(), bins=bins, range=range, color=color)
    plt.xlabel("Intensity Values")
    plt.ylabel("Frequency")
    plt.title("Volume Intensity Histogram")
    plt.grid(True, alpha=0.3)
    plt.show()


def get_sample_label_id(sample_path: str, is_nifti_dataset: bool = False) -> dict:
    """Extract data and label paths from sample directory.
    
    Args:
        sample_path: Path to sample directory
        is_nifti_dataset: Whether dataset is in NIfTI format
        
    Returns:
        Dictionary with 'data', 'label', and 'id' keys
    """
    sample_files = glob.glob(os.path.join(sample_path, '*'))
    sample_id = os.path.basename(sample_path).split('-')[1]
    
    # Handle case with only one file (no labels)
    if len(sample_files) == 1:
        sample_files.append(None)
    elif not is_nifti_dataset:
        sample_files.reverse()
    
    # Determine which file is data vs label
    if sample_files[0] and 'label' in sample_files[0]:
        label_path, data_path = sample_files[0], sample_files[1]
    else:
        data_path, label_path = sample_files[0], sample_files[1]
    
    return {
        'data': data_path,
        'label': label_path,
        'id': sample_id
    }


def split_label_channels(labels_matrix: np.ndarray, num_channels: int = 4) -> np.ndarray:
    """Split multi-class label volume into separate binary channels.
    
    Args:
        labels_matrix: 3D label volume with integer class labels
        num_channels: Number of label classes to extract
        
    Returns:
        4D array with shape (num_channels, depth, height, width)
    """
    split_channels = []
    
    for label_id in range(1, num_channels + 1):  # Start from 1 (skip background)
        # Create binary mask for this label
        binary_mask = np.zeros_like(labels_matrix)
        binary_mask[labels_matrix == label_id] = 1
        split_channels.append(binary_mask)
    
    return np.stack(split_channels)


def convert_dicom_dataset_to_nifti(data_paths_list: List[dict], new_root_path: str, 
                                  separate_label_channels: bool = True) -> None:
    """Convert DICOM dataset to NIfTI format with jaw isolation.
    
    Args:
        data_paths_list: List of sample dictionaries with 'data', 'label', 'id' keys
        new_root_path: Output directory for converted data
        separate_label_channels: Whether to split multi-class labels into channels
    """
    for index, sample_info in enumerate(data_paths_list):
        data_path = sample_info['data']
        label_path = sample_info['label']
        sample_id = sample_info['id']
        
        # Skip samples without labels
        if not label_path:
            print(f'{index}: Sample {sample_id} has no label, skipping')
            continue
            
        try:
            # Load and process DICOM data
            dicom_slices = load_scan(data_path)
            ct_volume = convert_to_hounsfield(dicom_slices)
            
            # Load label volume
            label_nifti = nib.load(label_path)
            label_volume = np.asarray(label_nifti.dataobj).transpose(2, 1, 0)

            # Resample to isotropic spacing
            spacing = get_dicom_spacing(dicom_slices)
            ct_resampled, _ = resample(ct_volume, spacing)
            label_resampled, _ = resample(label_volume, spacing)

            # Isolate jaw region
            min_box, max_box = jaw_isolation(
                ct_resampled, iterations=4, growth_rate=0.98, size=[75, 75, 75]
            )
            ct_isolated = extract_roi(ct_resampled, min_box, max_box)
            label_isolated = extract_roi(label_resampled, min_box, max_box)
            
            # Split label channels if requested
            if separate_label_channels:
                label_isolated = split_label_channels(label_isolated)

            # Create output directory
            output_dir = os.path.join(new_root_path, f'case-{sample_id}')
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Save as NIfTI files
            ct_nifti = nib.Nifti1Image(ct_isolated, np.eye(4))
            label_nifti = nib.Nifti1Image(label_isolated, np.eye(4))
            
            nib.save(ct_nifti, os.path.join(output_dir, f'{sample_id}.nii'))
            nib.save(label_nifti, os.path.join(output_dir, f'{sample_id}_label.nii'))
            
            print(f'{index}: Successfully converted sample {sample_id}')
            
        except Exception as e:
            print(f'{index}: Error converting sample {sample_id}: {str(e)}')
            continue


