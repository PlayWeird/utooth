import os

import numpy
import numpy as np  # linear algebra
import pydicom
import scipy.ndimage
import imageio
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def convert_to_hounsfield(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def get_dicom_spacing(scan):
    return np.array([float(scan[0].SliceThickness)] + [float(x) for x in scan[0].PixelSpacing], dtype=np.float32)


def resample(image, spacing, new_spacing=[1, 1, 1]):
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest', order=0)
    return image, new_spacing


# TODO build plot_3d functionality into plot_3D_with_labels...rename to plot_3D
def plot_3d(image, threshold=0, transpose=[0, 1, 2], step_size=2):
    # perform a image transformation
    # transpose = [2, 1, 0] for upright sample
    p = image.transpose(*transpose)
    verts, faces, _, _ = measure.marching_cubes(p, threshold, step_size=step_size)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def plot_3d_with_labels(image, labels, threshold=0, transpose=[0, 1, 2], step_size=2):
    # perform a image transformation
    # transpose = [2, 1, 0] for upright sample
    p = image.transpose(*transpose)
    p2 = labels.transpose(*transpose)
    verts, faces, _, _ = measure.marching_cubes(p, threshold, step_size=step_size)
    # setting threshold to 0 for binary labels
    l_verts, l_faces, _, _ = measure.marching_cubes(p2, 0, step_size=step_size)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.50)
    mesh2 = Poly3DCollection(l_verts[l_faces], alpha=0.85)
    face_color = [0.45, 0.45, 0.75]
    l_face_color = [1.0, 0, 0]
    mesh.set_facecolor(face_color)
    mesh2.set_facecolor(l_face_color)
    ax.add_collection3d(mesh)
    ax.add_collection3d(mesh2)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def make_gifs(ctvol, outprefix, chosen_views):
    """Save GIFs of the <ctvol> in the axial, sagittal, and coronal planes.
    This assumes the final orientation produced by the preprocess_volumes.py
    script: [slices, square, square].

    <chosen_views> is a list of strings that can contain any or all of
        ['axial','coronal','sagittal']. It specifies which view(s) will be
        made into gifs."""
    # https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python

    # First fix the grayscale colors.
    # imageio assumes only 256 colors (uint8): https://stackoverflow.com/questions/41084883/imageio-how-to-increase-quality-of-output-gifs
    # If you do not truncate to a 256 range, imageio will do so on a per-slice
    # basis, which creates weird brightening and darkening artefacts in the gif.
    # Thus, before making the gif, you should truncate to the 0-256 range
    # and cast to a uint8 (the dtype imageio requires):
    # how to truncate to 0-256 range: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    ctvol = np.clip(ctvol, a_min=-800, a_max=400)
    ctvol = (((ctvol + 800) * (255)) / (400 + 800)).astype('uint8')

    # Now create the gifs in each plane
    if 'axial' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[0]):
            images.append(ctvol[slicenum, :, :])
        images.reverse()
        imageio.mimsave(outprefix + '_axial.gif', images)
        print('\t\tdone with axial gif')

    if 'coronal' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[1]):
            images.append(ctvol[:, slicenum, :])
        imageio.mimsave(outprefix + '_coronal.gif', images)
        print('\t\tdone with coronal gif')

    if 'sagittal' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[2]):
            images.append(ctvol[:, :, slicenum])
        imageio.mimsave(outprefix + '_sagittal.gif', images)
        print('\t\tdone with sagittal gif')


# Filter the image between Hounsfield bounds
def filter_hounsfield_bounds(image, min_bound=-2000, max_bound=4000):
    new_image = np.copy(image)
    new_image[image < min_bound] = -2000
    new_image[image > max_bound] = -2000
    return new_image


def binarize(image):
    new_image = np.zeros_like(image)
    new_image[image > -2000] = 1
    return new_image


def jaw_isolation(volume, hu_threshold=(1600, 2000), iterations=2, cut_off=1.5, growth_rate=1, size=None):
    hu_min = hu_threshold[0]
    hu_max = hu_threshold[1]
    filtered_image = filter_hounsfield_bounds(volume, hu_min, hu_max)
    binary_image = binarize(filtered_image)

    bounding_coords = np.array(list(zip(*map(list, binary_image.nonzero()))))
    for iter_num in range(iterations):
        distance_mat = distance_matrix([bounding_coords.mean(axis=0)], bounding_coords)
        std_dev = np.mean(distance_mat)
        args_to_keep = np.argwhere(distance_mat[0] < std_dev * cut_off)
        bounding_coords = np.squeeze(bounding_coords[args_to_keep])
        cut_off *= growth_rate

    if not size:
        max_box = np.amax(bounding_coords, axis=0)
        min_box = np.amin(bounding_coords, axis=0)
    else:
        mean = bounding_coords.mean(axis=0)
        max_box = [int(mean[0] + size[0]/2), int(mean[1] + size[1]/2), int(mean[2] + size[2]/2)]
        min_box = [int(mean[0] - size[0]/2), int(mean[1] - size[1]/2), int(mean[2] - size[2]/2)]
    return min_box, max_box


def extract_roi(volume, min_region, max_region):
    return volume[min_region[0]:max_region[0], min_region[1]:max_region[1], min_region[2]:max_region[2]]


def plot_volume_histogram(volume, bins=None, range=None, color='c'):
    plt.hist(volume.flatten(), bins=bins, range=range, color=color)
    plt.xlabel("Values")
    plt.ylabel("Count")
    plt.show()
