"""
========================================================================================================================
Package
========================================================================================================================
"""
from typing import Literal

import numpy as np
import nibabel as nib

from scipy import ndimage


"""
========================================================================================================================
Otsu Algorithm
========================================================================================================================
"""
def otsu_algo(mode: str | Literal['CT', 'MR', 'PET'],
              file_path: str = None,
              save_path: str = None,
              temp_path: str = None,
              overlay: bool = False) -> None:

    print()
    print('===========================================================================================================')
    print("Otsu's Algorithm")
    print('===========================================================================================================')
    print()

    # Load Data and Image
    datum = nib.load(file_path)
    image = datum.get_fdata().astype('float32')

    # Check Mode
    if mode == 'CT':
        # Remove CT Background with HU
        image = np.where(image > -250, image, -1000)
        # Set Air Value
        air_value = -1000
    elif mode == 'MR':
        # Set Air Value
        air_value = 0
    elif mode == 'PET':
        # Set Air Value
        air_value = 0
    else:
        # Error
        raise ValueError('Invalid Mode. Mode Must Be "CT", "MR", or "PET".')

    # Sort in Ascending Order
    sorted = np.sort(image.flatten())

    # Cumulative Distribution
    cdf = np.cumsum(sorted) / np.sum(sorted)

    # Get Threshold (90,)
    percentile = np.arange(50, 200) / 1000

    index = np.searchsorted(cdf, percentile)
    value = sorted[index]

    # Thresholding (90, H, W, D)
    binary = image > value[:, None, None, None]  

    # Compute Weight (90,)
    weight_1 = binary.sum(axis = (1, 2, 3)) / image.size
    weight_0 = 1 - weight_1

    # Extrene Case
    valid = (weight_1 > 0) & (weight_0 > 0)

    # Compute Variance (90,)
    var_1 = np.var(image *  binary, axis = (1, 2, 3), where =  binary)
    var_0 = np.var(image * ~binary, axis = (1, 2, 3), where = ~binary)

    # Compute Criteria (90,)
    criteria = np.array((weight_0[valid] * var_0[valid]) + (weight_1[valid] * var_1[valid]))

    # Get Best Threshold in All Criteria
    index = np.searchsorted(cdf, percentile[criteria.argmin()])
    value = sorted[index]

    # Thresholding
    binary = (image > value)

    # Get Connective Component
    components, features = ndimage.label(binary)

    # Compute Size of Each Component
    sizes = ndimage.sum(binary, components, range(1, features + 1))

    # Find Largest Component
    largest = np.argmax(sizes) + 1

    # Slect Largest Component
    hmask = (components == largest)

    for j in range(hmask.shape[2]):

        # Thresholding
        binary = hmask[:, :, j]

        # Fill Holes + Remove Small Conjection
        binary = ndimage.binary_fill_holes(binary, structure = np.ones((3, 3)))
        binary = ndimage.binary_erosion(binary, structure = np.ones((2, 2)), iterations = 2)

        # Get Connective Component
        components, features = ndimage.label(binary)

        # Compute Centroids
        centroids = np.array(ndimage.center_of_mass(binary, components, range(1, features + 1)))

        # Compute Center
        image_center = np.array(binary.shape) // 2

        # Extreme Case
        if centroids.size == 0:
            hmask[:, :, j] = np.zeros((hmask.shape[0], hmask.shape[1]))
            continue

        # Find the Component Closest to the Center
        distances = np.linalg.norm(centroids - image_center, axis = 1)
        center = np.argmin(distances) + 1

        # Extreme Case
        if distances.min() > 50:
            hmask[:, :, j] = np.zeros((hmask.shape[0], hmask.shape[1]))
            continue

        # Select Closest Component
        hmask[:, :, j] &= (components == center)

    # Head Mask Buffer
    mask = hmask.copy()

    # Closing Element Structure
    struct = int(hmask.shape[0] / 4 // 2) * 2 + 1
    while struct >= 3:

        # Fill Holes in Mask (Along Z-Axis)
        for j in range(hmask.shape[2]):
            hmask[:, :, j] = ndimage.binary_closing(hmask[:, :, j], np.ones((struct, struct)))
        
        # Narrow Down Element Structure
        struct = int(struct / 3 // 2) * 2 + 1

        # Element-Wise Or Operation of Refined Mask with Original Mask
        hmask |= mask

    # Apply Mask
    image = np.where(hmask, image, air_value)
    hmask = np.where(hmask, 1, 0)

    # Save Data
    image = nib.Nifti1Image(image, datum.affine, datum.header)
    if overlay:
        nib.save(image, file_path)
    else:
        nib.save(image, temp_path)

    hmask = nib.Nifti1Image(hmask, datum.affine, datum.header)
    nib.save(hmask, save_path)

    return


"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    file_path = ""
    save_path = ""
    temp_path = ""

    otsu_algo(mode = 'CT', file_path = file_path, save_path = save_path, temp_path = temp_path, overlay = False)