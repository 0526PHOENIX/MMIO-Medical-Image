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

    # Flatten Data
    flat = image.flatten()

    # Sort in Ascending Order
    sorted = np.sort(flat)

    # Get Cumulative Distribution
    dis = np.cumsum(sorted)
    dis = dis / dis[-1]

    # Get Criteria
    criteria = []
    threshold_range = range(5, 100)
    for j in threshold_range:

        # Get Threshold
        index = np.where(dis <= j / 1600)[0][-1]
        value = sorted[index]

        # Thresholding
        binary = (image > value)

        # Compute Weight
        weight_1 = binary.sum() / image.size
        weight_0 = 1 - weight_1

        # Extrene Case
        if weight_1 == 0 or weight_0 == 0:
            criteria.append(np.inf)
            continue

        # Compute Variance
        var_1 = image[binary == 1].var() if image[binary == 1].size > 0 else 0
        var_0 = image[binary == 0].var() if image[binary == 0].size > 0 else 0

        # Save Criteria to Buffer
        criteria.append(weight_0 * var_0 + weight_1 * var_1)

    # Python List to Numpy Array
    criteria = np.array(criteria)

    # Get Best Threshold in All Criteria
    index = np.where(dis <= threshold_range[criteria.argmin()] / 1600)[0][-1]
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