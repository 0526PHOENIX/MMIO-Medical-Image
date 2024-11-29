# Otsu Algorithm: Specific to Medical Image

Otsu Algorithm for 3D Medical Image.


## ***def otsu_algorithm()***
    
### Otsu Algorithm
    
    * Parameter

        ** mode (str)
            - 'CT'  : Computed Tomography
            - 'MR'  : Magnetic Resonance Imaging
            - 'PET' : Positron Emission Tomography

        ** file_path (str)
            - Path of Input image

        ** save_path (str)
            - Path of Saved Head Mask

        ** temp_path (str)
            - Path of Temporarily Saved Masked Image

        ** overaly (bool)
            - True  : Original Image Will be Overwritten by Masked Image
            - False : Original Image Will Remain Unchanged