from src.stitching.kmm import kmm_thinning
from src.stitching.morphorogical_skeletonization import morphological_skeletonization
import numpy as np
import cv2

def test_kmm_thinning(input_path, output_path, apply_preprocessing=False, apply_postprocessing=False):
    """
    Test function to demonstrate the KMM thinning algorithm.

    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the output image
        apply_preprocessing (bool): Whether to apply preprocessing techniques
        apply_postprocessing (bool): Whether to apply postprocessing techniques
    """
    kmm_thinning(
        input_path, 
        output_path, 
        apply_preprocessing=apply_preprocessing, 
        apply_postprocessing=apply_postprocessing
    )
    print(f"Thinned image saved to {output_path}")

def test_kmm_with_enhancements(input_path, output_path):
    """
    Test function to demonstrate the KMM thinning algorithm with morphological enhancements.

    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the output image
    """
    # Define preprocessing parameters
    preprocessing_params = {
        'noise_removal': True,
        'dilation': True,
        'kernel_size': 3
    }

    # Define postprocessing parameters
    postprocessing_params = {
        'closing': True,
        'bridging': True,
        'kernel_size': 3,
        'directional_closing': False,

    }

    # Apply KMM thinning with enhancements
    kmm_thinning(
        input_path, 
        output_path, 
        apply_preprocessing=False,
        apply_postprocessing=True,
        preprocessing_params=preprocessing_params,
        postprocessing_params=postprocessing_params
    )
    print(f"Enhanced thinned image saved to {output_path}")


if __name__ == "__main__":
    # Test original KMM thinning
    # test_kmm_thinning("example2.bmp", "example2_thinned_original.bmp")

    # Test KMM thinning with enhancements
    test_kmm_with_enhancements("data/example2.bmp", "data/example2_thinned_enhanced.bmp")
    
    test_image = cv2.imread("data/example2.bmp", cv2.IMREAD_GRAYSCALE)
    skeleton_result = morphological_skeletonization(test_image)

    # Wyświetl wyniki
    cv2.imshow("Oryginalny Obraz Binarny", test_image)
    cv2.imshow("Szkielet Morfologiczny", skeleton_result)
