# Fingerprint Image Thinning with KMM Algorithm

This repository contains an implementation of the KMM (Kang-Wang-Morelli) algorithm for thinning binary images, particularly effective for fingerprint images. The implementation includes morphological enhancement techniques to improve the quality of thinned images.

## Features

- KMM thinning algorithm for creating one-pixel-wide skeletons of binary images
- Preprocessing techniques to enhance images before thinning:
  - Noise removal (median blur and Gaussian blur)
  - Dilation to connect broken ridges
- Postprocessing techniques to enhance thinned images:
  - Morphological closing to connect nearby endpoints
  - Gap bridging to connect broken segments

## Requirements

- Python 3.x
- NumPy
- PIL (Python Imaging Library)
- OpenCV (cv2)

## Usage

### Basic Usage

```python
from src.kmm import kmm_thinning

# Basic thinning without enhancements
kmm_thinning("input_image.bmp", "output_image.bmp")
```

### With Morphological Enhancements

```python
from src.kmm import kmm_thinning

# Thinning with preprocessing and postprocessing
kmm_thinning(
    "input_image.bmp", 
    "output_image.bmp", 
    apply_preprocessing=True, 
    apply_postprocessing=True
)
```

### Customizing Enhancement Parameters

```python
from src.kmm import kmm_thinning

# Define preprocessing parameters
preprocessing_params = {
    'noise_removal': True,  # Apply noise removal
    'dilation': True,       # Apply dilation
    'kernel_size': 3        # Kernel size for morphological operations
}

# Define postprocessing parameters
postprocessing_params = {
    'closing': True,        # Apply morphological closing
    'bridging': True,       # Apply gap bridging
    'kernel_size': 3        # Kernel size for morphological operations
}

# Apply KMM thinning with custom enhancement parameters
kmm_thinning(
    "input_image.bmp", 
    "output_image.bmp", 
    apply_preprocessing=True, 
    apply_postprocessing=True,
    preprocessing_params=preprocessing_params,
    postprocessing_params=postprocessing_params
)
```

## Testing

The repository includes a test script (`test_kmm.py`) that demonstrates the usage of the KMM thinning algorithm with and without enhancements:

```bash
python test_kmm.py
```

This will generate two output files:
- `example2_thinned_original.bmp`: Result of the original KMM thinning algorithm without enhancements
- `example2_thinned_enhanced.bmp`: Result of the KMM thinning algorithm with preprocessing and postprocessing enhancements

## How It Works

1. **Preprocessing**: The input image is enhanced before thinning to improve the quality of the skeleton:
   - Noise removal: Applies median blur and Gaussian blur to remove noise
   - Dilation: Connects broken ridges to ensure continuity

2. **KMM Thinning**: The enhanced image is thinned using the KMM algorithm:
   - Contour pixel labeling: Identifies edge and corner pixels
   - Neighborhood analysis: Analyzes the neighborhood of each pixel
   - Pixel deletion: Removes pixels based on predefined deletion criteria
   - Continuity restoration: Ensures the skeleton remains continuous

3. **Postprocessing**: The thinned image is enhanced to improve the quality of the skeleton:
   - Morphological closing: Connects nearby endpoints
   - Gap bridging: Connects broken segments to ensure continuity

## Example

For the example image `example2.bmp`, the original KMM thinning algorithm may produce a "shattered" skeleton with broken segments. By applying preprocessing and postprocessing techniques, the enhanced algorithm produces a more continuous skeleton with fewer breaks.