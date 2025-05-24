import numpy as np
from PIL import Image
import cv2

def preprocess_image(image, noise_removal=True, dilation=True, kernel_size=3):
    """
    Apply preprocessing techniques to enhance the image before thinning.

    Args:
        image (numpy.ndarray): Binary image array (0 for background, 1 for foreground)
        noise_removal (bool): Whether to apply noise removal
        dilation (bool): Whether to apply dilation
        kernel_size (int): Size of the kernel for morphological operations

    Returns:
        numpy.ndarray: Preprocessed binary image
    """
    # Convert to OpenCV format (0 for black, 255 for white)
    cv_image = np.where(image == 1, 255, 0).astype(np.uint8)

    if noise_removal:
        # Apply median blur to remove salt-and-pepper noise
        cv_image = cv2.medianBlur(cv_image, kernel_size)

        # Apply Gaussian blur to smooth the image
        cv_image = cv2.GaussianBlur(cv_image, (kernel_size, kernel_size), 0)

        # Apply binary threshold to get back a binary image
        _, cv_image = cv2.threshold(cv_image, 127, 255, cv2.THRESH_BINARY)

    if dilation:
        # Create a kernel for dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply dilation to connect broken ridges
        cv_image = cv2.dilate(cv_image, kernel, iterations=1)

    # Convert back to the original format (0 for background, 1 for foreground)
    return np.where(cv_image > 127, 1, 0)

def postprocess_image(image, closing=True, bridging=True, directional_closing=False, kernel_size=3,
                     directional_angles=None, directional_length=7, directional_width=1):

    """
    Apply postprocessing techniques to enhance the thinned image.

    Args:
        image (numpy.ndarray): Binary image array (0 for background, 255 for foreground)
        closing (bool): Whether to apply morphological closing
        bridging (bool): Whether to apply gap bridging
        kernel_size (int): Size of the kernel for morphological operations

    Returns:
        numpy.ndarray: Postprocessed binary image
    """
    # Convert to OpenCV format (0 for black, 255 for white)
    cv_image = image.copy()


    # Invert the image (make ridges white, background black)
    # This is necessary for morphological operations to properly connect the ridges
    cv_image = cv2.bitwise_not(cv_image)

    if closing:
        # Create a kernel for closing
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply closing to connect nearby endpoints
        cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)

    if bridging:
        # Create a kernel for bridging gaps
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply dilation followed by erosion to bridge small gaps
        cv_image = cv2.dilate(cv_image, kernel, iterations=1)
        cv_image = cv2.erode(cv_image, kernel, iterations=1)

    if directional_closing:
        # If no angles provided, use default angles
        if directional_angles is None:
            directional_angles = [0, 45, 90, 135]  # Common fingerprint ridge orientations

        # Apply directional closing for each specified angle
        for angle in directional_angles:
            cv_image = directional_close(cv_image, angle, directional_length, directional_width)

    # Invert back to original format (ridges black, background white)
    cv_image = cv2.bitwise_not(cv_image)

    return cv_image


def directional_close(image, angle, length=7, width=1):
    """
    Apply morphological closing with a directional structuring element.

    Args:
        image (numpy.ndarray): Binary image (255 for foreground, 0 for background)
        angle (float): Angle of the structuring element in degrees
        length (int): Length of the structuring element
        width (int): Width of the structuring element

    Returns:
        numpy.ndarray: Image after directional closing
    """
    # Create a line-shaped kernel
    if angle in [0, 180]:
        # Horizontal line
        kernel = np.ones((width, length), np.uint8)
    elif angle in [90, 270]:
        # Vertical line
        kernel = np.ones((length, width), np.uint8)
    else:
        # For angles that are not horizontal or vertical,
        # create a horizontal kernel and rotate it
        kernel = np.ones((width, length), np.uint8)

        # Get the rotation matrix
        center = (length // 2, width // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Determine the size of the rotated kernel
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((width * sin) + (length * cos))
        new_height = int((width * cos) + (length * sin))

        # Adjust the rotation matrix
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Rotate the kernel
        kernel = cv2.warpAffine(kernel, rotation_matrix, (new_width, new_height))

        # Binarize the kernel again in case rotation introduced non-binary values
        kernel = np.where(kernel > 0.5, 1, 0).astype(np.uint8)

    # Apply morphological closing with the directional kernel
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return closed_image


def kmm_thinning(image_path, output_path=None, apply_preprocessing=False, apply_postprocessing=False, 
                preprocessing_params=None, postprocessing_params=None):
    """
    Implements the KMM (Kang-Wang-Morelli) algorithm for thinning binary images.
    Particularly effective for fingerprint images.

    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the output image. If None, returns the thinned image array
        apply_preprocessing (bool): Whether to apply preprocessing techniques before thinning
        apply_postprocessing (bool): Whether to apply postprocessing techniques after thinning
        preprocessing_params (dict, optional): Parameters for preprocessing:
            - noise_removal (bool): Whether to apply noise removal (default: True)
            - dilation (bool): Whether to apply dilation (default: True)
            - kernel_size (int): Size of the kernel for morphological operations (default: 3)
        postprocessing_params (dict, optional): Parameters for postprocessing:
            - closing (bool): Whether to apply morphological closing (default: True)
            - bridging (bool): Whether to apply gap bridging (default: True)
            - kernel_size (int): Size of the kernel for morphological operations (default: 3)

    Returns:
        numpy.ndarray or None: Thinned image array if output_path is None, otherwise None
    """
    # Set default parameters if not provided
    if preprocessing_params is None:
        preprocessing_params = {'noise_removal': True, 'dilation': True, 'kernel_size': 3}
    if postprocessing_params is None:
        postprocessing_params = {'closing': True, 'bridging': True, 'kernel_size': 3}
    # Load and convert image to binary
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # Step 1: Binary Image Initialization
    # Convert to binary where foreground (black) is 1 and background (white) is 0
    binary = np.where(img_array < 128, 1, 0)

    # Apply preprocessing if requested
    if apply_preprocessing:
        binary = preprocess_image(
            binary,
            noise_removal=preprocessing_params.get('noise_removal', True),
            dilation=preprocessing_params.get('dilation', True),
            kernel_size=preprocessing_params.get('kernel_size', 3)
        )

    # Predefine the deletion array as specified
    deletion_array = np.array([
        3, 5, 7, 12, 13, 14, 15, 20,
        21, 22, 23, 28, 29, 30, 31, 48,
        52, 53, 54, 55, 56, 60, 61, 62,
        63, 65, 67, 69, 71, 77, 79, 80,
        81, 83, 84, 85, 86, 87, 88, 89,
        91, 92, 93, 94, 95, 97, 99, 101,
        103, 109, 111, 112, 113, 115, 116, 117,
        118, 119, 120, 121, 123, 124, 125, 126,
        127, 131, 133, 135, 141, 143, 149, 151,
        157, 159, 181, 183, 189, 191, 192, 193,
        195, 197, 199, 205, 207, 208, 209, 211,
        212, 213, 214, 215, 216, 217, 219, 220,
        221, 222, 223, 224, 225, 227, 229, 231,
        237, 239, 240, 241, 243, 244, 245, 246,
        247, 248, 249, 251, 252, 253, 254, 255
    ])

    # Define the 8-connected neighbor weights
    weights = np.array([128, 1, 2, 64, 0, 4, 32, 16, 8])

    # Create a copy of the binary image for processing
    thinned = binary.copy()

    # Iterative thinning until no more pixels can be removed
    iteration = 0
    changed = True

    while changed:
        iteration += 1
        changed = False

        # Step 2: Contour Pixel Labeling
        labeled = thinned.copy()

        # Pad the image to handle border pixels
        padded = np.pad(thinned, pad_width=1, mode='constant', constant_values=0)

        # Identify contour pixels (adjacent to background)
        for i in range(1, padded.shape[0] - 1):
            for j in range(1, padded.shape[1] - 1):
                if padded[i, j] == 1:  # Foreground pixel
                    # Get 8-connected neighborhood
                    neighbors = padded[i-1:i+2, j-1:j+2].flatten()

                    # Check if it's a contour pixel (adjacent to background)
                    if 0 in neighbors:
                        # Check if it's a corner pixel (diagonally adjacent to background)
                        if (padded[i-1, j-1] == 0 or padded[i-1, j+1] == 0 or 
                            padded[i+1, j-1] == 0 or padded[i+1, j+1] == 0):
                            labeled[i-1, j-1] = 3  # Corner pixel
                        else:
                            labeled[i-1, j-1] = 2  # Edge pixel

        # Step 3: Neighborhood Analysis
        marked_for_deletion = np.zeros_like(labeled)

        for i in range(1, labeled.shape[0] - 1):
            for j in range(1, labeled.shape[1] - 1):
                if labeled[i, j] in [2, 3]:  # Edge or corner pixel
                    # Get 8-connected neighborhood
                    neighborhood = labeled[i-1:i+2, j-1:j+2].flatten()

                    # Count foreground neighbors (values 1, 2, or 3)
                    neighbor_count = np.sum((neighborhood > 0) & (neighborhood <= 3)) - 1  # Subtract 1 to exclude the pixel itself

                    # Mark for deletion if it has 2, 3, or 4 neighbors
                    if neighbor_count in [2, 3, 4]:
                        marked_for_deletion[i, j] = 4

        # Step 4: Pixel Deletion
        for i in range(1, labeled.shape[0] - 1):
            for j in range(1, labeled.shape[1] - 1):
                if marked_for_deletion[i, j] == 4:
                    # Get 8-connected neighborhood
                    neighborhood = labeled[i-1:i+2, j-1:j+2].copy()
                    neighborhood[1, 1] = 0  # Set center pixel to 0 for weight calculation

                    # Calculate neighbor weight sum
                    weight_sum = 0
                    for k in range(9):
                        if k != 4:  # Skip the center pixel
                            row, col = k // 3, k % 3
                            if neighborhood[row, col] > 0:
                                weight_sum += weights[k]

                    # Delete pixel if weight sum is in deletion array
                    if weight_sum in deletion_array:
                        thinned[i, j] = 0
                        changed = True

        # Step 5: Continuity Restoration
        # Check if remaining edge/corner pixels are critical for continuity
        for i in range(1, labeled.shape[0] - 1):
            for j in range(1, labeled.shape[1] - 1):
                if labeled[i, j] in [2, 3] and thinned[i, j] == 1:
                    # Get 8-connected neighborhood
                    neighborhood = thinned[i-1:i+2, j-1:j+2].copy()
                    neighborhood[1, 1] = 0  # Set center pixel to 0

                    # Count foreground neighbors
                    neighbor_count = np.sum(neighborhood)

                    # If removing this pixel would break continuity, keep it
                    if neighbor_count <= 1:
                        thinned[i, j] = 1

    # Convert back to binary image format (0 for background, 255 for foreground)
    result = np.where(thinned == 1, 0, 255).astype(np.uint8)

    # Apply postprocessing if requested
    if apply_postprocessing:
        result = postprocess_image(
            result,
            closing=postprocessing_params.get('closing', True),
            bridging=postprocessing_params.get('bridging', True),
            kernel_size=postprocessing_params.get('kernel_size', 3),
            directional_closing=postprocessing_params.get('directional_closing', False),
            directional_angles=postprocessing_params.get('directional_angles', None),
            directional_length=postprocessing_params.get('directional_length', 7),
            directional_width=postprocessing_params.get('directional_width', 1)
        )

    if output_path:
        # Save the thinned image
        Image.fromarray(result).save(output_path)
        return None
    else:
        return result
