import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


def _calculate_crossing_number(window_3x3):
    """
    Oblicza liczbę skrzyżowań (Crossing Number - CN) dla centralnego piksela
    w sąsiedztwie 3x3.
    Sąsiedztwo powinno być binarne (0 dla tła, 1 dla linii).
    """
    p_neighbors = [window_3x3[0, 0], window_3x3[0, 1], window_3x3[0, 2],
                   window_3x3[1, 2], window_3x3[2, 2], window_3x3[2, 1],
                   window_3x3[2, 0], window_3x3[1, 0]]
    
    cn = 0
    for i in range(len(p_neighbors)):
        cn += abs(p_neighbors[i] - p_neighbors[(i + 1) % len(p_neighbors)])
    return cn / 2

def detect_minutiae(thinned_image, border_margin=10, select_one_of_each_type=False, min_distance = 50):
    """
    Wykrywa minucje (zakończenia i bifurkacje) na ścienionym obrazie odcisku palca.
    Oczekuje obrazu, gdzie linie papilarne są białe (255), a tło czarne (0).
    Metoda CN nie jest odpowiednia do detekcji rdzeni i delt.

    Args:
        thinned_image (numpy.ndarray): Obraz binarny po ścienianiu.
        border_margin (int): Margines od krawędzi obrazu.
        select_one_of_each_type (bool): Jeśli True, funkcja spróbuje zwrócić tylko
                                        pierwsze znalezione zakończenie i pierwszą
                                        bifurkację. Jeśli False, zwróci wszystkie.
                                        Dla rdzeni i delt zawsze zwraca puste listy.

    Returns:
        dict: Słownik zawierający listy krotek (r, c) dla każdego typu minucji.
              Jeśli select_one_of_each_type jest True, listy dla zakończeń i bifurkacji
              będą zawierać co najwyżej jeden element.
              {
                  "terminations": [(r,c)] lub [], 
                  "bifurcations": [(r,c)] lub [],
                  "cores": [],  // Zawsze puste, metoda CN nie jest odpowiednia.
                  "deltas": [] // Zawsze puste, metoda CN nie jest odpowiednia.
              }
    """
    if thinned_image is None or thinned_image.dtype != np.uint8 or len(thinned_image.shape) != 2:
        print("Nieprawidłowy obraz wejściowy dla detect_minutiae.")
        return {"terminations": [], "bifurcations": [], "cores": [], "deltas": []}

    binary_for_cn = (thinned_image == 255).astype(int) # Linie = 1, Tło = 0
    
    all_terminations = []
    all_bifurcations = []
    rows, cols = binary_for_cn.shape

    # Wykryj wszystkie potencjalne zakończenia i bifurkacje
    for r in range(border_margin, rows - border_margin):
        for c in range(border_margin, cols - border_margin):
            if binary_for_cn[r, c] == 1:  # Jeśli to piksel linii
                window = binary_for_cn[r-1:r+2, c-1:c+2]
                cn = _calculate_crossing_number(window)
                
                if cn == 1:
                    all_terminations.append((r, c))
                elif cn == 3:
                    all_bifurcations.append((r, c))
    
    # Metoda CN nie jest odpowiednia do detekcji rdzeni i delt.
    cores_list = [] 
    deltas_list = []

    if select_one_of_each_type:
        selected_terminations = all_terminations[:1] 
        selected_bifurcations = all_bifurcations[:1]
    else:
        selected_terminations = all_terminations
        selected_bifurcations = all_bifurcations

    filtered_terminations = distance_filter(selected_terminations, min_distance = min_distance)
    filtered_bifurcations = distance_filter(selected_bifurcations, min_distance = min_distance)

    return {
        "terminations": filtered_terminations,
        "bifurcations": filtered_bifurcations,
        "cores": cores_list, # Zawsze puste
        "deltas": deltas_list  # Zawsze puste
    }

def draw_minutiae(image_to_draw_on, minutiae_data_dict, 
                  show_labels=True, marker_radius=4, font_scale=0.35, line_thickness=1):
    """
    Rysuje wykryte minucje na podanym obrazie używając pełnych polskich nazw.
    minutiae_data_dict to słownik zwrócony przez detect_minutiae.
    """
    if image_to_draw_on is None:
        print("Obraz do rysowania minucji jest None.")
        # Zwróć kopię pustego obrazu lub obrazu wejściowego, jeśli nie ma co rysować
        return image_to_draw_on if image_to_draw_on is not None else np.zeros((100,100,3), dtype=np.uint8)


    output_image = image_to_draw_on.copy()
    if len(output_image.shape) == 2: # Jeśli obraz jest w skali szarości
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)

    minutiae_config = {
        "terminations": {"color": (0, 0, 255), "label": "Zakonczenie"}, 
        "bifurcations": {"color": (0, 255, 0), "label": "Bifurkacja"}, 
        "cores": {"color": (255, 0, 0), "label": "Rdzen"},        
        "deltas": {"color": (255, 165, 0), "label": "Delta"} # Orange for Delta
    }

    for m_type, minutiae_list in minutiae_data_dict.items():
        if m_type in minutiae_config and minutiae_list: 
            config = minutiae_config[m_type]
            color = config["color"]
            full_label = config["label"]
            
            for r, c in minutiae_list: 
                cv2.circle(output_image, (c, r), marker_radius, color, line_thickness)
                if show_labels:
                    text_origin_x = c + marker_radius + 3
                    text_origin_y = r + marker_radius // 2
                    
                    img_h, img_w = output_image.shape[:2]
                    (text_width, text_height), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
                    
                    if text_origin_x + text_width > img_w - 5: # 5px margin from right
                        text_origin_x = c - marker_radius - 3 - text_width 
                    if text_origin_x < 5 : # 5px margin from left
                        text_origin_x = 5
                    
                    if text_origin_y - text_height < 5: # 5px margin from top
                        text_origin_y = r + marker_radius + 3 + text_height 
                    elif text_origin_y + text_height // 2 > img_h - 5: # 5px margin from bottom
                         text_origin_y = r - marker_radius - 3 - text_height // 2
                    if text_origin_y < 5: # ensure not cut at top after adjustment
                        text_origin_y = 5 + text_height

                    cv2.putText(output_image, full_label, (text_origin_x, text_origin_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, line_thickness)
    return output_image




############################
# Improvements
############################


def compute_orientation_blocks(thinned_image, block_size=16):
    # Ensure the input is binary (0 and 1)
    thinned_image = (thinned_image > 0).astype(np.uint8)
    h, w = thinned_image.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    orientations = np.full((h_blocks, w_blocks), np.nan, dtype=np.float32)
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = thinned_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            y, x = np.where(block == 1)
            if len(x) < 2:
                continue  # Skip blocks with insufficient points
            
            # PCA to compute orientation
            x_mean, y_mean = np.mean(x), np.mean(y)
            x_centered = x - x_mean
            y_centered = y - y_mean
            Cxx = np.sum(x_centered ** 2)
            Cyy = np.sum(y_centered ** 2)
            Cxy = np.sum(x_centered * y_centered)
            
            if (Cxx - Cyy) == 0 and Cxy == 0:
                theta = 0.0
            else:
                theta = 0.5 * np.arctan2(2 * Cxy, (Cxx - Cyy))
            
            orientations[i, j] = np.degrees(theta) % 180  # Store in 0-180 degrees
    
    valid_orientations = np.sum(~np.isnan(orientations))
    print(f"Valid orientation blocks: {valid_orientations}/{h_blocks * w_blocks}")
    return orientations

def compute_poincare_index(orientations, i, j):
    # Check if the block or its neighbors are out of bounds
    if i < 1 or i >= orientations.shape[0]-1 or j < 1 or j >= orientations.shape[1]-1:
        return 0
    
    # ADD
    # orientations = gaussian_filter(orientations, sigma=1, mode='nearest')
    
    # Collect 8 neighboring orientations (clockwise)
    neighbors = [
        (i-1, j-1), (i-1, j), (i-1, j+1),
        (i, j+1),   (i+1, j+1), (i+1, j),
        (i+1, j-1), (i, j-1)
    ]
    angles = []
    for ni, nj in neighbors:
        angle = orientations[ni, nj]
        if np.isnan(angle):
            return 0  # Skip if any neighbor is invalid
        angles.append(angle)
    
    # Calculate cumulative angle difference (adjusted for 180-degree ambiguity)
    sum_diff = 0
    for k in range(8):
        current = angles[k]
        next_angle = angles[(k+1) % 8]
        diff = next_angle - current
        # Handle 180-degree wrap-around correctly
        diff = (diff + 90) % 180 - 90  # Now in range [-90, 90)
        sum_diff += diff
    
    # Core (+180) or Delta (-180) check with tolerance
    if 160 < sum_diff < 200:    # Core (sum ~ +180)
        return 1
    elif -200 < sum_diff < -160: # Delta (sum ~ -180)
        return -1
    
    else:
        return 0

def cluster_points(points, eps=25):
    if len(points) == 0:
        return []
    points_array = np.array(points)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(points_array)
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(points[idx])
    return list(clusters.values())

def detect_cores_deltas(thinned_image, block_size = 8, cluster_eps=30, debug=False):

    h, w = thinned_image.shape
    orientations = compute_orientation_blocks(thinned_image, block_size)

    if debug:
        plt.imshow(orientations, cmap='hsv')
        plt.colorbar()
        plt.title("Orientation Field")
        plt.show()
    
    core_candidates = []
    delta_candidates = []
    
    # Iterate only through inner blocks to avoid boundary issues
    for i in range(1, orientations.shape[0]-1):
        for j in range(1, orientations.shape[1]-1):
            if np.isnan(orientations[i, j]):
                continue
            poincare = compute_poincare_index(orientations, i, j)
            if poincare == 1:
                x = (j * block_size) + block_size // 2
                y = (i * block_size) + block_size // 2
                core_candidates.append((x, y))
            elif poincare == -1:
                x = (j * block_size) + block_size // 2
                y = (i * block_size) + block_size // 2
                delta_candidates.append((x, y))
    
    print(f"Core candidates found: {len(core_candidates)}")
    print(f"Delta candidates found: {len(delta_candidates)}")

    if debug:
        plt.imshow(thinned_image, cmap='gray')
        plt.scatter([p[0] for p in core_candidates], [p[1] for p in core_candidates], c='red', s=5, label='Cores')
        plt.scatter([p[0] for p in delta_candidates], [p[1] for p in delta_candidates], c='blue', s=5, label='Deltas')
        plt.legend()
        plt.title("Candidate Points")
        plt.show()
    
    # Cluster candidates
    core = None
    if core_candidates:
        core_clusters = cluster_points(core_candidates, eps=cluster_eps)
        if core_clusters:
            largest_cluster = max(core_clusters, key=lambda c: len(c))
            core_x = np.mean([p[0] for p in largest_cluster])
            core_y = np.mean([p[1] for p in largest_cluster])
            core = (int(round(core_x)), int(round(core_y)))
    
    delta = None
    if delta_candidates:
        delta_clusters = cluster_points(delta_candidates, eps=cluster_eps)
        if delta_clusters:
            largest_cluster = max(delta_clusters, key=lambda c: len(c))
            delta_x = np.mean([p[0] for p in largest_cluster])
            delta_y = np.mean([p[1] for p in largest_cluster])
            delta = (int(round(delta_x)), int(round(delta_y)))
    
    return {
        "cores": [core] if core else None,
        "deltas": [delta] if delta else None,
        "terminations": [],
        "bifurcations": []
    }




def improved_detect_minutiae(skeletonized_image, min_distance=8, border_margin=10, 
                            min_ridge_length=5, max_minutiae=50, select_every_other=False):
    """
    Enhanced minutiae detection function that addresses the limitations of the basic CN method.
    
    Args:
        skeletonized_image (numpy.ndarray): Binary image after skeletonization (ridges=255, background=0).
        min_distance (int): Minimum distance between minutiae to filter duplicates/clusters.
        border_margin (int): Margin from image border to ignore (can be different for actual fingerprint region).
        min_ridge_length (int): Minimum ridge length to consider a valid termination.
        max_minutiae (int): Maximum number of minutiae to return per type.
        select_every_other (bool): If True, selects every other minutia to reduce clutter.
        
    Returns:
        dict: Dictionary containing coordinates of different minutiae types.
    """
    if skeletonized_image is None or len(skeletonized_image.shape) != 2:
        print("Invalid input image for improved_detect_minutiae.")
        return {"terminations": [], "bifurcations": [], "cores": [], "deltas": []}
    
    # Make a copy to avoid modifying the original
    image = skeletonized_image.copy()
    
    # Step 1: Ensure proper binary format (ridge=255, background=0) for OpenCV morphology operations
    # Fix: OpenCV morphological operations expect 0 and 255 values, not 0 and 1
    binary = np.where(image > 0, 255, 0).astype(np.uint8)
    
    # For CN calculation, we still need 0 and 1 values
    binary_for_cn = np.where(image > 0, 1, 0).astype(np.int8)
    
    # Step 2: Create a mask for the actual fingerprint region
    # This helps ignore the border areas that don't contain valid fingerprint
    kernel = np.ones((3, 3), np.uint8)
    fingerprint_region = cv2.dilate(binary, kernel, iterations=3)
    fingerprint_region = cv2.erode(fingerprint_region, kernel, iterations=2)  # Less erosion
    
    # Convert region mask to binary for logical operations
    fingerprint_region = np.where(fingerprint_region > 0, 1, 0).astype(np.uint8)
    
    # Apply border margin to the mask
    mask = np.zeros_like(binary_for_cn)
    mask[border_margin:-border_margin, border_margin:-border_margin] = 1
    mask = mask & fingerprint_region
    
    # Add debug check - if mask has no positive pixels, use a simpler approach
    if np.sum(mask) == 0:
        print("Warning: Mask is empty, using border-only mask")
        mask = np.zeros_like(binary_for_cn)
        mask[border_margin:-border_margin, border_margin:-border_margin] = 1
    
    # Step 3: Detect minutiae using crossing number
    terminations = []
    bifurcations = []
    
    rows, cols = binary_for_cn.shape
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            # Only process ridge pixels that are in the valid region
            if binary_for_cn[r, c] == 1 and mask[r, c] == 1:
                # Get 3x3 neighborhood
                block = binary_for_cn[r-1:r+2, c-1:c+2].copy()
                
                # Calculate crossing number with explicit type conversion to avoid overflow
                neighbors = [int(block[0, 0]), int(block[0, 1]), int(block[0, 2]),
                            int(block[1, 2]), int(block[2, 2]), int(block[2, 1]),
                            int(block[2, 0]), int(block[1, 0])]
                
                # Calculate transitions (0->1 or 1->0)
                cn = 0
                for i in range(len(neighbors)):
                    cn += abs(neighbors[i] - neighbors[(i+1) % len(neighbors)])
                cn = cn // 2  # Integer division instead of floating point
                
                # Classify minutiae
                if cn == 1:
                    # This is a potential termination
                    terminations.append((r, c))
                elif cn == 3:
                    # This is a potential bifurcation
                    bifurcations.append((r, c))
    
    # Debug print - check if we found any minutiae before filtering
    print(f"Pre-filtering: {len(terminations)} terminations, {len(bifurcations)} bifurcations")
    
    # Step 4: Filter minutiae based on ridge length (for terminations)
    filtered_terminations = []
    for r, c in terminations:
        # Check if this is a valid termination by examining ridge length
        ridge_length = trace_ridge_length(binary_for_cn, r, c, max_length=min_ridge_length*2)
        if ridge_length >= min_ridge_length:
            filtered_terminations.append((r, c))
    
    # Step 5: Apply distance-based filtering to remove clusters
    filtered_terminations = distance_filter(filtered_terminations, min_distance)
    filtered_bifurcations = distance_filter(bifurcations, min_distance)
    
    # Step 6: Select every other minutia to reduce clutter if requested
    if select_every_other:
        filtered_terminations = select_every_other_minutia(filtered_terminations)
        filtered_bifurcations = select_every_other_minutia(filtered_bifurcations)
        print(f"After select_every_other: {len(filtered_terminations)} terminations, {len(filtered_bifurcations)} bifurcations")
    
    # Debug print - check how many minutiae remain after filtering
    print(f"Post-filtering: {len(filtered_terminations)} terminations, {len(filtered_bifurcations)} bifurcations")
    
    # Step 7: Limit the number of minutiae if necessary
    if max_minutiae > 0:
        filtered_terminations = filtered_terminations[:max_minutiae]
        filtered_bifurcations = filtered_bifurcations[:max_minutiae]
    
    return {
        "terminations": filtered_terminations,
        "bifurcations": filtered_bifurcations,
        "cores": [],  # CN method cannot detect cores
        "deltas": []  # CN method cannot detect deltas
    }

def select_every_other_minutia(minutiae_list):
    """
    Selects every other minutia from the list to reduce clutter.
    
    Args:
        minutiae_list: List of minutiae coordinates
        
    Returns:
        list: Filtered list with every other minutia
    """
    return minutiae_list[::2]  # Select elements with even indices (0, 2, 4, ...)

def trace_ridge_length(binary_image, start_r, start_c, max_length=30):
    """
    Traces the length of a ridge starting from a termination point.
    Used to filter out short ridges that are likely noise.
    
    Args:
        binary_image: Binary image with ridges=1, background=0
        start_r, start_c: Starting coordinates (termination point)
        max_length: Maximum ridge length to trace (to avoid infinite loops)
        
    Returns:
        int: Length of the ridge in pixels
    """
    # Check if the starting point is valid
    if start_r < 0 or start_r >= binary_image.shape[0] or start_c < 0 or start_c >= binary_image.shape[1]:
        return 0
        
    if binary_image[start_r, start_c] != 1:
        return 0
    
    # Create a copy to mark visited pixels
    visited = np.zeros_like(binary_image)
    visited[start_r, start_c] = 1
    
    length = 0
    curr_r, curr_c = start_r, start_c
    
    # Define 8-neighborhood
    neighbors = [(-1, -1), (-1, 0), (-1, 1), 
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    while length < max_length:
        found_next = False
        
        # Look for the next ridge pixel
        for dr, dc in neighbors:
            nr, nc = curr_r + dr, curr_c + dc
            
            # Check boundaries
            if (nr < 0 or nr >= binary_image.shape[0] or 
                nc < 0 or nc >= binary_image.shape[1]):
                continue
            
            # Check if it's an unvisited ridge pixel
            if binary_image[nr, nc] == 1 and visited[nr, nc] == 0:
                visited[nr, nc] = 1
                curr_r, curr_c = nr, nc
                length += 1
                found_next = True
                break
        
        if not found_next:
            break
    
    return length

def distance_filter(minutiae_list, min_distance):
    """
    Filters minutiae that are too close to each other.
    
    Args:
        minutiae_list: List of (r, c) coordinates of minutiae
        min_distance: Minimum Euclidean distance between minutiae
        
    Returns:
        list: Filtered list of minutiae
    """
    if not minutiae_list:
        return []
    
    filtered = [minutiae_list[0]]  # Start with the first minutia
    
    for i in range(1, len(minutiae_list)):
        r1, c1 = minutiae_list[i]
        is_far_enough = True
        
        for r2, c2 in filtered:
            dist = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
            if dist < min_distance:
                is_far_enough = False
                break
        
        if is_far_enough:
            filtered.append((r1, c1))
    
    return filtered

def improved_draw_minutiae(image, minutiae_data, show_labels=True, 
                          termination_color=(0, 0, 255), bifurcation_color=(0, 255, 0),
                          core_color=(255, 0, 0), delta_color=(255, 165, 0),
                          marker_size=5, font_scale=0.6, thickness=2):
    """
    Enhanced version of draw_minutiae with more customization options.
    """
    if image is None:
        return None
    
    # Convert to color if grayscale
    if len(image.shape) == 2:
        display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display_image = image.copy()
    
    # Configuration for each minutiae type
    config = {
        "terminations": {"color": termination_color, "label": "Zakonczenie", "symbol": "o"},
        "bifurcations": {"color": bifurcation_color, "label": "Bifurkacja", "symbol": "x"},
        "cores": {"color": core_color, "label": "Rdzen", "symbol": "●"},
        "deltas": {"color": delta_color, "label": "Delta", "symbol": "▲"}
    }
    
    # Draw each minutiae type
    for m_type, points in minutiae_data.items():
        if m_type in config and points:
            for r, c in points:
                # Draw marker based on type
                if m_type == "bifurcations":
                    # Draw X for bifurcations
                    cv2.line(display_image, (c-marker_size, r-marker_size), 
                             (c+marker_size, r+marker_size), config[m_type]["color"], thickness)
                    cv2.line(display_image, (c-marker_size, r+marker_size), 
                             (c+marker_size, r-marker_size), config[m_type]["color"], thickness)
                else:
                    # Draw circle for other types
                    cv2.circle(display_image, (c, r), marker_size, config[m_type]["color"], thickness)
                
                # Add label if requested
                if show_labels:
                    label = config[m_type]["label"]
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                         font_scale, thickness)
                    
                    # Position text to avoid going out of bounds
                    text_x = c + marker_size + 3
                    text_y = r + marker_size
                    
                    # Adjust if too close to edges
                    if text_x + text_w > display_image.shape[1] - 5:
                        text_x = c - marker_size - 3 - text_w
                    if text_y + text_h > display_image.shape[0] - 5:
                        text_y = r - marker_size - 3
                    
                    cv2.putText(display_image, label, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                               config[m_type]["color"], thickness)
    
    return display_image

def advanced_detect_minutiae(skeletonized_image, orientation_field=None, quality_map=None, 
                            min_distance=8, border_margin=10, min_ridge_length=5, 
                            max_minutiae_per_type=150, morphological_cleaning=True):
    """
    Advanced minutiae detection that implements all improvement recommendations.
    """
    if skeletonized_image is None or len(skeletonized_image.shape) != 2:
        print("Invalid input image for advanced_detect_minutiae.")
        return {"terminations": [], "bifurcations": [], "cores": [], "deltas": []}
    
    # Make a copy and ensure proper format
    image = skeletonized_image.copy()
    
    # Fix: Use 0/255 for OpenCV morphological operations
    binary = np.where(image > 0, 255, 0).astype(np.uint8)
    
    # Create a binary 0/1 image for CN calculations
    binary_for_cn = np.where(image > 0, 1, 0).astype(np.int8)
    
    # Step 1: Apply morphological operations to reduce noise
    if morphological_cleaning:
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        binary = cleaned
        # Update binary_for_cn to match
        binary_for_cn = np.where(binary > 0, 1, 0).astype(np.int8)
    
    # Step 2: Create an intelligent fingerprint region mask
    mask = np.zeros_like(binary_for_cn)
    
    # Apply initial mask using morphological operations
    kernel_big = np.ones((5, 5), np.uint8)
    region_mask = cv2.dilate(binary, kernel_big, iterations=3)
    region_mask = cv2.erode(region_mask, kernel_big, iterations=2)
    
    # Convert to binary mask (0/1)
    region_mask_binary = np.where(region_mask > 0, 1, 0).astype(np.uint8)
    
    # Find largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region_mask_binary, connectivity=8)
    
    if num_labels > 1:
        largest_component_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mask = (labels == largest_component_idx).astype(np.int8)
    else:
        mask = region_mask_binary.astype(np.int8)
    
    # Apply border margin
    border_mask = np.zeros_like(binary_for_cn)
    border_mask[border_margin:-border_margin, border_margin:-border_margin] = 1
    mask = mask & border_mask
    
    # Check if mask is empty
    if np.sum(mask) == 0:
        print("Warning: Advanced mask is empty, using border-only mask")
        mask = border_mask
    
    # Rest of the function remains unchanged
    rows, cols = binary_for_cn.shape
    terminations = []
    bifurcations = []
    
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if binary_for_cn[r, c] == 1 and mask[r, c] == 1:
                block = binary_for_cn[r-1:r+2, c-1:c+2]
                neighbors = [block[0, 0], block[0, 1], block[0, 2],
                             block[1, 2], block[2, 2], block[2, 1],
                             block[2, 0], block[1, 0]]
                cn = 0
                for i in range(len(neighbors)):
                    cn += abs(neighbors[i] - neighbors[(i+1) % len(neighbors)])
                cn = cn // 2
                
                if cn == 1:
                    terminations.append((r, c))
                elif cn == 3:
                    bifurcations.append((r, c))
    
    filtered_terminations = distance_filter(terminations, min_distance)
    filtered_bifurcations = distance_filter(bifurcations, min_distance)
    
    return {
        "terminations": filtered_terminations,
        "bifurcations": filtered_bifurcations,
        "cores": [],
        "deltas": []
    }

