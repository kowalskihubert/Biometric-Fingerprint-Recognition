import numpy as np
import cv2

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

def detect_minutiae(thinned_image, border_margin=10, select_one_of_each_type=True):
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
        
    return {
        "terminations": selected_terminations,
        "bifurcations": selected_bifurcations,
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
        "terminations": {"color": (0, 0, 255), "label": "Zakończenie"}, 
        "bifurcations": {"color": (0, 255, 0), "label": "Bifurkacja"}, 
        "cores": {"color": (255, 0, 0), "label": "Rdzeń"},        
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