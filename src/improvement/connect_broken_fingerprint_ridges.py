import cv2
import numpy as np


def connect_broken_fingerprint_ridges(thinned_image):
    """
    Modyfikuje obraz na podstawie sąsiedztwa pikseli.
    - Biały piksel (linia = 255) w okolicy czarnego (tło = 0) zmienia się na czarny.
    - Czarny piksel (tło = 0) w okolicy białego (linia = 255) zmienia się na biały.
    Operacje bazują na oryginalnym stanie pikseli i ich sąsiadów.

    Uwaga: Ta logika prawdopodobnie nie będzie skutecznie łączyć przerwanych linii
    papilarnych w tradycyjnym sensie. Może prowadzić do erozji linii i dylatacji tła
    w specyficzny sposób.

    Args:
        thinned_image (numpy.ndarray): Obraz binarny po ścienianiu (linie=255, tło=0).

    Returns:
        numpy.ndarray: Zmodyfikowany obraz.
    """
    if thinned_image is None or thinned_image.dtype != np.uint8 or len(thinned_image.shape) != 2:
        print("Nieprawidłowy obraz wejściowy dla connect_broken_fingerprint_ridges.")
        return thinned_image

    rows, cols = thinned_image.shape
    output_image = thinned_image.copy() # Pracujemy na kopii, aby decyzje były oparte na oryginale

    # Definiujemy standardowy kernel 3x3 do sprawdzania sąsiedztwa
    # Wbudowane funkcje erode/dilate efektywnie robią to, co potrzebujemy
    # do określenia, czy piksel ma sąsiada innego koloru.
    kernel = np.ones((2, 2), np.uint8)

    # 1. Identyfikacja białych pikseli (linii), które mają czarnego sąsiada:
    #    Te piksele staną się czarne. To jest efekt erozji na zbiorze białych pikseli.
    eroded_image = cv2.erode(thinned_image, kernel, iterations=1)

    # 2. Identyfikacja czarnych pikseli (tła), które mają białego sąsiada:
    #    Te piksele staną się białe. To jest efekt dylatacji na zbiorze białych pikseli
    #    (lub erozji na zbiorze czarnych pikseli, jeśli patrzeć od strony tła).
    dilated_image = cv2.dilate(thinned_image, kernel, iterations=1)

    # Zastosowanie logiki:
    # - Jeśli oryginalny piksel był biały (255): użyj wartości z obrazu po erozji.
    #   (Jeśli miał czarnego sąsiada, erozja zmieniła go na 0, inaczej pozostał 255).
    # - Jeśli oryginalny piksel był czarny (0): użyj wartości z obrazu po dylatacji.
    #   (Jeśli miał białego sąsiada, dylatacja zmieniła go na 255, inaczej pozostał 0).

    output_image = np.where(thinned_image == 255, eroded_image, dilated_image)
    
    # Alternatywna, bardziej bezpośrednia implementacja pętlowa (wolniejsza):
    # for r in range(1, rows - 1):
    #     for c in range(1, cols - 1):
    #         original_pixel_is_white = (thinned_image[r, c] == 255)
    #         has_black_neighbor = False
    #         has_white_neighbor = False

    #         for i in range(-1, 2):
    #             for j in range(-1, 2):
    #                 if i == 0 and j == 0:
    #                     continue
    #                 if thinned_image[r + i, c + j] == 0:
    #                     has_black_neighbor = True
    #                 elif thinned_image[r + i, c + j] == 255:
    #                     has_white_neighbor = True
            
    #         if original_pixel_is_white:
    #             if has_black_neighbor:
    #                 output_image[r, c] = 0  # Biały -> Czarny
    #             # else: pozostaje biały (już jest w output_image)
    #         else: # Oryginalny piksel jest czarny
    #             if has_white_neighbor:
    #                 output_image[r, c] = 255 # Czarny -> Biały
    #             # else: pozostaje czarny (już jest w output_image)
                
    return output_image