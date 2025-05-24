import cv2
import numpy as np

# def morphological_skeletonization(image):
#     """
#     Wykonuje szkieletyzację morfologiczną obrazu binarnego.

#     Bazując na formule z wykładu:
#     S_k(X) = (X erodowane przez kB) - ((X erodowane przez kB) otwarte przez B)
#     Szkielet(X) = Suma S_k(X) dla k=0...K
#     gdzie K = max{k | X erodowane przez kB != zbiór pusty}

#     Argumenty:
#         image (numpy.ndarray): Obraz wejściowy. Powinien być obrazem binarnym
#                                (0 dla tła, 255 dla obiektu). Jeśli jest
#                                w skali szarości lub kolorowy, zostanie
#                                przekonwertowany na binarny.

#     Zwraca:
#         numpy.ndarray: Obraz szkieletu.
#     """
#     # 1. Upewnij się, że obraz jest binarny (0 i 255)
#     if len(image.shape) == 3: # Obraz kolorowy
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else: # Obraz w skali szarości lub już binarny
#         gray_image = image.copy()

#     # Binaryzacja (jeśli nie jest już binarny)
#     # Wartość progowa 127 jest typowa, ale można dostosować
#     _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

#     # 2. Zdefiniuj element strukturalny (B)
#     # Wykład nie precyzuje, ale typowo używa się krzyża 3x3 lub kwadratu 3x3.
#     # Użyjmy krzyża, jest często stosowany w szkieletyzacji.
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     # Alternatywnie kwadrat:
#     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

#     # 3. Inicjalizacja obrazu szkieletu
#     skeleton = np.zeros(binary_image.shape, np.uint8)

#     # 4. Pętla iteracyjna do obliczania S_k(X)
#     # `eroded_image` będzie reprezentować (X ⊖ kB)
#     eroded_image = binary_image.copy()
    
#     iteration = 0
#     while True:
#         # `temp_eroded` to (X ⊖ kB) dla bieżącego k (lub bieżącej iteracji)
#         temp_eroded = eroded_image.copy()

#         # Sprawdź warunek końca: X ⊖ kB == ∅
#         # Jeśli obraz po erozji jest całkowicie czarny (pusty), kończymy.
#         if cv2.countNonZero(temp_eroded) == 0:
#             print(f"Zakończono po {iteration} iteracjach.")
#             break
        
#         # Oblicz otwarcie (X ⊖ kB) ○ B
#         opened_version = cv2.morphologyEx(temp_eroded, cv2.MORPH_OPEN, kernel)
        
#         # Oblicz S_k(X) = (X ⊖ kB) \ ((X ⊖ kB) ○ B)
#         # Różnica zbiorów to odjęcie obrazów
#         sk_element = cv2.subtract(temp_eroded, opened_version)
        
#         # Dodaj S_k(X) do szkieletu (suma/unia zbiorów)
#         skeleton = cv2.bitwise_or(skeleton, sk_element)
        
#         # Przygotuj (X ⊖ (k+1)B) na następną iterację
#         # Eroduj `eroded_image` (które było X ⊖ kB) jeszcze raz elementem B
#         eroded_image = cv2.erode(eroded_image, kernel)
#         iteration += 1
        
#         # Zabezpieczenie przed nieskończoną pętlą (choć nie powinno wystąpić)
#         if iteration > 1000: # Przykładowy limit
#             print("Osiągnięto limit iteracji.")
#             break
            
#     return skeleton
def morphological_skeletonization(image, threshold_value=127, kernel_type='cross'):
    """
    Wykonuje szkieletyzację morfologiczną obrazu.
    Zwraca obraz, gdzie szkielet (linie) jest biały (255), a tło czarne (0).
    """
    processed_image = image.copy()
    if len(processed_image.shape) == 3:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    if processed_image.dtype != np.uint8:
        if processed_image.dtype in [np.float32, np.float64] and \
           np.max(processed_image) <= 1.0 and np.min(processed_image) >=0.0:
            processed_image = (processed_image * 255).astype(np.uint8)
        else:
            processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Binaryzacja: obiekt=255 (biały), tło=0 (czarny)
    _, binary_image = cv2.threshold(processed_image, threshold_value, 255, cv2.THRESH_BINARY)

    if kernel_type == 'square':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    elif kernel_type == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    else:
        print(f"Nieprawidłowy typ kernela '{kernel_type}'. Używam 'cross'.")
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    skeleton = np.zeros(binary_image.shape, np.uint8)
    eroded_image = binary_image.copy()
    iteration = 0
    max_iterations = max(binary_image.shape) 

    while True:
        if cv2.countNonZero(eroded_image) == 0:
            break
        
        opened_version = cv2.morphologyEx(eroded_image, cv2.MORPH_OPEN, kernel)
        sk_element = cv2.subtract(eroded_image, opened_version)
        skeleton = cv2.bitwise_or(skeleton, sk_element)
        
        pixels_before_erosion = cv2.countNonZero(eroded_image)
        eroded_image = cv2.erode(eroded_image, kernel)
        pixels_after_erosion = cv2.countNonZero(eroded_image)
        
        iteration += 1
        if (pixels_after_erosion == pixels_before_erosion and pixels_after_erosion > 0) or iteration >= max_iterations:
            break
            
    return skeleton

