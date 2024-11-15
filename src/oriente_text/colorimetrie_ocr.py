import sys
import os

# Ajouter dynamiquement le dossier parent à sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))  # Chemin vers le dossier parent contenant src
sys.path.append(project_root)

import cv2
import numpy as np
from PIL import Image
from src.crop_ticket.apply_crop_ticket import crop_highest_confidence_box
from PIL import ImageEnhance
from src.oriente_text.oriente_text import straighten_using_ocr

def preprocess_image_for_ocr(image):
    """
    Prépare une image de ticket pour l'OCR en améliorant la visibilité du texte
    avec des traitements spécialisés pour éviter le flou sur les zones de texte.

    Args:
        image (PIL.Image.Image): Image d'entrée à prétraiter.

    Returns:
        PIL.Image.Image: Image prétraitée pour l'OCR.
    """
    # Convertir l'image PIL en format OpenCV (BGR)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Appliquer la réduction de bruit de manière légère
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Utiliser un seuil adaptatif pour isoler les zones de texte
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Détecter les contours pour isoler les zones de texte
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer un masque pour les zones de texte
    mask = np.zeros_like(denoised)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Appliquer CLAHE pour améliorer le contraste uniquement sur les zones de texte
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    equalized = clahe.apply(denoised)

    # Mélanger l'image égalisée uniquement sur les zones de texte
    combined = cv2.bitwise_and(equalized, mask) + cv2.bitwise_and(denoised, cv2.bitwise_not(mask))

    # Filtre de netteté appliqué uniquement sur les zones de texte
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(combined, -1, sharpen_kernel)

    # Convertir l'image OpenCV traitée en format PIL
    processed_image = Image.fromarray(sharpened)

    # Ajuster le contraste global pour l'image entière
    enhancer = ImageEnhance.Contrast(processed_image)
    processed_image = enhancer.enhance(1.8)

    return processed_image

if __name__ == "__main__":
    # Charger l'image d'exemple
    image_path = "tests/static/erreur_recente2.jpg"
    image = Image.open(image_path)

    image = crop_highest_confidence_box(image)

    image = straighten_using_ocr(image)

    image = crop_highest_confidence_box(image)

    # Prétraiter l'image pour l'OCR
    processed_image = preprocess_image_for_ocr(image)

    # Afficher l'image prétraitée
    processed_image.show()