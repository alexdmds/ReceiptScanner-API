import pytest
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ExifTags
from process_ticket.extract_text import get_text_from_image

def test_get_text_from_image():
    """Teste la fonction de détection de texte avec une image locale et affiche l'image avec le texte extrait."""
    # Chemin de l'image de test
    image_path = Path("tests/static/ticket_test_image.png")
    
    # Vérification que l'image existe avant de continuer
    if not image_path.exists():
        pytest.fail(f"L'image de test n'existe pas : {image_path}")

    # Lecture du fichier image
    with image_path.open("rb") as image_file:
        content = image_file.read()

    # Exécution de la fonction pour obtenir le texte extrait
    detected_text = get_text_from_image(content)
    
    # Afficher l'image et le texte extrait
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # Charger l'image avec PIL et corriger l'orientation si nécessaire
    ax0 = plt.subplot(gs[0])
    image = Image.open(image_path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass  # Pas de correction si les métadonnées EXIF sont absentes

    # Afficher l'image
    ax0.imshow(image)
    ax0.axis('off')
    ax0.set_title("Ticket Image")

    # Afficher le texte extrait
    ax1 = plt.subplot(gs[1])
    ax1.text(0.05, 0.5, detected_text, fontsize=10, va='center', ha='left', wrap=True)
    ax1.axis('off')
    ax1.set_title("Texte Extrait")

    plt.tight_layout()
    plt.show()

    # Assertion pour le test
    assert detected_text, "Aucun texte détecté dans l'image."