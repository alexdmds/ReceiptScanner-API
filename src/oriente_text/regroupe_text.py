import sys
import os

# Ajouter dynamiquement le dossier parent à sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))  # Chemin vers le dossier parent contenant src
sys.path.append(project_root)


from google.cloud import vision
import io
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image
from src.crop_ticket.apply_crop_ticket import crop_highest_confidence_box
from src.oriente_text.oriente_text import straighten_using_ocr
from src.oriente_text.colorimetrie_ocr import preprocess_image_for_ocr

def ocr_ticket_with_clustering(image):
    """
    Applique l'OCR sur un ticket droit et retourne le texte ligne par ligne.

    Args:
        image (PIL.Image.Image): Image d'un ticket droit.

    Returns:
        list: Liste de chaînes représentant chaque ligne de texte détectée.
    """
    # Convertir l'image PIL en bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    # Initialiser le client Vision API
    client = vision.ImageAnnotatorClient()

    # Envoyer l'image à l'API
    image_content = image_bytes.read()
    vision_image = vision.Image(content=image_content)
    response = client.text_detection(image=vision_image)

    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")

    # Extraire les annotations
    annotations = response.text_annotations
    if not annotations:
        print("Aucun texte détecté.")
        return []  # Retourner une liste vide si aucun texte n'est détecté

    # Préparer les annotations (sauter le premier élément qui contient tout le texte)
    text_data = []
    for annotation in annotations[1:]:
        vertices = [(vertex.x, vertex.y) for vertex in annotation.bounding_poly.vertices]
        text_data.append({
            "description": annotation.description,
            "bounding_poly": vertices
        })

    # Extraire les coordonnées y pour chaque annotation
    y_coords = np.array([np.mean([vertex[1] for vertex in ann['bounding_poly']]) for ann in text_data]).reshape(-1, 1)

    # Appliquer le clustering DBSCAN pour regrouper les annotations par ligne
    eps = 20  # Rayon de voisinage (ajustable selon l'espacement des lignes)
    dbscan = DBSCAN(eps=eps, min_samples=1)
    dbscan.fit(y_coords)

    # Organiser les annotations par cluster
    clusters = {}
    for idx, label in enumerate(dbscan.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(text_data[idx])

    # Trier les clusters par position y moyenne pour obtenir l'ordre des lignes
    sorted_clusters = sorted(clusters.values(), key=lambda cluster: np.mean([np.mean([vertex[1] for vertex in ann['bounding_poly']]) for ann in cluster]))

    # Construire le texte ligne par ligne
    lines = []
    for cluster in sorted_clusters:
        line_text = " ".join([ann['description'] for ann in sorted(cluster, key=lambda ann: ann['bounding_poly'][0][0])])
        lines.append(line_text)

    return lines

if __name__ == '__main__':
    # Charger une image de ticket
    image_path = 'tests/static/462562326_1394586214848752_426626803125374696_n.jpg'

    image = Image.open(image_path)

    #crop ticket
    image = crop_highest_confidence_box(image)

    # Redresser l'image
    image = straighten_using_ocr(image)

    #crop ticket
    image = crop_highest_confidence_box(image)

    #colorimetrie
    image = preprocess_image_for_ocr(image)

    #afficher l'image
    image.show()

    # Appliquer la fonction OCR avec clustering
    lines = ocr_ticket_with_clustering(image)

    # Afficher les lignes de texte détectées
    for line in lines:
        print(line)