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


def ocr_ticket_with_clustering_and_columns(image):
    """
    Applique l'OCR sur un ticket et regroupe le texte par ligne,
    en supprimant les zones de bruit en dehors du ticket.

    Args:
        image (PIL.Image.Image): Image d'un ticket.

    Returns:
        list: Liste structurée des colonnes contenant les lignes de texte.
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
    angles = []
    for annotation in annotations[1:]:
        vertices = [(vertex.x, vertex.y) for vertex in annotation.bounding_poly.vertices]
        pt1, pt2, pt3, pt4 = vertices
        # Calculer les angles des côtés horizontaux
        horizontal_angles = []
        for A, B in [(pt1, pt2), (pt4, pt3)]:
            if A[0] != B[0]:  # Éviter les divisions par zéro
                angle = np.arctan((B[1] - A[1]) / (B[0] - A[0]))
                horizontal_angles.append(np.degrees(angle))
        # Calculer l'angle moyen de la boîte englobante
        if horizontal_angles:
            avg_angle = np.mean(horizontal_angles)
            angles.append(avg_angle)
            # Ajouter l'annotation avec l'angle
            text_data.append({
                "description": annotation.description,
                "bounding_poly": vertices,
                "x_center": np.mean([v[0] for v in vertices]),  # Centre horizontal
                "y_center": np.mean([v[1] for v in vertices]),  # Centre vertical
                "angle": avg_angle  # Ajouter l'angle pour filtrage
            })

    # Filtrer les annotations avec un angle proche de zéro
    angle_threshold = 10  # En degrés
    filtered_data = [ann for ann in text_data if abs(ann["angle"]) <= angle_threshold]

    if not filtered_data:
        print("Aucune annotation avec un angle proche de zéro détectée.")
        return []

    #appliquer un clustering par ligne
    y_coords = np.array([ann["y_center"] for ann in filtered_data]).reshape(-1, 1)
    # Calculer la hauteur moyenne des boîtes englobantes
    heights = [abs(ann['bounding_poly'][2][1] - ann['bounding_poly'][0][1]) for ann in filtered_data]
    avg_height = np.mean(heights)

    # Ajuster dynamiquement `eps`
    eps = avg_height * 0.1  # 10% de la hauteur moyenne
    print(f"eps={eps}")
    dbscan = DBSCAN(eps=eps, min_samples=1)
    dbscan.fit(y_coords)

    # Organiser les annotations par cluster
    clusters = {}
    for idx, label in enumerate(dbscan.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filtered_data[idx])
    
    # Trier les clusters par position y moyenne pour obtenir l'ordre des lignes
    sorted_clusters = sorted(clusters.values(), key=lambda cluster: np.mean([ann["y_center"] for ann in cluster]))

    # Construire les colonnes en regroupant les lignes
    lignes = []
    for cluster in sorted_clusters:
        ligne_text = [ann["description"] for ann in sorted(cluster, key=lambda ann: ann["x_center"])]
        lignes.append(ligne_text)

    return lignes

if __name__ == '__main__':
    # Charger une image de ticket
    image_path = 'tests/static/462579997_1099947294855729_767080428195100157_n.jpg'
    image = Image.open(image_path)

    # Appliquer les prétraitements
    image = crop_highest_confidence_box(image)
    image = straighten_using_ocr(image)
    image = preprocess_image_for_ocr(image)

    # Appliquer l'OCR avec clustering
    lignes = ocr_ticket_with_clustering_and_columns(image)

    # Afficher les lignes de texte détectées en mettant un espace entre chaque mot
    for ligne in lignes:
        print(" ".join(ligne))