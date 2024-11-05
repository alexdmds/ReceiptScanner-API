import logging
import io
import numpy as np
from google.cloud import vision
from sklearn.cluster import DBSCAN
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_text_from_image(content):
    """Détecte et retourne le texte d'une image au propre, ligne par ligne."""
    try:
        logging.info("Début de la détection de texte dans l'image.")
        
        # Lire les dimensions de l'image avec Pillow
        image = Image.open(io.BytesIO(content))
        width, height = image.size
        logging.debug(f"Dimensions de l'image: largeur={width}, hauteur={height}")

        # Initialisation du client Vision
        client = vision.ImageAnnotatorClient()

        # Conversion du contenu en objet Image pour l'API Google Cloud Vision
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        annotations = response.text_annotations

        # Vérification des annotations
        if not annotations:
            logging.warning("Aucun texte trouvé dans l'image.")
            return ""

        # Extraction des données textuelles
        text_data = []
        for annotation in annotations[1:]:  # On saute le premier élément, qui est le texte complet
            vertices = [(vertex.x, vertex.y) for vertex in annotation.bounding_poly.vertices]
            text_data.append({
                "description": annotation.description,
                "bounding_poly": vertices
            })

        # Pivotage et clustering des annotations
        angle = find_angle(text_data)
        logging.debug(f"Angle de rotation détecté : {angle:.2f} degrés.")
        annotations_pivot = rotate_annotations(text_data, angle, width, height)
        clusters = cluster_annotations(annotations_pivot, eps=15)

        # Combiner les annotations par ligne
        combined_annotations = combine_annotations(clusters)
        final_text = extract_text(combined_annotations)
        
        logging.info("Texte détecté avec succès.")
        return final_text
    except Exception as e:
        logging.error(f"Erreur lors de la détection de texte : {e}")
        return ""

def find_angle(annotations):
    """Calcule l'angle moyen des annotations pour redresser l'image."""
    angle = 0
    nb_ignore = 0
    for annotation in annotations:
        vertices = annotation['bounding_poly']
        pt1, pt2, pt3, pt4 = [pt for pt in vertices]
        if pt1[0] == pt2[0]:
            nb_ignore += 1
        if pt2[0] == pt3[0]:
            nb_ignore += 1
        if pt1[0] != pt2[0] and pt4[0] != pt3[0]:
            angle += np.arctan((pt2[1] - pt1[1]) / (pt2[0] - pt1[0]))
            angle += np.arctan((pt4[1] - pt3[1]) / (pt4[0] - pt3[0]))
    angle /= (2 * len(annotations) - nb_ignore)
    return np.degrees(angle)

def rotate_annotations(annotations, angle, image_width, image_height):
    """Pivote les annotations autour du centre de l'image."""
    x0, y0 = image_width // 2, image_height // 2
    M = np.array([
        [np.cos(np.radians(-angle)), -np.sin(np.radians(-angle)), x0 - x0*np.cos(np.radians(-angle)) + y0*np.sin(np.radians(-angle))],
        [np.sin(np.radians(-angle)), np.cos(np.radians(-angle)), y0 - x0*np.sin(np.radians(-angle)) - y0*np.cos(np.radians(-angle))]
    ])
    for annotation in annotations:
        new_vertices = []
        for vertex in annotation['bounding_poly']:
            point = np.array([vertex[0], vertex[1], 1])
            rotated_point = np.dot(M, point)
            new_vertices.append([int(rotated_point[0]), int(rotated_point[1])])
        annotation['bounding_poly'] = new_vertices
    return annotations

def cluster_annotations(annotations, eps=15):
    """Regroupe les annotations par ligne en utilisant DBSCAN."""
    y_coords = np.array([np.mean([vertex[1] for vertex in ann['bounding_poly']]) for ann in annotations]).reshape(-1, 1)
    dbscan = DBSCAN(eps=eps, min_samples=2)
    dbscan.fit(y_coords)

    clusters = {}
    for idx, label in enumerate(dbscan.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(annotations[idx])

    sorted_clusters = sorted(clusters.values(), key=lambda cluster: np.mean([np.mean([vertex[1] for vertex in ann['bounding_poly']]) for ann in cluster]))
    return sorted_clusters

def combine_annotations(clusters):
    """Combine les annotations de chaque ligne en une seule annotation avec une boîte englobante."""
    combined_annotations = []
    for line in clusters:
        if not line:
            continue
        combined_description = " ".join([annotation['description'] for annotation in line])
        x_min = min(ann['bounding_poly'][0][0] for ann in line)
        y_min = min(ann['bounding_poly'][0][1] for ann in line)
        x_max = max(ann['bounding_poly'][2][0] for ann in line)
        y_max = max(ann['bounding_poly'][2][1] for ann in line)
        
        combined_bounding_poly = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        combined_annotation = {
            "description": combined_description,
            "bounding_poly": combined_bounding_poly
        }
        combined_annotations.append(combined_annotation)
    return combined_annotations

def extract_text(combined_annotations):
    """Extrait et formate le texte détecté."""
    return "\n".join([annotation['description'] for annotation in combined_annotations])

if __name__ == '__main__':
    image_path = '/path/to/your/image.jpg'
    output_file_path = '/path/to/your/output_text.txt'

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    detected_text = get_text_from_image(content)
    logging.info("Texte détecté :\n" + detected_text)

    with open(output_file_path, 'w') as file:
        file.write(detected_text)

    logging.info(f"Texte enregistré dans {output_file_path}")