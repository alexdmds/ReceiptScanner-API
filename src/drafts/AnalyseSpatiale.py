from google.cloud import vision
import io
import numpy as np
import json
from sklearn.cluster import DBSCAN
from PIL import Image

path_image_test = '/Users/alexisdemonts/Documents/GitHub/c_troop_app_gae/2e457f8c-7ce5-41b4-9611-b9cda354379b6090562080649726705.jpg'
output_file_path = '/Users/alexisdemonts/Documents/GitHub/c_troop_app_gae/ticket_test.json'

path_image_tournee = '/Users/alexisdemonts/Documents/GitHub/c_troop_app_gae/ticket_tourne.jpg'
output_file_path_tournee = '/Users/alexisdemonts/Documents/GitHub/c_troop_app_gae/ticket_tourne.json'

path_image_3 = '/Users/alexisdemonts/Documents/GitHub/c_troop_app_gae/images.jpeg'
output_file_path_3 = '/Users/alexisdemonts/Documents/GitHub/c_troop_app_gae/ticket_3.json'

def get_image_from_file(image_path):
    """Charge une image à partir d'un fichier."""
    return cv2.imread(image_path)

def draw_boxes_from_annotations(img, annotations):
    """Dessine des boîtes englobantes autour du texte à partir des données JSON et affiche l'image."""

    # Obtenir les dimensions de l'image
    dimensions = img.shape
    print("Image dimensions: ", dimensions)
    ymax = img.shape[1]

    # Dessiner les boîtes englobantes pour chaque bloc de texte à partir des données JSON
    for annotation in annotations:
        vertices = annotation['bounding_poly']
        # Assurez-vous que les sommets sont dans l'ordre attendu : haut-gauche, haut-droit, bas-droit, bas-gauche
        pt1, pt2, pt3, pt4 = [pt for pt in vertices]
        cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 0), 3)
        cv2.line(img, tuple(pt2), tuple(pt3), (0, 255, 0), 3)
        cv2.line(img, tuple(pt3), tuple(pt4), (0, 255, 0), 3)
        cv2.line(img, tuple(pt4), tuple(pt1), (0, 255, 0), 3)

    # Afficher l'image avec des boîtes englobantes
    cv2.imshow('Text Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_detected_text_to_file(image_path, output_file_path):
    """Détecte le texte dans l'image de ticket de caisse et enregistre les résultats dans un fichier."""
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    annotations = response.text_annotations

    # Préparer les données pour l'enregistrement
    text_data = []
    if not annotations:
        print('Aucun texte trouvé.')
    else:
        for annotation in annotations[1:]:  # On saute le premier élément
            vertices = [(vertex.x, vertex.y) for vertex in annotation.bounding_poly.vertices]
            text_data.append({
                "description": annotation.description,
                "bounding_poly": vertices
            })

    # Enregistrer les données dans un fichier JSON
    with open(output_file_path, 'w') as json_file:
        json.dump(text_data, json_file, indent=2)

    print(f'Données enregistrées dans {output_file_path}')

def find_angle(annotations):
    # Utilisation de image_width et image_height au lieu de charger l'image
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
    angle = np.degrees(angle)

    return angle

def pivote_image(img, angle):
    dimensions = img.shape
    ymax = img.shape[1]
    center = (dimensions[1] // 2, dimensions[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (dimensions[1], dimensions[0]))
    return rotated_img

def pivote_annotations(annotations, angle, image_width, image_height):
    x0, y0 = image_width // 2, image_height // 2

    # Calcul de la matrice de rotation autour du centre de l'image sans utiliser cv2
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

def get_annotations(json_path):
    with open(json_path, 'r') as json_file:
        annotations = json.load(json_file)
    
    return annotations


    # Extraire les hauteurs de chaque annotation (la différence en y entre le point haut et le point bas)
    heights = [abs(vert[3][1] - vert[0][1]) for annot in annotations for vert in [annot['bounding_poly']]]
    
    # Calculer la médiane des hauteurs, vous pourriez également choisir d'utiliser la moyenne
    median_height = np.median(heights)
    
    # Utiliser un facteur de l'espacement attendu entre les lignes basé sur la médiane des hauteurs
    # Ce facteur peut être ajusté en fonction de la taille réelle des espaces entre les lignes sur vos tickets
    line_spacing_factor = 1.5
    
    vertical_threshold = median_height * line_spacing_factor
    return vertical_threshold

    """
    Filtrer les annotations inutiles basées sur la taille et le contenu du texte.

    :param annotations: La liste des annotations à filtrer.
    :param size_threshold: Le seuil minimum pour la largeur et la hauteur des annotations.
    :param valid_chars: Une chaîne contenant des caractères valides attendus dans les annotations.
    :return: Une liste d'annotations filtrées.
    """
    filtered_annotations = []

    for annotation in annotations:
        # Calculer la largeur et la hauteur de l'annotation
        width = abs(annotation['bounding_poly'][1][0] - annotation['bounding_poly'][0][0])
        height = abs(annotation['bounding_poly'][2][1] - annotation['bounding_poly'][0][1])

        # Vérifier si l'annotation passe le filtre de taille
        if width > size_threshold and height > size_threshold:
            # Vérifier si le texte de l'annotation contient uniquement des caractères valides
            if all(char in valid_chars for char in annotation['description']):
                filtered_annotations.append(annotation)

    return filtered_annotations

def combine_annotations(lines):
    """
    Combine les annotations de chaque ligne pour obtenir une liste d'annotations concaténées avec une boîte englobante.
    
    :param annotations: La liste des lignes à combiner.
    :return: Une liste d'annotations combinées.
    """

    annotations_combinees = []

    for line in lines:


        if not line:
            return None
        
        # Concaténation des descriptions
        combined_description = " ".join([annotation['description'] for annotation in line])

        # Calcul des points de la boîte englobante
        x_min = min(ann['bounding_poly'][0][0] for ann in line)
        y_min = min(ann['bounding_poly'][0][1] for ann in line)
        x_max = max(ann['bounding_poly'][2][0] for ann in line)
        y_max = max(ann['bounding_poly'][2][1] for ann in line)
        
        combined_bounding_poly = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]
        
        combined_annotation = {
            "description": combined_description,
            "bounding_poly": combined_bounding_poly
        }

        annotations_combinees.append(combined_annotation)
    
    return annotations_combinees

def cluster_annotations_by_dbscan(annotations, eps):
    # Extraire la coordonnée y du centre de chaque annotation pour le clustering
    y_coords = np.array([np.mean([vertex[1] for vertex in ann['bounding_poly']]) for ann in annotations]).reshape(-1, 1)
    
    # Appliquer DBSCAN pour regrouper les coordonnées y
    dbscan = DBSCAN(eps, min_samples=2)  # eps est le rayon de voisinage et min_samples le nombre de points minimum pour former un cluster
    dbscan.fit(y_coords)

    clusters = {}
    for idx, label in enumerate(dbscan.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(annotations[idx])

    # Convertir le dictionnaire de clusters en liste pour plus de commodité
    cluster_list = list(clusters.values())

    # Trier les clusters par la position y moyenne pour ordonner les lignes de haut en bas
    sorted_cluster_list = sorted(cluster_list, key=lambda cluster: np.mean([np.mean([vertex[1] for vertex in ann['bounding_poly']]) for ann in cluster]))

    return sorted_cluster_list

def extract_text(annotations):
    text = ""
    for annotation in annotations:
        text += annotation['description'] + "\n"
    return text

def get_text_from_image(content):

    # Utiliser Pillow pour lire les dimensions de l'image
    image = Image.open(io.BytesIO(content))
    width, height = image.size
    # Initialisation du client Vision

    client = vision.ImageAnnotatorClient()

    # Conversion du contenu en objet Image de Google Cloud Vision
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    annotations = response.text_annotations

    # Préparer les données
    text_data = []
    if not annotations:
        print('Aucun texte trouvé.')
    else:
        for annotation in annotations[1:]:  # On saute le premier élément
            vertices = [(vertex.x, vertex.y) for vertex in annotation.bounding_poly.vertices]
            text_data.append({
                "description": annotation.description,
                "bounding_poly": vertices
            })

    annotations = text_data
    angle = find_angle(annotations)
    annotations_pivot = pivote_annotations(annotations, angle, width, height)
    eps = 15
    cluster = cluster_annotations_by_dbscan(annotations_pivot, eps)
    cluster_annotations = combine_annotations(cluster)
    return extract_text(cluster_annotations)


if __name__ == '__main__':
    image_path = '/Users/alexisdemonts/Documents/GitHub/c_troop_app_gae/photos_tickets/435187413_1375929267132715_7729131294039733013_n.jpg'
    
    #print(get_text_from_image(image_path))

    output_file_path = '/Users/alexisdemonts/Documents/GitHub/c_troop_app_gae/extract_json/435187413_1375929267132715_7729131294039733013_n.json'

    save_detected_text_to_file(image_path, output_file_path)

