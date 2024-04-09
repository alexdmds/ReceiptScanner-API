from google.cloud import vision
import io
import cv2
import numpy as np
import json

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

def find_angle(annotations, img):

    # Obtenir les dimensions de l'image
    dimensions = img.shape
    print("Image dimensions: ", dimensions)
    ymax = img.shape[1]

    # Calculer l'angle de rotation de l'image
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

    angle /= (2*len(annotations)-nb_ignore)
    angle = np.degrees(angle)

    return angle

def pivote_image(img, angle):
    dimensions = img.shape
    ymax = img.shape[1]
    center = (dimensions[1] // 2, dimensions[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (dimensions[1], dimensions[0]))
    return rotated_img

    with open(json_path, 'r') as json_file:
        annotations = json.load(json_file)
    
    # Charger l'image avec OpenCV
    img = cv2.imread(image_path)

    # Obtenir les dimensions de l'image
    dimensions = img.shape
    print("Image dimensions: ", dimensions)
    ymax = img.shape[1]

    # Calculer l'angle de rotation de l'image
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

    angle /= (2*len(annotations)-nb_ignore)
    angle = np.degrees(angle)

    return angle

def pivote_annotations(annotations, img, angle):
    height, width = img.shape[:2]
    x0, y0 = width // 2, height // 2

    # Calcul de la matrice de rotation autour du centre de l'image
    M = cv2.getRotationMatrix2D((x0, y0), angle, 1)
    
    for annotation in annotations:
        new_vertices = []
        for vertex in annotation['bounding_poly']:
            # Appliquer la matrice de rotation à chaque point
            point = np.array([vertex[0], vertex[1], 1])
            rotated_point = M.dot(point)
            new_vertices.append([int(rotated_point[0]), int(rotated_point[1])])
        
        annotation['bounding_poly'] = new_vertices

    return annotations

def get_annotations(json_path):
    with open(json_path, 'r') as json_file:
        annotations = json.load(json_file)
    
    return annotations

def group_annotations_by_lines(annotations, vertical_threshold):
    # Trier les annotations de haut en bas en fonction de la coordonnée y du point haut-gauche
    annotations.sort(key=lambda a: a['bounding_poly'][0][1])
    
    # Initialiser la première ligne et la liste des lignes
    lines = []
    current_line = [annotations[0]]
    
    # Parcourir les annotations pour les regrouper par ligne
    for annotation in annotations[1:]:
        current_y = annotation['bounding_poly'][0][1]
        last_y = current_line[-1]['bounding_poly'][0][1]
        
        # Si l'annotation est verticalement proche de la ligne actuelle, l'ajouter à la ligne
        if abs(current_y - last_y) < vertical_threshold:
            current_line.append(annotation)
        else:
            # Sinon, commencer une nouvelle ligne
            lines.append(current_line)
            current_line = [annotation]
    
    # Ajouter la dernière ligne si elle contient des annotations
    if current_line:
        lines.append(current_line)

    # Trier chaque ligne horizontalement
    for line in lines:
        line.sort(key=lambda a: a['bounding_poly'][0][0])
    
    return lines

def filter_noise_from_lines(lines, horizontal_threshold):
    filtered_lines = []
    for line in lines:
        # Supposer que la première boîte est la référence de départ pour la ligne
        reference_x = line[0]['bounding_poly'][0][0]
        
        # Filtrer les annotations trop éloignées horizontalement de la première boîte
        filtered_line = [box for box in line if abs(box['bounding_poly'][0][0] - reference_x) < horizontal_threshold]
        
        if filtered_line:
            filtered_lines.append(filtered_line)
    
    return filtered_lines


annotations = get_annotations(output_file_path_tournee)
img = get_image_from_file(path_image_tournee)

angle = find_angle(annotations, img)
img_pivot = pivote_image(img, angle)
annotations_pivot = pivote_annotations(annotations, img, angle)

draw_boxes_from_annotations(img_pivot, annotations_pivot)

# Exemple d'utilisation
vertical_threshold = 10  # Ajustez en fonction de la hauteur des caractères et de l'espacement des lignes
horizontal_threshold = 3000  # Ajustez en fonction de la largeur attendue du document

lines = group_annotations_by_lines(annotations_pivot, vertical_threshold)
#filtered_lines = filter_noise_from_lines(lines, horizontal_threshold)

for line in lines:
    print("Ligne:", [(box['description'], box['bounding_poly']) for box in line])
