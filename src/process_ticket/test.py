import io
import cv2
import numpy as np
from google.cloud import vision
from PIL import Image
import matplotlib.pyplot as plt

def detect_text_and_display_boxes(image_path):
    """Détecte le texte dans une image et affiche l'image avec les boîtes englobantes autour du texte."""
    # Initialisation du client Vision
    client = vision.ImageAnnotatorClient()

    # Lecture de l'image
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    # Conversion de l'image pour l'API Google Vision
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    annotations = response.text_annotations

    if not annotations:
        print("Aucun texte détecté dans l'image.")
        return

    # Charger l'image avec OpenCV
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Dessiner les boîtes englobantes
    for annotation in annotations[1:]:  # annotations[0] contient tout le texte
        vertices = [(vertex.x, vertex.y) for vertex in annotation.bounding_poly.vertices]
        # Boucle pour dessiner les lignes des boîtes englobantes
        for i in range(len(vertices)):
            pt1 = vertices[i]
            pt2 = vertices[(i + 1) % 4]
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)  # couleur verte

    # Afficher l'image avec Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Texte détecté avec boîtes englobantes")
    plt.show()

# Exemple d'utilisation
detect_text_and_display_boxes("tests/static/erreur_recente.jpg")