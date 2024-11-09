from google.cloud import vision
import io
from PIL import Image, ImageDraw

def detect_ticket_contours(image_path):
    # Initialiser le client Google Vision
    client = vision.ImageAnnotatorClient()

    # Charger l'image
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Détecter le texte dans l'image
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        print("Aucun texte détecté.")
        return None

    # Le premier élément contient le texte global détecté
    ticket_contour = texts[0].bounding_poly

    # Extraire les coordonnées des sommets du contour
    contour_coordinates = [(vertex.x, vertex.y) for vertex in ticket_contour.vertices]
    print("Coordonnées du contour du ticket :", contour_coordinates)
    
    return contour_coordinates

def draw_contours(image_path, contour_coordinates):
    # Ouvrir l'image
    with Image.open(image_path) as img:
        # Créer un objet de dessin
        draw = ImageDraw.Draw(img)
        
        # Dessiner les contours en utilisant les coordonnées
        draw.line([contour_coordinates[0], contour_coordinates[1], contour_coordinates[2], contour_coordinates[3], contour_coordinates[0]], fill="red", width=3)
        
        # Afficher l'image avec les contours
        img.show()

# Appeler la fonction pour détecter et afficher les contours
        
image_path = 'tests/static/erreur_recente.jpg'

contour_coordinates = detect_ticket_contours(image_path)
if contour_coordinates:
    draw_contours(image_path, contour_coordinates)