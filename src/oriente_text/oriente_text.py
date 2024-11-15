from google.cloud import vision
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import io
from PIL.ExifTags import TAGS as ExifTags

def straighten_using_ocr(image):
    """
    Redresse une image en fonction de l'inclinaison moyenne du texte détecté via Google Cloud Vision API.

    Args:
        image (PIL.Image.Image): Image à redresser.

    Returns:
        PIL.Image.Image: Image redressée.
    """

    # Correction de l'orientation selon les métadonnées EXIF
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None and orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Pas d'information EXIF pour l'orientation, continuer sans correction
        pass

    #pivoter l'image de 90° dans le sens horaire
    image = image.rotate(-90, expand=True)
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
        return image  # Retourner l'image d'origine si aucun texte n'est détecté

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

    # Dessiner les boîtes englobantes sur l'image
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        vertices = annotation['bounding_poly']
        draw.polygon(vertices, outline="red", width=2)

    # Calculer l'angle moyen des boîtes englobantes
    angle = 0
    nb_angles = 0
    for annotation in annotations:
        vertices = annotation['bounding_poly']
        pt1, pt2, pt3, pt4 = [pt for pt in vertices]
        for A, B in [(pt1, pt2), (pt4, pt3)]:
            if A[0] > B[0]:
                angle += np.pi + np.arctan((A[1] - B[1]) / (A[0] - B[0]))
                nb_angles += 1
            if A[0] < B[0]:
                angle += np.arctan((A[1] - B[1]) / (A[0] - B[0]))
                nb_angles += 1
                

    angle /= -(nb_angles or 1)
    angle = np.degrees(angle)
    print(f"Angle moyen détecté : {angle:.2f}°")

    # Appliquer la rotation pour redresser l'image
    rotated_image = image.rotate(-angle, expand=True, fillcolor=(255, 255, 255))

    return rotated_image