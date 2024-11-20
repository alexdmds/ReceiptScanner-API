from google.cloud import vision
import numpy as np
from PIL import Image
import io
from PIL.ExifTags import TAGS as ExifTags
from sklearn.cluster import DBSCAN

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

    # Repérer tous les angles de rotation
    angle = []
    for annotation in annotations:
        vertices = annotation['bounding_poly']
        pt1, pt2, pt3, pt4 = [pt for pt in vertices]
        for A, B in [(pt1, pt2), (pt4, pt3)]:
            if A[0] > B[0]:
                angle.append(np.pi + np.arctan((A[1] - B[1]) / (A[0] - B[0])))
            if A[0] < B[0]:
                angle.append(np.arctan((A[1] - B[1]) / (A[0] - B[0])))
    

    #appliquer DBSCAN sur les angles pour détecter le cluster principal

    angle = np.array(angle).reshape(-1, 1)
    dbscan = DBSCAN(eps=0.1, min_samples=1)
    dbscan.fit(angle)
    cluster_id = np.argmax(np.bincount(dbscan.labels_))
    angle = np.mean(angle[dbscan.labels_ == cluster_id]) * 180 / np.pi


    print(f"Angle moyen du cluster pirncipal détecté : {angle:.2f}°")

    # Détecter le mode de l'image et ajuster la couleur de remplissage
    fillcolor = 255 if image.mode == "L" else (255, 255, 255)

    # Appliquer la rotation pour redresser l'image
    rotated_image = image.rotate(angle, expand=True, fillcolor=fillcolor)

    return rotated_image

if __name__ == '__main__':
    # Charger une image de ticket
    image_path = 'tests/static/erreur_recente.jpg'

    image = Image.open(image_path)
    rotated_image = straighten_using_ocr(image)
    rotated_image.show()