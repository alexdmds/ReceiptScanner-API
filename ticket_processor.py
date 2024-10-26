from google.cloud import vision
from AnalyseText import get_structured_json_from_text

def process_ticket_image(image_content):
    # Initialiser le client Vision
    client = vision.ImageAnnotatorClient()

    # Créer un objet Image à partir du contenu de l'image
    image = vision.Image(content=image_content)

    # Utiliser le client Vision pour détecter le texte dans l'image
    response = client.text_detection(image=image)
    texts = response.text_annotations
    texte = texts[0].description if texts else ''

    # Convertir le texte en JSON structuré
    response_text = get_structured_json_from_text(texte)
    return response_text