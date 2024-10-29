from google.cloud import vision
from utils.OpenAICall import get_structured_json_from_text

def analyse_image(image_content):
    # Initialisez le client Vision
    client = vision.ImageAnnotatorClient()

    # Créer un objet Image à partir du contenu de l'image téléchargée
    image = vision.Image(content=image_content)

    # Utiliser le client Vision pour détecter du texte dans l'image
    response = client.text_detection(image=image)
    texts = response.text_annotations
    texte = texts[0].description if texts else ''

    # Convertir le texte en JSON structuré
    response_text = get_structured_json_from_text(texte)

    return response_text