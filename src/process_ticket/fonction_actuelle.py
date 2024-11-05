import logging
from google.cloud import vision
from API.OpenAICall import get_structured_json_from_text

# Configurer le logger
logging.basicConfig(level=logging.INFO)  # Niveau par défaut, ajustable

def analyse_image(image_content):
    logging.info("Début de l'analyse de l'image.")

    # Initialisez le client Vision
    try:
        client = vision.ImageAnnotatorClient()
        logging.debug("Client Vision initialisé avec succès.")
    except Exception as e:
        logging.error("Erreur lors de l'initialisation du client Vision : %s", e)
        raise

    # Créer un objet Image à partir du contenu de l'image téléchargée
    image = vision.Image(content=image_content)
    logging.debug("Image créée à partir du contenu téléchargé.")

    # Utiliser le client Vision pour détecter du texte dans l'image
    try:
        response = client.text_detection(image=image)
        texts = response.text_annotations
        texte = texts[0].description if texts else ''
        logging.info("Texte détecté dans l'image : %s", texte[:100])  # Afficher les 100 premiers caractères
        logging.debug("Texte complet détecté dans l'image : %s", texte)
    except Exception as e:
        logging.error("Erreur lors de la détection de texte dans l'image : %s", e)
        raise

    # Vérifier si des textes ont été détectés
    if not texte:
        logging.warning("Aucun texte détecté dans l'image.")
        return {}

    # Convertir le texte en JSON structuré
    try:
        response_text = get_structured_json_from_text(texte)
        logging.debug("Texte converti en JSON structuré.")
    except Exception as e:
        logging.error("Erreur lors de la conversion du texte en JSON structuré : %s", e)
        raise

    logging.info("Analyse de l'image terminée avec succès.")
    return response_text