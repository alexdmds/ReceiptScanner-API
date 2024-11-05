from process_ticket.fonction_actuelle import analyse_image
#Ce fichier va contenir la fonction principale appelée dans le main qui prend en entrée une image et qui retourne un json structuré
#Il appelle d'autres utilitaires pour réaliser les fonctions unitaires
from process_ticket.extract_text import get_text_from_image
from API.OpenAICall import get_structured_json_from_text

import logging
import os

def configure_logging():
    # Récupérer l'environnement (local ou production) depuis une variable d'environnement
    env = os.getenv("ENV", "local")

    # Créer un logger de base
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Niveau de log par défaut

    # Définir le format des logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Configurer un handler pour la console
    console_handler = logging.StreamHandler()
    
    if env == "local":
        # En local, afficher tous les logs (niveau DEBUG)
        console_handler.setLevel(logging.DEBUG)
    else:
        # En production (App Engine), afficher seulement les logs de niveau INFO et supérieur
        console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Désactiver les logs DEBUG pour les bibliothèques externes
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)  # Limiter les logs de PIL

# Appeler la fonction de configuration des logs
configure_logging()


def analyse_ticket(image):

    #on va appeler la fonction d'analyse de l'image qui va retourner le texte extrait de l'image
    text = get_text_from_image(image)

    # Convertir le texte en JSON structuré
    try:
        response_json = get_structured_json_from_text(text)
        logging.debug("Texte converti en JSON structuré.")
    except Exception as e:
        logging.error("Erreur lors de la conversion du texte en JSON structuré : %s", e)
        raise

    return response_json