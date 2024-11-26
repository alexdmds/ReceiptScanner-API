import sys
import os

# Ajouter dynamiquement le dossier parent à sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))  # Chemin vers le dossier parent contenant src
sys.path.append(project_root)

#Ce fichier va contenir la fonction principale appelée dans le main qui prend en entrée une image et qui retourne un json structuré
#Il appelle d'autres utilitaires pour réaliser les fonctions unitaires
from src.API.OpenAICall import get_structured_json_from_text
from src.crop_ticket.apply_crop_ticket import crop_highest_confidence_box
from src.oriente_text.oriente_text import straighten_using_ocr
from src.oriente_text.colorimetrie_ocr import preprocess_image_for_ocr
from src.organize_text.organize_text import ocr_ticket_with_clustering_and_columns

import logging
from PIL import Image
import io

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


def analyse_ticket(image_content):
    """
    Analyse une image de ticket et retourne un JSON structuré.

    Args:
        image_content (bytes): Contenu binaire de l'image du ticket.

    Returns:
        dict: JSON structuré contenant les informations extraites du ticket.
    """
    logger = logging.getLogger("analyse_ticket")

    try:
        logger.info("Début de l'analyse du ticket.")

        # Convertir le contenu binaire en image PIL
        try:
            image = Image.open(io.BytesIO(image_content))
            logger.debug("Image chargée avec succès.")
        except Exception as e:
            logger.error("Erreur lors du chargement de l'image : %s", e)
            raise

        # Étape 1 : Appliquer les prétraitements
        logger.debug("Début du prétraitement de l'image.")
        image = crop_highest_confidence_box(image)
        logger.debug("Étape 1 : Image recadrée avec succès.")
        
        image = straighten_using_ocr(image)
        logger.debug("Étape 2 : Image redressée avec succès.")
        
        image = preprocess_image_for_ocr(image)
        logger.debug("Étape 3 : Image prétraitée pour l'OCR avec succès.")

        # Étape 2 : Appliquer l'OCR avec clustering
        logger.info("Début de l'OCR et du clustering.")
        try:
            lignes = ocr_ticket_with_clustering_and_columns(image)
            logger.debug(f"OCR terminé. {len(lignes)} lignes détectées.")
        except Exception as e:
            logger.error(f"Erreur lors de l'OCR et du clustering : {e}")
            raise

        # Étape 3 : Convertir les lignes en texte brut
        try:
            # lignes est une liste de listes. Il faut les aplatir et les joindre.
            text = "\n".join([" ".join(ligne) for ligne in lignes])
            logger.info("Conversion des lignes en texte brut terminée.")
            logger.debug(f"Texte brut extrait : {text[:200]}...")  # Limiter l'affichage à 200 caractères
        except Exception as e:
            logger.error(f"Erreur lors de la conversion des lignes en texte brut : {e}")
            raise

        # Étape 4 : Convertir le texte en JSON structuré
        logger.info("Début de la conversion du texte en JSON structuré.")
        response_json = get_structured_json_from_text(text)
        logger.info("Conversion du texte en JSON structuré réussie.")

        return response_json

    except Exception as e:
        logger.error("Une erreur s'est produite lors de l'analyse du ticket : %s", e)
        raise  # Propager l'exception pour un traitement éventuel en amont

    finally:
        logger.info("Fin de l'analyse du ticket.")

if __name__ == "__main__":
    try:
        image_path = "tests/static/p0BRZDxUjvJVIfS7RNjz_photo.jpg"  # Remplace par le chemin vers ton image
        with open(image_path, "rb") as f:
            image_content = f.read()
        ticket_info = analyse_ticket(image_content)
        print(ticket_info)
    except Exception as e:
        print(f"Erreur : {e}")