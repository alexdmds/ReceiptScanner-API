import sys
import os

# Ajouter dynamiquement le dossier parent à sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))  # Chemin vers le dossier parent contenant src
sys.path.append(project_root)

import torch
import torchvision.transforms as T
import torchvision

def crop_highest_confidence_box(image):
    """
    Applique le modèle sur une image, sélectionne la boîte la plus probable et
    retourne la zone recadrée au format d'origine si le score dépasse un seuil.
    
    Args:
        image_path (str): Chemin de l'image à traiter.
        model (torch.nn.Module): Modèle de détection d'objets.
        confidence_threshold (float): Seuil de confiance minimum pour sélectionner une boîte.
        
    Returns:
        PIL.Image.Image ou None: La zone recadrée au format d'origine ou None si aucune boîte valide.
    """

    # Charger le modèle complet sauvegardé
    model = torch.load("src/crop_ticket/model_complet.pth", map_location=torch.device("cpu"))

    # Parcourir toutes les images dans le dossier
    confidence_threshold = 0.5  # Seuil de confiance pour afficher les prédictions
    # Charger l'image d'origine
    original_image = image
    original_width, original_height = original_image.size

    # Transformation de l'image pour le modèle
    transform = T.Compose([
        T.Resize((400, 400)),  # Appliquer la même transformation qu'à l'entraînement
        T.ToTensor()
    ])
    image_tensor = transform(original_image).unsqueeze(0)  # Ajouter une dimension batch

    # Faire la prédiction
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extraire les boîtes et les scores
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']

    if len(scores) == 0:
        return None  # Aucun objet détecté

    # Sélectionner la boîte avec le score le plus élevé
    max_score, max_idx = torch.max(scores, dim=0)
    if max_score < confidence_threshold:
        return None  # Aucune boîte avec un score suffisant

    # Récupérer la boîte correspondante
    selected_box = boxes[max_idx].tolist()
    x_min, y_min, x_max, y_max = selected_box

    # Recalculer les coordonnées pour l'image d'origine
    x_min = int(x_min / 400 * original_width)
    y_min = int(y_min / 400 * original_height)
    x_max = int(x_max / 400 * original_width)
    y_max = int(y_max / 400 * original_height)

    # Recadrer l'image au format d'origine
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
    return cropped_image