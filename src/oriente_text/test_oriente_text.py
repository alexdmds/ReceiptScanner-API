import sys
import os

# Ajouter dynamiquement le dossier parent à sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))  # Chemin vers le dossier parent contenant src
sys.path.append(project_root)

import matplotlib.pyplot as plt
from src.crop_ticket.apply_crop_ticket import crop_highest_confidence_box
from src.oriente_text.oriente_text import straighten_using_ocr


# Dossier contenant les images à tester
image_folder = "local_images"

# Parcourir toutes les images dans le dossier
confidence_threshold = 0.5  # Seuil de confiance pour afficher les prédictions
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if not image_path.lower().endswith(('png', 'jpg', 'jpeg')):
        continue  # Ignorer les fichiers non image

    print(f"Traitement de l'image : {image_name}")

    # Recadrer l'image avec la boîte de plus haute confiance
    cropped_image = crop_highest_confidence_box(image_path)

    if cropped_image is not None:
        # Afficher l'image recadrée avant redressement
        plt.figure(figsize=(10, 10))
        plt.imshow(cropped_image)
        plt.axis("off")
        plt.title(f"Image avant redressement - {image_name}")
        plt.show()

        # Appliquer le redressement
        straightened_image = straighten_using_ocr(cropped_image)

        # Afficher l'image redressée
        plt.figure(figsize=(10, 10))
        plt.imshow(straightened_image)
        plt.axis("off")
        plt.title(f"Image après redressement - {image_name}")
        plt.show()
    else:
        print(f"Aucune boîte valide pour {image_name}.")