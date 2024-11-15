import os
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision
from apply_crop_ticket import crop_highest_confidence_box

# Dossier contenant les images à tester
image_folder = "local_images"

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if not image_path.lower().endswith(('png', 'jpg', 'jpeg')):
        continue  # Ignorer les fichiers non image

    print(f"Traitement de l'image : {image_name}")

    # Recadrer l'image avec la boîte de plus haute confiance
    cropped_image = crop_highest_confidence_box(image_path)

    if cropped_image is not None:
        # Afficher l'image recadrée
        plt.figure(figsize=(8, 8))
        plt.imshow(cropped_image)
        plt.axis("off")
        plt.title(f"Image recadrée pour {image_name}")
        plt.show()
    else:
        print(f"Aucune boîte valide pour {image_name}.")