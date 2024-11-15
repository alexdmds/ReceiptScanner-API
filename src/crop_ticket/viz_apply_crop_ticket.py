import os
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision
from apply_crop_ticket import crop_highest_confidence_box

# Charger le modèle sauvegardé
model_checkpoint_path = "src/crop_ticket/model_v1.pth"
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=None  # Ne pas charger les poids pré-entraînés
)
num_classes = 2  # Modifier selon votre modèle
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Charger l'état du modèle
model.load_state_dict(torch.load(model_checkpoint_path, map_location=torch.device("cpu")))

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
    cropped_image = crop_highest_confidence_box(image_path, model, confidence_threshold)

    if cropped_image is not None:
        # Afficher l'image recadrée
        plt.figure(figsize=(8, 8))
        plt.imshow(cropped_image)
        plt.axis("off")
        plt.title(f"Image recadrée pour {image_name}")
        plt.show()
    else:
        print(f"Aucune boîte valide pour {image_name}.")