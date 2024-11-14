import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import TicketDataset
from custom_transform import CustomTransform


annotation_folder = "local_annotations"
image_folder = "local_images"

def visualize_sample(image, bboxes, labels=None):
    """
    Affiche une image avec ses boîtes englobantes.
    
    Args:
        image: Image en format tensor ou numpy.
        bboxes: Liste des boîtes englobantes au format [x_min, y_min, x_max, y_max].
        labels: (optionnel) Liste des labels associés aux boîtes englobantes.
    """
    # Convertir le tensor en numpy pour l'affichage
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    # Convertir en format BGR pour OpenCV
    image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        color = (0, 255, 0)  # Couleur verte pour les boîtes
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        if labels is not None:
            label = str(labels[i])
            cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Affichage avec Matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def collate_fn(batch):
    return tuple(zip(*batch))

# Initialiser le dataset avec les données locales
dataset = TicketDataset(annotation_folder, image_folder, transform=CustomTransform())
# Charger un échantillon aléatoire pour vérifier les transformations et les annotations
sample_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
for i, (images, targets) in enumerate(sample_loader):
    image = images[0]
    target = targets[0]
    bboxes = target['boxes'].cpu().numpy()  # Récupérer les boîtes englobantes en numpy
    labels = target['labels'].cpu().numpy() if 'labels' in target else None

    print("Image shape:", image.shape)
    print("Boîtes englobantes:", bboxes)
    print("Labels:", labels)

    # Visualiser l'image avec les annotations
    visualize_sample(image, bboxes, labels)

    # Limiter l'affichage à quelques exemples
    if i >= 20:
        break