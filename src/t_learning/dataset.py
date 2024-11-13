import json
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TicketDataset(Dataset):
    def __init__(self, annotation_folder, image_folder, transform=None):
        # Dossiers locaux
        self.annotation_folder = annotation_folder
        self.image_folder = image_folder
        #on récupère les vraies dimensions de l'image

        # Charger la liste des fichiers d'annotations locaux
        self.annotation_files = [
            f for f in os.listdir(annotation_folder) if f.endswith(".json")
        ]
        print("Nombre de fichiers d'annotation trouvés :", len(self.annotation_files))
        self.transform = transform

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        # Chemin de l'annotation
        annotation_path = os.path.join(self.annotation_folder, self.annotation_files[idx])
        
        # Charger l'annotation JSON
        with open(annotation_path, "r") as f:
            annotation_data = json.load(f)

        # Chemin de l'image
        image_name = os.path.basename(annotation_data['task']['data']['image'])
        image_path = os.path.join(self.image_folder, image_name)

        # Charger l'image
        image = Image.open(image_path).convert("RGB")

        self.image_size = image.size  # Récupère la taille sous la forme (largeur, hauteur)


        # Convertir l'image en numpy array pour Albumentations
        image = np.array(image)

        # Préparer les boîtes et labels pour Albumentations
        bboxes, labels = self.get_bboxes_and_labels(annotation_data)

        # Appliquer les transformations si définies
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        
        # Dénormaliser les boîtes pour les ramener aux valeurs en pixels
        bboxes = [
            [
                int(box[0] * self.image_size[0]), 
                int(box[1] * self.image_size[1]), 
                int(box[2] * self.image_size[0]), 
                int(box[3] * self.image_size[1])
            ]
            for box in bboxes
        ]
        # Vérifier si bboxes est vide et créer les tenseurs correspondants
        if not bboxes:
            targets = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }
        else:
            targets = {
                "boxes": torch.tensor(bboxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }
        
        image = image.float() / 255.0  # Convertir en float et normaliser entre 0 et 1
        return image, targets

    def get_bboxes_and_labels(self, annotation):
        bboxes = []
        labels = []
        for result in annotation.get("result", []):
            if result['type'] == 'rectanglelabels':
                box = result['value']
                # Normaliser les coordonnées par la taille de l'image
                x_min = box['x'] / 100  # Albumentations attend des valeurs entre 0 et 1
                y_min = box['y'] / 100
                x_max = (box['x'] + box['width']) / 100
                y_max = (box['y'] + box['height']) / 100
                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(1)  # Label pour "ticket"
        return bboxes, labels