import json
from torch.utils.data import Dataset
from PIL import Image
import torch
import os

class TicketDataset(Dataset):
    def __init__(self, annotation_folder, image_folder, transform=None):
        # Dossiers locaux
        self.annotation_folder = annotation_folder
        self.image_folder = image_folder
        self.transform = transform

        # Charger la liste des fichiers d'annotations locaux
        self.annotation_files = [
            f for f in os.listdir(annotation_folder) if f.endswith(".json")
        ]
        print("Nombre de fichiers d'annotation trouvés :", len(self.annotation_files))

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
        
        # Préparer les cibles (annotations) pour l'image
        targets = self.prepare_target(annotation_data)
        
        if self.transform:
            image = self.transform(image)
        
        return image, targets

    def prepare_target(self, annotation):
        # On laisse les boîtes en pourcentages
        targets = {"boxes": [], "labels": []}
        
        for result in annotation.get("result", []):
            if result['type'] == 'rectanglelabels':
                box = result['value']
                x_min = box['x']
                y_min = box['y']
                x_max = x_min + box['width']
                y_max = y_min + box['height']

                # Ajouter la boîte englobante et le label en pourcentages
                targets["boxes"].append([x_min, y_min, x_max, y_max])
                targets["labels"].append(1)  # 1 pour "ticket"

        # Si aucune boîte n'est présente, on retourne des tenseurs vides pour "boxes" et "labels"
        if len(targets["boxes"]) == 0:
            targets["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            targets["labels"] = torch.zeros((0,), dtype=torch.int64)
        else:
            # Sinon, on convertit en tenseurs
            targets["boxes"] = torch.tensor(targets["boxes"], dtype=torch.float32)
            targets["labels"] = torch.tensor(targets["labels"], dtype=torch.int64)

        return targets

# Collate function pour ignorer les échantillons `None`
def collate_fn(batch):
    # Retirer les éléments `None` (par exemple, si une image ou des annotations ne sont pas valides)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    return tuple(zip(*batch))