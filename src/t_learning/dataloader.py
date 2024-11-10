import json
from google.cloud import storage
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import torch

class TicketDataset(Dataset):
    def __init__(self, bucket_name, annotation_folder, transform=None):
        client = storage.Client()
        self.bucket = client.get_bucket(bucket_name)
        
        # Liste des blobs d'annotation dans le dossier spécifié
        self.annotation_blobs = list(self.bucket.list_blobs(prefix=annotation_folder))
        print("Nombre de blobs d'annotation trouvés :", len(self.annotation_blobs))
        self.transform = transform

    def __len__(self):
        return len(self.annotation_blobs)

    def __getitem__(self, idx):
        # Récupérer le blob d'annotation
        annotation_blob = self.annotation_blobs[idx]
        print(f"Traitement du blob : {annotation_blob.name}")
        
        annotation_text = annotation_blob.download_as_text()
        print(f"Contenu brut de {annotation_blob.name} : {annotation_text[:100]}...")

        # Vérifier si le blob est vide
        if not annotation_text.strip():
            print(f"Le blob {annotation_blob.name} est vide, il sera ignoré.")
            return None  # Ignorer cette annotation et l'image correspondante

        # Charger le contenu JSON
        try:
            annotation_data = json.loads(annotation_text)
        except json.JSONDecodeError as e:
            print(f"Erreur de décodage JSON pour le blob {annotation_blob.name}: {e}")
            return None  # Ignorer cette annotation et l'image correspondante
        
        # Récupérer le chemin de l'image depuis l'annotation
        image_path = annotation_data['task']['data']['image']
        image_blob = self.bucket.blob(image_path.replace("gs://kadi_label_studio/", ""))
        
        # Télécharger et charger l'image
        image_data = image_blob.download_as_bytes()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Préparer les cibles (annotations) pour l'image
        targets = self.prepare_target(annotation_data, image.size)
        
        if self.transform:
            image = self.transform(image)
        
        return image, targets

    def prepare_target(self, annotation, image_size):
        targets = {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)}
        
        for result in annotation.get("result", []):
            if result['type'] == 'rectanglelabels':
                box = result['value']
                x_min = box['x'] / 100 * image_size[0]
                y_min = box['y'] / 100 * image_size[1]
                width = box['width'] / 100 * image_size[0]
                height = box['height'] / 100 * image_size[1]
                x_max = x_min + width
                y_max = y_min + height

                # Ajouter la boîte englobante et le label à `targets`
                targets["boxes"] = torch.cat((targets["boxes"], torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)), dim=0)
                targets["labels"] = torch.cat((targets["labels"], torch.tensor([1], dtype=torch.int64)), dim=0)  # 1 pour "ticket"

        # Retourner `targets` vide si aucune boîte n'a été trouvée
        return targets

# Collate function pour ignorer les échantillons `None`
def collate_fn(batch):
    # Retirer les éléments `None` (par exemple, si une image ou des annotations ne sont pas valides)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    return tuple(zip(*batch))