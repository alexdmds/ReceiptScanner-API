import torch
from torch.utils.data import Dataset, DataLoader
from import_data import download_image, download_annotations
from google.cloud import storage

def prepare_target(annotation, image_size):
    targets = []
    for ann in annotation["annotations"]:
        x_min, y_min, width, height = ann["bbox"]
        x_max, y_max = x_min + width, y_min + height

        # Normaliser les coordonnées en fonction de la taille de l'image
        boxes = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        labels = torch.tensor([ann["category_id"]], dtype=torch.int64)
        
        targets.append({"boxes": boxes, "labels": labels})

    return targets

class TicketDataset(Dataset):
    def __init__(self, bucket_name, image_folder, annotation_folder, transform=None):
        # Initialiser le client GCS et le bucket
        client = storage.Client()
        self.bucket = client.get_bucket(bucket_name)
        
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.image_blobs = list(self.bucket.list_blobs(prefix=image_folder))
    
    def __len__(self):
        return len(self.image_blobs)
    
    def __getitem__(self, idx):
        image_blob = self.image_blobs[idx]
        image_name = image_blob.name.split("/")[-1]
        
        # Corriger le chemin d'accès en ajoutant un slash entre annotation_folder et image_name
        annotation_path = f"{self.annotation_folder}/{image_name.replace('.jpg', '.json')}"
        annotation_blob = self.bucket.blob(annotation_path)
        
        image = download_image(image_blob)
        annotations = download_annotations(annotation_blob)

        # Préparer les cibles
        targets = prepare_target(annotations, image.size)
        
        if self.transform:
            image = self.transform(image)
        
        return image, targets