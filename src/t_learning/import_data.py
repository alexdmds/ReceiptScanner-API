from google.cloud import storage
import json
import io
from PIL import Image
import torchvision.transforms as T

# Initialiser le client GCS
client = storage.Client()

# Spécifier le nom du bucket et les dossiers
bucket_name = "kadi_label_studio"
image_folder = "tickets/"
annotation_folder = "annotations/"

bucket = client.get_bucket(bucket_name)

def download_image(image_blob):
    # Télécharger l'image en mémoire et la charger avec PIL
    image_data = image_blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

def download_annotations(annotation_blob):
    # Télécharger les annotations en mémoire
    annotation_data = annotation_blob.download_as_bytes()
    annotations = json.loads(annotation_data)
    return annotations

def load_dataset():
    dataset = []

    # Parcourir les images et annotations
    image_blobs = bucket.list_blobs(prefix=image_folder)
    for image_blob in image_blobs:
        image_name = image_blob.name.split("/")[-1]  # Extraire le nom du fichier

        # Rechercher l'annotation correspondante
        annotation_path = annotation_folder + image_name.replace('.jpg', '.json')
        annotation_blob = bucket.blob(annotation_path)

        if annotation_blob.exists():
            image = download_image(image_blob)
            annotations = download_annotations(annotation_blob)

            # Préparer l'entrée pour le dataset
            dataset.append((image, annotations))
    return dataset