import os
from google.cloud import storage

# Initialiser le client GCS
storage_client = storage.Client()
bucket_name = "kadi_label_studio"
bucket = storage_client.bucket(bucket_name)

# Chemins des dossiers locaux
local_annotation_folder = "local_annotations"
local_image_folder = "local_images"

# Créer les dossiers locaux si nécessaire
os.makedirs(local_annotation_folder, exist_ok=True)
os.makedirs(local_image_folder, exist_ok=True)

# Télécharger les annotations en ajoutant .json aux fichiers
annotation_blobs = bucket.list_blobs(prefix="annotations/")
for blob in annotation_blobs:
    # Ignorer les dossiers et ne traiter que les fichiers
    if not blob.name.endswith("/"):
        local_path = os.path.join(local_annotation_folder, os.path.basename(blob.name) + ".json")
        print(f"Téléchargement de {blob.name} vers {local_path}")
        blob.download_to_filename(local_path)

'''
# Télécharger les images en conservant leur extension d'origine
image_blobs = bucket.list_blobs(prefix="tickets/")
for blob in image_blobs:
    # Ignorer les dossiers et ne traiter que les fichiers d'image
    if blob.name.endswith((".jpg", ".png", ".jpeg")):
        local_path = os.path.join(local_image_folder, os.path.basename(blob.name))
        print(f"Téléchargement de {blob.name} vers {local_path}")
        blob.download_to_filename(local_path)

'''