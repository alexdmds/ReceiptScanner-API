import os
from google.cloud import storage

def download_model_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """
    Télécharge un fichier depuis GCS si nécessaire.

    Args:
        bucket_name (str): Nom du bucket GCS.
        source_blob_name (str): Chemin du fichier dans GCS.
        destination_file_name (str): Chemin local où enregistrer le fichier.

    Returns:
        str: Chemin local du fichier téléchargé.
    """
    if not os.path.exists(destination_file_name):
        print(f"Téléchargement du modèle depuis GCS : {source_blob_name}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Modèle téléchargé et enregistré à : {destination_file_name}")
    else:
        print(f"Modèle déjà disponible localement : {destination_file_name}")

    return destination_file_name