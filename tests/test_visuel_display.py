import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.process_ticket.analyse_ticket import analyse_ticket
from PIL import Image, ExifTags
from src.utils.destructure_expected_json import format_ticket_data

# Chemin de l'image de test
image_path = "tests/static/ticket_test_image.png"

# Charger et lire le contenu de l'image
with open(image_path, "rb") as image_file:
    image_content = image_file.read()

# Appeler la fonction de traitement pour obtenir le JSON
response_text = analyse_ticket(image_content)
print(f"Response Text: {response_text}")  # Pour vérifier ce que contient response_text
# Charger le JSON si `response_text` est une chaîne
if isinstance(response_text, str):
    response_text = json.loads(response_text)

# Préparer l'affichage de l'image et du JSON
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# Charger l'image avec PIL et corriger l'orientation si nécessaire
ax0 = plt.subplot(gs[0])
image = Image.open(image_path)
try:
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    exif = image._getexif()
    if exif is not None:
        orientation_value = exif.get(orientation)
        if orientation_value == 3:
            image = image.rotate(180, expand=True)
        elif orientation_value == 6:
            image = image.rotate(270, expand=True)
        elif orientation_value == 8:
            image = image.rotate(90, expand=True)
except (AttributeError, KeyError, IndexError):
    # Pas de correction d'orientation si les métadonnées EXIF sont absentes
    pass

# Afficher l'image
ax0.imshow(image)
ax0.axis('off')
ax0.set_title("Ticket Image")

# Afficher le texte formaté avec `format_ticket_data`
ax1 = plt.subplot(gs[1])
formatted_text = format_ticket_data(response_text)  # Utilisation de la fonction formatée
ax1.text(0.05, 0.5, formatted_text, fontsize=10, va='center', ha='left', wrap=True)
ax1.axis('off')
ax1.set_title("JSON Output")

plt.tight_layout()
plt.show()