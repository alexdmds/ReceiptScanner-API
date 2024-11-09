from transformers import DonutProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

# Charger le processor et le modèle
processor = DonutProcessor.from_pretrained("mychen76/invoice-and-receipts_donut_v1")
model = AutoModelForImageTextToText.from_pretrained("mychen76/invoice-and-receipts_donut_v1")

# Charger une image
image = Image.open("tests/static/ticket_test_image.png").convert("RGB")

# Préparer l’image
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Passer l'image au modèle
output = model.generate(pixel_values)

# Décoder la sortie
result = processor.decode(output[0], skip_special_tokens=True)
print(result)