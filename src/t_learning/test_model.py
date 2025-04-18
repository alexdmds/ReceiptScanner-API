import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision

# Charger le modèle sauvegardé
model_checkpoint_path = "src/t_learning/model.pth"
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=None  # Ne pas charger les poids pré-entraînés
)
num_classes = 2  # Modifier selon votre modèle
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Charger l'état du modèle
model.load_state_dict(torch.load(model_checkpoint_path, map_location=torch.device("cpu")))
model.eval()

# Charger l'image de test
image_path = "tests/static/462542886_926539116209447_6025966826638074524_n.jpg"  # Remplacez par le chemin de votre image
original_image = Image.open(image_path).convert("RGB")

# Appliquer la rotation si nécessaire
original_image = original_image.rotate(-90, expand=True)  # Rotation de 90 degrés dans le sens antihoraire

# Transformation de l'image pour le modèle
transform = T.Compose([
    T.Resize((400, 400)),  # Appliquez la même transformation que pendant l'entraînement
    T.ToTensor()
])
image_tensor = transform(original_image).unsqueeze(0)  # Ajouter une dimension batch

# Faire la prédiction
with torch.no_grad():
    predictions = model(image_tensor)
print(predictions)

# Extraire les boîtes prédites avec un seuil de confiance
confidence_threshold = 0.2
boxes = predictions[0]['boxes']
scores = predictions[0]['scores']

# Filtrer les boîtes en fonction du seuil de confiance
filtered_boxes = boxes[scores > confidence_threshold].tolist()
print("Nombre de boîtes prédites :", len(filtered_boxes))
print("Boîtes prédites (x_min, y_min, x_max, y_max) :", filtered_boxes)

# Convertir l'image transformée (400x400) en image PIL pour dessin
resized_image = transform(original_image).permute(1, 2, 0).numpy() * 255
resized_image = Image.fromarray(resized_image.astype('uint8'))

# Afficher l'image redimensionnée avec les boîtes
draw = ImageDraw.Draw(resized_image)
for box in filtered_boxes:
    x_min, y_min, x_max, y_max = box
    draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=3)

# Afficher l'image avec les prédictions
plt.imshow(resized_image)
plt.axis("off")
plt.title("Image redimensionnée avec boîte(s) prédite(s)")
plt.show()