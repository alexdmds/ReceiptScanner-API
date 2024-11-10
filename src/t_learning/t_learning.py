import torch
from dataloader import TicketDataset, collate_fn
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision.transforms as T

# Utiliser les dossiers locaux pour les annotations et les images
annotation_folder = "local_annotations"
image_folder = "local_images"

# Initialiser le dataset avec les données locales
dataset = TicketDataset(annotation_folder, image_folder, transform=T.Compose([T.ToTensor()]))
print("Nombre total d'images dans le dataset :", len(dataset))

# Diviser le dataset en ensembles d'entraînement et de validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Créer les DataLoaders pour l'entraînement et la validation
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
print("Nombre de mini-batchs pour l'entraînement :", len(train_loader))
# Initialiser le modèle et l'optimiseur
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Utilisation de l'appareil :", device)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Entraînement et validation
num_epochs = 5
print("début de l'entraînement")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'image_name'} for t in targets if isinstance(t, dict)]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}")