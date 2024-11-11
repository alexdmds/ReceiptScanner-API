import torch
from dataloader import TicketDataset, collate_fn
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision.transforms as T
import time

# Utiliser les dossiers locaux pour les annotations et les images
annotation_folder = "local_annotations"
image_folder = "local_images"
model_checkpoint_path = "src/t_learning/model_checkpoint.pth"  # Chemin pour sauvegarder le modèle
model_path = "src/t_learning/model.pth"  # Chemin pour sauvegarder le modèle final

# Transformation avec redimensionnement
transform = T.Compose([
    T.Resize((400, 400)),  # Redimensionner les images à 800x800
    T.ToTensor()
])


# Initialiser le dataset avec les données locales
dataset = TicketDataset(annotation_folder, image_folder, transform=transform)
print("Nombre total d'images dans le dataset :", len(dataset))

# Diviser le dataset en ensembles d'entraînement et de validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Créer les DataLoaders pour l'entraînement et la validation
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
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
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)


# Charger le checkpoint
checkpoint = torch.load(model_checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1  # Reprendre à l'époque suivante
loss = checkpoint['loss']  # Optionnel

# Mettre le modèle en mode entraînement
model.train()

# Entraînement et validation
num_epochs = 5
print("début de l'entraînement")
for epoch in range(num_epochs):
    start_time = time.time()  # Enregistrer le temps de début
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'image_name'} for t in targets if isinstance(t, dict)]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        print(losses)
        running_loss += losses.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    avg_train_loss = running_loss / len(train_loader)
    epoch_duration = time.time() - start_time  # Calculer le temps écoulé pour l'époque
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Duration: {epoch_duration:.2f} seconds")

    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_train_loss,
}, model_checkpoint_path)
    print(f"Modèle sauvegardé après l'époque {epoch+1} à {model_checkpoint_path}")


torch.save(model.state_dict(), model_path)
print("Modèle sauvegardé à", model_path)