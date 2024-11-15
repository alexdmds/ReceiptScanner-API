import torch
from dataset import TicketDataset
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import time
import csv
from custom_transform import CustomTransform
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ouvrir ou créer un fichier CSV pour enregistrer les pertes
loss_file_path = "train_val_loss.csv"

# Écrire les en-têtes du fichier
with open(loss_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

# Utiliser les dossiers locaux pour les annotations et les images
annotation_folder = "local_annotations"
image_folder = "local_images"
model_checkpoint_path = "src/t_learning/model_checkpoint.pth"  # Chemin pour sauvegarder le modèle
model_path = "src/t_learning/model.pth"  # Chemin pour sauvegarder le modèle final

def collate_fn(batch):
    return tuple(zip(*batch))


# Initialiser le dataset avec les données locales
dataset = TicketDataset(annotation_folder, image_folder, transform=CustomTransform())
print("Nombre total d'images dans le dataset :", len(dataset))

# Diviser le dataset en ensembles d'entraînement et de validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Créer les DataLoaders pour l'entraînement et la validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
print("Nombre de mini-batchs pour l'entraînement :", len(train_loader))


# Importer le modèle avec les poids pré-entraînés
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
)

# Adapter le nombre de classes (background + "ticket")
num_classes = 2  # par exemple, 1 pour "ticket" + 1 pour "background"
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Utilisation de l'appareil :", device)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9, weight_decay=0.001)

# Scheduler de réduction du taux d'apprentissage si la perte de validation stagne
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

# Paramètres pour l'arrêt anticipé
early_stop_patience = 15
best_val_loss = float("inf")
epochs_no_improve = 0

'''
# Charger le checkpoint
checkpoint = torch.load(model_checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1  # Reprendre à l'époque suivante
loss = checkpoint['loss']  # Optionnel
'''
# Mettre le modèle en mode entraînement
model.train()

# Entraînement et validation
num_epochs = 400
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

    # Validation
    start_time = time.time()  # Enregistrer le temps de début
    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # Désactiver le calcul des gradients pour économiser de la mémoire
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items() if k != 'image_name'} for t in targets if isinstance(t, dict)]
            
            # Passer temporairement en mode train pour calculer la perte
            model.train()
            loss_dict = model(images, targets)
            model.eval()  # Repasser en mode eval immédiatement après

            # Assurez-vous que loss_dict est bien un dictionnaire
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
            else:
                print("Loss non calculé en mode évaluation pour ces données.")
    
    val_duration = time.time() - start_time  # Calculer le temps écoulé pour la validation
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('nan')
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}", f"Duration: {val_duration:.2f} seconds")

    # Enregistrer les pertes dans le fichier CSV
    with open(loss_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_train_loss,
}, model_checkpoint_path)
    print(f"Modèle sauvegardé après l'époque {epoch+1} à {model_checkpoint_path}")

    # Enregistrer si la perte de validation s'améliore
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_checkpoint_path)
        print(f"Modèle amélioré, sauvegardé à l'époque {epoch+1}")
    else:
        epochs_no_improve += 1

    # Arrêt anticipé
    if epochs_no_improve >= early_stop_patience:
        print("Arrêt anticipé activé")
        break


    # Scheduler
    scheduler.step(avg_val_loss)
    print(f"Taux d'apprentissage actuel : {optimizer.param_groups[0]['lr']}")

    
    if epoch % 10 == 0:
        torch.save(model.state_dict(), model_path)
        print("Modèle sauvegardé à", model_path)
        