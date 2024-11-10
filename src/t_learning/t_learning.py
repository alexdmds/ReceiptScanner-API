import torch
from dataloader import TicketDataset, collate_fn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as T

# Initialiser le dataset et le DataLoader
transform = T.Compose([T.ToTensor()])  # Ajouter d'autres transformations si nécessaire
ticket_dataset = TicketDataset("kadi_label_studio", "annotations", transform=transform)
data_loader = DataLoader(ticket_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Initialiser le modèle et l'optimiseur
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Entraînement sur plusieurs époques
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        print(f"Nombre d'images: {len(images)}, Nombre de cibles: {len(targets)}")
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets if isinstance(t, dict)]
        
        # Vérifiez le contenu de `targets`
        print(f"Targets: {targets}")
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(f"Loss: {losses.item()}")
    
    print(f"Epoch {epoch+1}, Loss: {losses.item()}")