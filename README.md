# ReceiptScanner-API

## Description
API moderne développée avec Flask et déployée sur Google App Engine, spécialisée dans l'analyse et la structuration de tickets de caisse. L'application utilise l'intelligence artificielle et la vision par ordinateur pour extraire automatiquement les informations des tickets et les restituer dans un format JSON structuré.

## Technologies Principales
- **Backend**: Python 3.11, Flask
- **Déploiement**: Google App Engine
- **Traitement d'Images**: OpenCV, Pillow
- **OCR et IA**: Google Cloud Vision, PyTorch
- **Format de Sortie**: JSON structuré
- **Serveur WSGI**: Gunicorn

## Fonctionnalités
- Scan et analyse automatique de tickets de caisse
- Extraction intelligente des informations (montants, articles, dates, etc.)
- Restitution des données en JSON structuré
- API RESTful pour une intégration facile
- Traitement d'images haute performance
- Intégration avec Google Cloud Vision pour une reconnaissance précise

## Prérequis
- Python 3.11
- pip (gestionnaire de paquets Python)
- Compte Google Cloud Platform
- Accès à l'API Google Cloud Vision

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd ReceiptScanner-API
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/macOS
# ou
.\venv\Scripts\activate  # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Configuration
1. Configurer les variables d'environnement nécessaires
2. Configurer les credentials Google Cloud Vision
3. Ajuster les paramètres de traitement d'images selon vos besoins

## Déploiement
L'application est configurée pour être déployée sur Google App Engine. Utilisez la commande suivante pour déployer :
```bash
gcloud app deploy
```

## Tests
Les tests sont gérés avec pytest. Pour exécuter les tests :
```bash
pytest
```

## Structure du Projet
```
ReceiptScanner-API/
├── src/               # Code source principal
│   ├── vision/       # Traitement d'images et OCR
│   ├── api/          # Endpoints de l'API
│   └── utils/        # Utilitaires
├── tests/            # Tests unitaires et d'intégration
├── requirements.txt  # Dépendances du projet
├── app.yaml         # Configuration Google App Engine
└── ...
```

## Format de Sortie JSON
L'API renvoie les données extraites dans un format JSON structuré contenant :
- Informations du ticket (date, heure, magasin)
- Liste des articles avec prix
- Montants (sous-total, TVA, total)
- Métadonnées additionnelles

## Contribution
Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## Licence
[À spécifier]

## Contact
[À spécifier] 