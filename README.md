# C-Troop App GAE

## Description
Application web moderne développée avec Flask et déployée sur Google App Engine. Cette application utilise des technologies de pointe pour le traitement d'images et l'intelligence artificielle.

## Technologies Principales
- **Backend**: Python 3.11, Flask
- **Déploiement**: Google App Engine
- **Traitement d'Images**: OpenCV, Pillow
- **Intelligence Artificielle**: PyTorch, scikit-learn
- **Serveur WSGI**: Gunicorn

## Fonctionnalités
- Traitement d'images avancé
- Intégration avec Google Cloud Vision
- API RESTful
- Interface utilisateur moderne

## Prérequis
- Python 3.11
- pip (gestionnaire de paquets Python)
- Compte Google Cloud Platform

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd c_troop_app_gae
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
2. Assurez-vous d'avoir les credentials Google Cloud appropriés

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
c_troop_app_gae/
├── src/               # Code source principal
├── tests/            # Tests unitaires et d'intégration
├── requirements.txt  # Dépendances du projet
├── app.yaml         # Configuration Google App Engine
└── ...
```

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