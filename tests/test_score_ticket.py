import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
import pytest
import logging
from process_ticket.analyse_ticket import analyse_ticket  # Importer votre fonction principale
from utils.score_ticket import score_ticket  # Importer la fonction de scoring

# Configurer les logs pour afficher seulement les niveaux INFO et supérieurs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

@pytest.fixture(autouse=True)
def set_env_prod(monkeypatch):
    """
    Configure l'environnement à 'prod' pour les tests.
    """
    monkeypatch.setenv("ENV", "prod")

@pytest.mark.parametrize("image_name", [
    "ticket1.jpg",
    "ticket2.jpg",
    "ticket3.jpg",
    "ticket4.jpg",
    "ticket5.jpg",
    "ticket7.jpg",
    "ticket8.jpg",
])  # Ajouter tous les noms d'images à tester
def test_analyse_ticket(image_name):
    """
    Teste la fonction analyse_ticket en comparant les résultats avec les JSON attendus.
    """
    # Définir les chemins des fichiers
    image_path = os.path.join("tests/static", "images", image_name)
    json_path = os.path.join("tests/static", "json", image_name.replace(".jpg", ".json"))

    # Charger l'image de test
    with open(image_path, "rb") as f:
        image_content = f.read()

    # Appeler la fonction analyse_ticket
    result_json = analyse_ticket(image_content)

    # Charger le JSON attendu
    with open(json_path, "r") as f:
        expected_json = json.load(f)

    # Calculer le score
    score = score_ticket(expected_json, result_json)
    print(f"Score pour {image_name}: {score}/100")

    # Vérifier si le score est supérieur à 60%
    assert score >= 60, f"Le score pour {image_name} est trop faible : {score}/100"