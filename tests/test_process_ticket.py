import pytest
import requests
from unittest.mock import patch
from src.main import app

# Simuler le client Flask pour les tests
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Simuler un test avec une URL d'image et un mock de l'API Google Vision
@patch('google.cloud.vision.ImageAnnotatorClient.text_detection')
def test_process_ticket(mock_text_detection, client):  # Attention : mock_text_detection doit être avant client
    # Simuler une réponse de l'API Vision
    mock_text_detection.return_value = type('', (), {})()
    mock_text_detection.return_value.text_annotations = [
        type('', (), {"description": "Test text from image"})()
    ]

    # URL d'image simulée (par exemple, une image d'un ticket)
    image_url = 'https://example.com/test_image.png'

    # Simuler une requête POST avec une URL d'image valide
    response = client.post('/api/process-ticket', json={'image_url': image_url})

    # Vérifier que le statut de la réponse est 200 (succès)
    assert response.status_code == 200

    # Vérifier que le texte renvoyé par l'API Vision est bien inclus dans la réponse
    json_data = response.get_json()
    assert 'Test text from image' in json_data.values()

# Test pour vérifier si l'URL de l'image n'est pas fournie
def test_process_ticket_no_image_url(client):
    response = client.post('/api/process-ticket', json={})
    assert response.status_code == 400
    assert response.get_json() == {'error': 'No image URL provided.'}

# Test pour une erreur lors du téléchargement d'image
@patch('requests.get')
def test_process_ticket_failed_image_download(mock_get, client):  # Attention : mock_get doit être avant client
    mock_get.return_value.status_code = 404  # Simuler une erreur 404 lors du téléchargement

    # URL d'image invalide
    image_url = 'https://example.com/invalid_image.png'

    response = client.post('/api/process-ticket', json={'image_url': image_url})

    assert response.status_code == 500
    assert response.get_json() == {'error': 'Failed to download image from URL.'}