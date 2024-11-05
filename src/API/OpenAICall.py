import requests
import os
from google.cloud import secretmanager

def access_openai_key():
    client = secretmanager.SecretManagerServiceClient()
    secret_name = "projects/441263626560/secrets/cle_OPEN_AI/versions/latest"
    response = client.access_secret_version(request={"name": secret_name})
    return response.payload.data.decode("UTF-8")

def get_structured_json_from_text_OpenAI(text):
    headers = {
        "Authorization": f"Bearer {access_openai_key()}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "Analyse le texte brut suivant et détermine s'il s'agit d'un ticket de caisse. Retourne les informations dans un format JSON structuré comme indiqué ci-dessous. Si le texte est identifié comme un ticket de caisse, renvoie \"estTicket\": 1, sinon renvoie \"estTicket\": 0. Si certaines informations sont incertaines ou non disponibles, laisse les champs correspondants vides, mais assure-toi de toujours respecter la structure JSON suivante :\n\n{\n  \"estTicket\": 0 ou 1,\n  \"nom_magasin\": \"Nom du Magasin ou champ vide si non disponible\",\n  \"adresse\": \"Adresse du Magasin ou champ vide si non disponible\",\n  \"telephone\": \"Numéro de Téléphone ou champ vide si non disponible\",\n  \"date_achat\": \"Date d'Achat ou champ vide si non disponible\",\n  \"heure_achat\": \"Heure d'Achat ou champ vide si non disponible\",\n  \"prix_total\": \"Montant total du ticket ou champ vide si non disponible\",\n  \"produits_achetes\": [\n    {\n      \"produit\": \"Nom du Produit 1 ou champ vide si non disponible\",\n      \"quantite\": Quantité du Produit 1 ou champ vide si non disponible,\n      \"prix_unitaire\": \"Prix du Produit 1 ou champ vide si non disponible\"\n    }\n    // Ajoute d'autres produits selon les données disponibles\n  ]\n}\n\nGénère uniquement la structure JSON demandée, sans commentaire supplémentaire. Assure-toi de conserver la structure, même si certaines informations ne peuvent pas être remplies."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "temperature": 0.15,
        "max_tokens": 1024
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Erreur lors de l'appel à l'API: {response.status_code} - {response.text}")

# Fonction simulée pour l'environnement local
def mock_get_structured_json_from_text():
    return {
        "estTicket": 1,
        "nom_magasin": "magMock",
        "adresse": "dans l'espace",
        "telephone": "0145791934",
        "date_achat": "2024-11-04",
        "heure_achat": "21:14",
        "prix_total": "6.59",
        "produits_achetes": [
            {"produit": "un produit test", "quantite": 1, "prix_unitaire": "0.95"},
            {"produit": "un autre produit test", "quantite": 1, "prix_unitaire": "2.65"}
        ]
    }

# Sélection de la fonction en fonction de l'environnement
def get_structured_json_from_text(text):
    if os.getenv("ENV", "local") == "local":
        return mock_get_structured_json_from_text()
    else:
        return get_structured_json_from_text_OpenAI(text)