import requests
import base64
import os
from google.cloud import secretmanager

def access_openai_key():
    """
    Accède à la clé API OpenAI stockée dans Google Secret Manager.
    """
    client = secretmanager.SecretManagerServiceClient()
    secret_name = "projects/441263626560/secrets/cle_OPEN_AI/versions/latest"
    response = client.access_secret_version(request={"name": secret_name})
    return response.payload.data.decode("UTF-8")

def encode_image_to_base64(image_path):
    """
    Encode une image en base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_ticket_info_from_image(image_path):
    """
    Envoie une image à l'API ChatGPT et retourne les informations du ticket sous forme de JSON structuré.
    """
    base64_image = encode_image_to_base64(image_path)
    headers = {
        "Authorization": f"Bearer {access_openai_key()}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Analyse l'image suivante pour déterminer si elle contient un ticket de caisse. "
                    "Retourne un JSON structuré avec les informations suivantes :\n"
                    "{\n"
                    "  \"estTicket\": 1 ou 0,\n"
                    "  \"nom_magasin\": \"Nom du Magasin ou champ vide si non disponible\",\n"
                    "  \"adresse\": \"Adresse du Magasin ou champ vide si non disponible\",\n"
                    "  \"telephone\": \"Numéro de Téléphone ou champ vide si non disponible\",\n"
                    "  \"date_achat\": \"Date d'Achat ou champ vide si non disponible\",\n"
                    "  \"heure_achat\": \"Heure d'Achat ou champ vide si non disponible\",\n"
                    "  \"prix_total\": \"Montant total ou champ vide si non disponible\",\n"
                    "  \"produits_achetes\": [\n"
                    "    {\n"
                    "      \"produit\": \"Nom du Produit ou champ vide si non disponible\",\n"
                    "      \"quantite\": Quantité du Produit ou champ vide si non disponible,\n"
                    "      \"prix_unitaire\": \"Prix du Produit ou champ vide si non disponible\"\n"
                    "    }\n"
                    "  ]\n"
                    "}\n"
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyse cette image pour en extraire les informations du ticket de caisse."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.01,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Erreur lors de l'appel à l'API: {response.status_code} - {response.text}")

# Exemple d'utilisation
if __name__ == "__main__":
    try:
        image_path = "tests/static/p0BRZDxUjvJVIfS7RNjz_photo.jpg"  # Remplace par le chemin vers ton image
        ticket_info = get_ticket_info_from_image(image_path)
        print(ticket_info)
    except Exception as e:
        print(f"Erreur : {e}")