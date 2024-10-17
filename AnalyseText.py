import requests
import os
import time
import json

# Remplacez ceci par votre propre clé API d'OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

text_test = """
franprix FRANPRIX E 005408-11
23 BLD DE GRENELLE
BIENVENUE !
TEL : 01.45.79.19.34
* DUPLICATA * --
VIENNOIS.A.GAS 11 1.50Eur
LESS.24D.ECO 4 T2 8.59Eur
LAITUE 200G MF T1 1.89Eur
FINDUS POMMES 2 T1 3.85Eur
CHIPOLATAS X6 T1 4.55Eur
SOUS - TOTAL * 20.38Eur
TOTAL A PAYER 20.38Eur
CARTES BLEUES A 20.38Eur
S - Ttl en Francs qeia anse 133.68F
1 Euro = 6.55957 Frs
Taux - TT - Tot.HT -- Tot . TVA -- Tot . TTC-
11.18 0.611 11.79 | | 5.5 % | 11.181
7.16 1.431 8.59 ] | 20 % | 7.16
CUMULEZ DES EUROS !
Demandez à votre magasin
La carte de fidélité FRANPRIX !
06-02-2024 MARDI 09:16
CLK 36 R1 005408-11 37
5Article ( s )
* DUPLICATA *
"""

if OPENAI_API_KEY is None:
    with open('/Users/alexisdemonts/Documents/GitHub/c_troop_app_gae/cle_OPEN_AI.txt', 'r') as file:
        OPENAI_API_KEY = file.read().strip()
# L'ID de votre assistant spécifique
ASSISTANT_ID = "asst_IvRFQQMyRce12Q0JX4QuSaU1"

def call_openai_assistant(text):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v1",
    }

    # Création d'un nouveau thread
    thread_response = requests.post("https://api.openai.com/v1/threads", headers=headers)
    if thread_response.status_code != 200:
        raise Exception(f"Erreur lors de la création du Thread: {thread_response.status_code} {thread_response.text}")
    
    thread_id = thread_response.json()['id']

    # Ajout d'un message au thread
    message_data = {
        "role": "user",
        "content": text,
    }
    message_response = requests.post(f"https://api.openai.com/v1/threads/{thread_id}/messages", json=message_data, headers=headers)
    if message_response.status_code != 200:
        raise Exception(f"Erreur lors de l'ajout du message: {message_response.status_code} {message_response.text}")
    
    # Lancer un run avec l'assistant
    run_data = {
        "assistant_id": ASSISTANT_ID,
    }
    run_response = requests.post(f"https://api.openai.com/v1/threads/{thread_id}/runs", json=run_data, headers=headers)
    if run_response.status_code != 200:
        raise Exception(f"Erreur lors du lancement du run: {run_response.status_code} {run_response.text}")

    # Attendre que la réponse soit prête (cet exemple attend simplement, mais vous pourriez vouloir sonder le statut du run)
    time.sleep(60)  # Attente de 60 secondes pour la simplicité, mais considérez une boucle de sondage

    # Récupérer les messages du thread pour obtenir la réponse
    messages_response = requests.get(f"https://api.openai.com/v1/threads/{thread_id}/messages", headers=headers)
    if messages_response.status_code != 200:
        raise Exception(f"Erreur lors de la récupération des messages: {messages_response.status_code} {messages_response.text}")

    messages = messages_response.json()
    # Supposant que la dernière réponse est celle de l'assistant
    last_message = messages['data'][-1]['content']
    print(json.dumps(messages, indent=4))
    return last_message

def get_structured_json_from_text(text):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Ajustement ici pour utiliser "messages" au lieu de "prompt"
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
        "temperature": 0.7,
        "max_tokens": 1024
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
    if response.status_code == 200:
        # Note: La structure de la réponse peut être légèrement différente ici, donc ajuste si nécessaire
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Erreur lors de l'appel à l'API: {response.status_code} - {response.text}")


# Exemple d'utilisation
if __name__ == "__main__":
    try:
        structured_json = get_structured_json_from_text(text_test)
        print("Réponse structurée JSON de l'assistant:", structured_json)
    except Exception as e:
        print(e)
