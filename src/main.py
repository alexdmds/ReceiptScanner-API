import sys
import os

# Ajouter dynamiquement le dossier parent à sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))  # Chemin vers le dossier parent contenant src
sys.path.append(project_root)


from flask import Flask, request, jsonify
import requests
from src.process_ticket.analyse_ticket import analyse_ticket

app = Flask(__name__)

@app.route('/api/process-ticket', methods=['POST'])

def process_ticket():
    # Vérifier si l'URL de l'image est fournie dans les données JSON de la requête
    if not request.json or 'image_url' not in request.json:
        return jsonify({'error': 'No image URL provided.'}), 400

    # Récupérer l'URL de l'image
    image_url = request.json['image_url']

    # Télécharger l'image à partir de l'URL
    try:
        response = requests.get(image_url)
        # Vérifier si la requête a réussi
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download image from URL.'}), 500
        image_content = response.content
    except Exception as e:
        return jsonify({'error': 'Error downloading image from URL: {}'.format(e)}), 500
    
    # Appeler la fonction d'analyse de l'image
    json = analyse_ticket(image_content)

    return jsonify(json)



if __name__ == '__main__':
    app.run(debug=True)
