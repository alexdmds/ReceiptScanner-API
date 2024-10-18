from flask import Flask, request, jsonify
import requests
from google.cloud import vision
from AnalyseText import get_structured_json_from_text

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

    # Initialisez le client Vision
    client = vision.ImageAnnotatorClient()

    # Créer un objet Image à partir du contenu de l'image téléchargée
    image = vision.Image(content=image_content)

    # Utiliser le client Vision pour détecter du texte dans l'image
    response = client.text_detection(image=image)
    texts = response.text_annotations
    texte = texts[0].description if texts else ''

    # Convertir le texte en JSON structuré
    response_text = get_structured_json_from_text(texte)

    return jsonify(response_text)

if __name__ == '__main__':
    app.run(debug=True)
