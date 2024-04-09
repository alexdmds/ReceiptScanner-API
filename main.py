from flask import Flask, request, jsonify
from google.cloud import vision
from PIL import Image
import io

app = Flask(__name__)

@app.route('/api/process-ticket', methods=['POST'])
def process_ticket():
    if 'ticket_image' not in request.files:
        return jsonify({'error': 'No ticket image provided.'}), 400
    
    # Initialisez le client Vision
    client = vision.ImageAnnotatorClient()

    # Lisez l'image depuis le fichier envoyé
    file = request.files['ticket_image']
    content = file.read()

    # Préparez l'image pour Google Cloud Vision
    image = vision.Image(content=content)

    # Détectez le texte dans l'image
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        return jsonify({'error': response.error.message}), 500

    # Extrait le texte de la première annotation (contenu complet)
    extracted_text = texts[0].description if texts else ''

    return jsonify({'extracted_text': extracted_text})

if __name__ == '__main__':
    app.run(debug=True)
