from flask import Flask, request, jsonify
from google.cloud import vision
from AnalyseSpatiale import get_text_from_image

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

    texte = get_text_from_image(content)

    return jsonify({'extracted_text': texte})

if __name__ == '__main__':
    app.run(debug=True)
