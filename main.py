from flask import Flask, request, jsonify
from google.cloud import vision
from AnalyseSpatiale import get_text_from_image
from AnalyseText import get_structured_json_from_text

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

    response_text = get_structured_json_from_text(texte)

    return jsonify(response_text)

if __name__ == '__main__':
    app.run(debug=True)
