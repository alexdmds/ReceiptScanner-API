from google.cloud import vision
import cv2
import numpy as np
from PIL import Image, ImageDraw

def detect_ticket(image_path):
    # --- Étape 1 : Détection de texte ---
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Obtenir les coordonnées du texte
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        print("Aucun texte détecté.")
        return None

    # Obtenir les coordonnées du cadre englobant des blocs de texte
    ticket_contour = texts[0].bounding_poly
    text_coordinates = [(vertex.x, vertex.y) for vertex in ticket_contour.vertices]
    x_min = min(coord[0] for coord in text_coordinates)
    y_min = min(coord[1] for coord in text_coordinates)
    x_max = max(coord[0] for coord in text_coordinates)
    y_max = max(coord[1] for coord in text_coordinates)
    
    # Afficher le cadre englobant du texte
    img_with_text_contour = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img_with_text_contour)
    draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=3)
    img_with_text_contour.show(title="Cadre englobant du texte")
    
    # --- Étape 2 : Analyse des espaces blancs autour du texte ---
    with Image.open(image_path) as img:
        img_gray = img.convert("L")
        img_np = np.array(img_gray)
        ticket_region = img_np[y_min:y_max, x_min:x_max]

        # Calculer la moyenne de l'intensité dans la zone englobante
        mean_intensity = np.mean(ticket_region)
        print(f"Moyenne d'intensité dans la zone du texte : {mean_intensity}")
        if mean_intensity > 200:  # Supposons que > 200 signifie "espace blanc"
            print("Ticket non détecté : fond non suffisamment clair.")
            return None

    # --- Étape 3 : Détection des bords pour confirmation du rectangle ---
    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
    cv2.imshow("Image floutée", blurred)
    cv2.waitKey(0)
    
    edged = cv2.Canny(blurred, 30, 150)
    cv2.imshow("Contours détectés avec Canny", edged)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_all_contours = cv2.imread(image_path)
    cv2.drawContours(img_with_all_contours, contours, -1, (255, 0, 0), 2)
    cv2.imshow("Tous les contours détectés", img_with_all_contours)
    cv2.waitKey(0)

    for contour in contours:
        # Vérifier si le contour est suffisamment grand
        if cv2.contourArea(contour) > 1000:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                # Vérifier si ce contour contient le cadre du texte
                x, y, w, h = cv2.boundingRect(contour)
                if x <= x_min and y <= y_min and x + w >= x_max and y + h >= y_max:
                    # Afficher le contour détecté comme ticket
                    img_with_ticket_contour = cv2.imread(image_path)
                    cv2.drawContours(img_with_ticket_contour, [approx], -1, (0, 255, 0), 3)
                    cv2.imshow("Ticket détecté", img_with_ticket_contour)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    return

    print("Ticket non détecté.")
    cv2.destroyAllWindows()

# Appeler la fonction pour tester avec une image
detect_ticket('tests/static/ticket_test_image.png')