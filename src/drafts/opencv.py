import cv2
import numpy as np

# Charger l'image
image = cv2.imread('tests/static/ticket_test_image.png')
cv2.imshow("Image originale", image)
cv2.waitKey(0)

# Convertir en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image en niveaux de gris", gray)
cv2.waitKey(0)

# Appliquer un flou pour réduire le bruit
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Image avec flou gaussien", blurred)
cv2.waitKey(0)

# Détecter les contours
edged = cv2.Canny(blurred, 30, 100)
cv2.imshow("Contours détectés", edged)
cv2.waitKey(0)

# Trouver les contours
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrer les contours pour trouver un contour quadrilatère de taille suffisante
ticket_contour = None
for contour in contours:
    # Approximation du contour
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Vérifier si c'est un quadrilatère et si son aire est suffisante
    if len(approx) == 4 and cv2.contourArea(approx) > 1000:
        # Calculer le rectangle englobant pour vérifier les proportions
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        
        # Vérifier si le ratio est raisonnable pour un ticket (généralement entre 0.2 et 0.8)
        if 0.2 < aspect_ratio < 0.8:
            ticket_contour = approx
            break

# Si un contour de ticket est trouvé, l'afficher
if ticket_contour is not None:
    # Dessiner le contour détecté sur une copie de l'image originale
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, [ticket_contour], -1, (0, 255, 0), 3)
    cv2.imshow("Image avec le contour du ticket", image_with_contours)
    cv2.waitKey(0)
else:
    print("Aucun contour de ticket trouvé")

# Fermer toutes les fenêtres
cv2.destroyAllWindows()