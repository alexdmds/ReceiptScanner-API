import numpy as np
import cv2
import torch

class CustomTransform:
    def __init__(self, crop_size=(300, 300), resize_size=(400, 400), color_jitter_factor=0.1):
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.color_jitter_factor = color_jitter_factor

    def apply_color_jitter(self, image):
        # Variation de la luminosité
        brightness = np.random.uniform(1 - self.color_jitter_factor, 1 + self.color_jitter_factor)
        contrast = np.random.uniform(1 - self.color_jitter_factor, 1 + self.color_jitter_factor)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 50 - 50)
        return image

    def random_crop(self, image, bboxes, crop_pourcentage=0.1):
        # définit un random crop entre 0 et 10% de la taille d'origine sur tous les bords
        #remet l'image à sa dimension d'origine
        h0, w0 = image.shape[:2]

        x_start = np.random.randint(0, int(w0 * crop_pourcentage))
        x_end = np.random.randint(int(w0 * (1-crop_pourcentage)), w0)
        y_start = np.random.randint(0, int(h0 * crop_pourcentage))
        y_end = np.random.randint(int(h0 * (1- crop_pourcentage)), h0)

        image = image[y_start:y_end, x_start:x_end]
        h1, w1 = image.shape[:2]

        #on remet l'image à sa taille d'origine
        image = cv2.resize(image, (w0, h0))

        # Ajuster les boîtes englobantes pour le recadrage
        new_bboxes = []
        for box in bboxes:
            x_min, y_min, x_max, y_max = box
            x_min = min(max(0, x_min*w0 - x_start) / w1, 1)
            y_min = min(max(0, y_min*h0 - y_start) / h1, 1)
            x_max = min(max(0, x_max*w0 - x_start) / w1, 1)
            y_max = min(max(0, y_max*h0 - y_start) / h1, 1)
            new_bboxes.append([x_min, y_min, x_max, y_max])

        bboxes = new_bboxes
        return image, bboxes

    def random_rotation(self, image, bboxes, angle_range=(-5, 5)):
        # Choisir un angle de rotation aléatoire
        angle = np.random.uniform(*angle_range)

        # Obtenir la matrice de rotation et appliquer la rotation
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated_image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)

        # Transformer les boîtes englobantes
        new_bboxes = []
        for box in bboxes:
            points = np.array([
                [box[0] * w, box[1] * h],
                [box[2] * w, box[1] * h],
                [box[2] * w, box[3] * h],
                [box[0] * w, box[3] * h]
            ])
            # Appliquer la rotation sur chaque coin de la boîte englobante
            points = np.dot(rot_matrix[:, :2], points.T).T + rot_matrix[:, 2]
            # Calcul des nouvelles coordonnées min et max, limitées entre 0 et 1
            x_min, y_min = np.maximum(points.min(axis=0) / [w, h], 0)
            x_max, y_max = np.minimum(points.max(axis=0) / [w, h], 1)
            new_bboxes.append([x_min, y_min, x_max, y_max])

        return rotated_image, new_bboxes
    
    def add_gaussian_noise(self, image, noise_factor=0.02):
        noise = np.random.normal(0, noise_factor, image.shape)
        noisy_image = np.clip(image + noise * 255, 0, 255).astype(np.uint8)
        return noisy_image

    def apply_sharpness(self, image, alpha=1.2):
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        sharp_image = cv2.addWeighted(image, alpha, blurred, 1 - alpha, 0)
        return sharp_image

    def random_translation(self, image, bboxes, max_translation=0.10):
        h, w = image.shape[:2]
        
        # Calculer les décalages aléatoires sur les axes x et y
        tx = int(np.random.uniform(-max_translation, max_translation) * w)
        ty = int(np.random.uniform(-max_translation, max_translation) * h)
        

        # Calculer la taille des bordures nécessaires pour appliquer la réflexion
        left, right = abs(tx) if tx > 0 else 0, abs(tx) if tx < 0 else 0
        top, bottom = abs(ty) if ty > 0 else 0, abs(ty) if ty < 0 else 0


        # Ajouter les bordures réfléchies autour de l'image avant la translation
        image_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)



        # Créer une matrice de transformation pour appliquer uniquement la translation
        trans_matrix = np.float32([[1, 0, tx-left], [0, 1, ty-top]])
        translated_image = cv2.warpAffine(image_with_border, trans_matrix, (w, h))

        # Ajuster les boîtes englobantes
        new_bboxes = []
        for box in bboxes:
            x_min = min(max((box[0] * w + tx) / w, 0), 1)
            y_min = min(max((box[1] * h + ty) / h, 0), 1)
            x_max = min(max((box[2] * w + tx) / w, 0), 1)
            y_max = min(max((box[3] * h + ty) / h, 0), 1)
            new_bboxes.append([x_min, y_min, x_max, y_max])

        return translated_image, new_bboxes

    def zoom_out_with_reflection(self, image, bboxes, scale=0.8):
        h, w = image.shape[:2]
        
        actual_scale = np.random.uniform(scale, 1)

        # Calculer la nouvelle taille réduite de l'image
        new_w, new_h = int(w * actual_scale), int(h * actual_scale)
        scaled_image = cv2.resize(image, (new_w, new_h))
        
        # Calculer les bordures nécessaires pour revenir à la taille d'origine
        left = (w - new_w) // 2
        right = w - new_w - left
        top = (h - new_h) // 2
        bottom = h - new_h - top

        # Ajouter les bordures réfléchies
        image_with_border = cv2.copyMakeBorder(
            scaled_image, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, 
            value=(np.random.randint(256), np.random.randint(256), np.random.randint(256))
        )

        # Ajuster les boîtes englobantes pour tenir compte du dézoom
        new_bboxes = []
        for box in bboxes:
            x_min = left / w + box[0] * actual_scale
            y_min = top / h + box[1] * actual_scale
            x_max = left / w + box[2] * actual_scale
            y_max = top / h + box[3] * actual_scale
            new_bboxes.append([x_min, y_min, x_max, y_max])

        return image_with_border, new_bboxes

    def __call__(self, image, bboxes):
        # Appliquer jitter de couleur
        image = self.apply_color_jitter(image)

        # Appliquer zoom out avec réflexion
        image, bboxes = self.zoom_out_with_reflection(image, bboxes, scale=0.95)

        # Appliquer translation aléatoire
        image, bboxes = self.random_translation(image, bboxes, max_translation=0.05)

        # Appliquer rotation aléatoire
        image, bboxes = self.random_rotation(image, bboxes)

        # Ajouter du bruit gaussien
        image = self.add_gaussian_noise(image)

        # Appliquer un renforcement des bords
        image = self.apply_sharpness(image)

        # Appliquer recadrage aléatoire
        image, bboxes = self.random_crop(image, bboxes, crop_pourcentage=0.05)

        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0, bboxes