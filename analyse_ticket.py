from utils.fonction_actuelle import analyse_image
#Ce fichier va contenir la fonction principale appelée dans le main qui prend en entrée une image et qui retourne un json structuré
#Il appelle d'autres utilitaires pour réaliser les fonctions unitaires


def analyse_ticket(image):

    # Appeler la fonction d'analyse de l'image
    json = analyse_image(image)

    return json