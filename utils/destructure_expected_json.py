import json

# Liste des champs attendus dans le JSON
CHAMPS_ATTENDUS = {
    "nom_magasin": "Nom du magasin",
    "adresse": "Adresse",
    "telephone": "Téléphone",
    "date_achat": "Date d'achat",
    "heure_achat": "Heure d'achat",
    "prix_total": "Prix total",
    "produits_achetes": "Produits achetés"
}

# Liste des sous-champs attendus pour chaque produit
SOUS_CHAMPS_PRODUIT = {"produit", "quantite", "prix_unitaire"}


def format_ticket_data(data):
    """
    Retourne les informations du ticket sous forme de texte structuré.
    :param data: JSON des données du ticket.
    :return: Texte structuré pour affichage.
    """
    output = "\n=== Ticket de caisse ===\n\n"
    
    # Vérifier les champs manquants et en trop
    champs_manquants = [champ for champ in CHAMPS_ATTENDUS if champ not in data]
    champs_supplémentaires = [champ for champ in data if champ not in CHAMPS_ATTENDUS]

    if champs_manquants:
        output += "Attention : Champs manquants : " + ", ".join(champs_manquants) + "\n"
    if champs_supplémentaires:
        output += "Attention : Champs supplémentaires non prévus : " + ", ".join(champs_supplémentaires) + "\n"

    # Affichage des informations générales du ticket
    for champ, description in CHAMPS_ATTENDUS.items():
        if champ != "produits_achetes":
            output += f"{description} : {data.get(champ, 'N/A')}\n"

    # Affichage des produits achetés
    output += "\nProduits achetés :\n"
    produits = data.get("produits_achetes", [])
    if not isinstance(produits, list):
        output += "Attention : 'produits_achetes' n'est pas au format attendu (liste)\n"
    else:
        for produit in produits:
            champs_produit_manquants = SOUS_CHAMPS_PRODUIT - produit.keys()
            champs_produit_supplémentaires = produit.keys() - SOUS_CHAMPS_PRODUIT

            # Afficher les champs manquants ou en trop pour chaque produit
            if champs_produit_manquants:
                output += "  - Attention : Champs manquants dans un produit : " + ", ".join(champs_produit_manquants) + "\n"
            if champs_produit_supplémentaires:
                output += "  - Attention : Champs supplémentaires non prévus dans un produit : " + ", ".join(champs_produit_supplémentaires) + "\n"

            # Afficher les détails du produit
            nom_produit = produit.get("produit", "N/A")
            quantite = produit.get("quantite", "N/A")
            prix_unitaire = produit.get("prix_unitaire", "N/A")
            output += f"  - {nom_produit} (Quantité: {quantite}, Prix unitaire: {prix_unitaire})\n"

    output += "\n=======================\n"
    return output

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de JSON pour le test
    json_data = """
    {
        "nom_magasin": "FRANPRIX",
        "adresse": "23 BLD DE GRENELLE",
        "telephone": "01.45.79.19.34",
        "date_achat": "08-02-2024",
        "heure_achat": "11:28",
        "prix_total": "18.60EUR",
        "produits_achetes": [
            {"produit": "LASAGNES BOLOGI", "quantite": "1", "prix_unitaire": "5.65EUR"},
            {"produit": "PTI.SACR.CHOC.", "quantite": "1", "prix_unitaire": "2.05EUR"},
            {"produit": "COCA PET 1L75", "quantite": "1"}
        ],
        "champ_inattendu": "valeur_inattendue"
    }
    """

    # Charger le JSON
    data = json.loads(json_data)

    # Appeler la fonction pour afficher le ticket
    format_ticket_data(data)