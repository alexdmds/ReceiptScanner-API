import re

def normalize_json(data):
    """
    Normalise un JSON en appliquant des traitements spécifiques par champ.
    
    Args:
        data (dict): JSON à normaliser.
        
    Returns:
        dict: JSON normalisé.
    """

    def normalize_date(date):
        """Normalise une date au format JJ/MM/AAAA."""
        if not isinstance(date, str):
            return date
        # Tenter différents formats
        match = re.match(r"(\d{2})/(\d{2})/(\d{4})", date)  # Format DD/MM/YYYY
        if match:
            return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
        match = re.match(r"(\d{2})-(\d{2})-(\d{4})", date)  # Format DD-MM-YYYY
        if match:
            return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
        return date  # Retourne inchangé si aucun format reconnu
    

    def normalize_price(price):
        """Normalise un prix en flottant (float)."""
        if isinstance(price, str):
            price = re.sub(r"[^\d.,]", "", price)  # Supprime les symboles non numériques
            price = price.replace(",", ".")  # Remplace les virgules par des points
        try:
            return float(price)
        except ValueError:
            return price
        
    def normalize_phone(phone):
        """Normalise un numéro de téléphone au format 01 43 48 85 96."""
        if not isinstance(phone, str):
            return phone
        phone = re.sub(r"[^\d]", "", phone)  # Supprime tout sauf les chiffres
        if len(phone) == 10:
            return f"{phone[:2]} {phone[2:4]} {phone[4:6]} {phone[6:8]} {phone[8:]}"
        return phone  # Retourne inchangé si le format est invalide

    def normalize_quantity(quantity):
        """Normalise la quantité, retire les préfixes comme 'T'."""
        if isinstance(quantity, str):
            quantity = re.sub(r"[^\d]", "", quantity)  # Retire tout sauf les chiffres
        try:
            return int(quantity)
        except ValueError:
            return quantity

    normalized_data = {}
    for key, value in data.items():
        if isinstance(value, dict):  # Si le champ est un objet, traiter récursivement
            normalized_data[key] = normalize_json(value)
        elif isinstance(value, list):  # Si le champ est une liste, traiter chaque élément
            normalized_data[key] = [normalize_json(item) if isinstance(item, dict) else item for item in value]
        else:
            if key == "date_achat":
                normalized_data[key] = normalize_date(value)
            elif key == "telephone":
                normalized_data[key] = normalize_phone(value)
            elif key in {"prix_total", "prix_unitaire"}:
                normalized_data[key] = normalize_price(value)
            elif key == "quantite":
                normalized_data[key] = normalize_quantity(value)
            else:
                normalized_data[key] = value  # Laisser inchangé si aucune règle spécifique

    return normalized_data

if __name__ == "__main__":
    json_data = {
        "nom_magasin": "DIAGONAL  ",
        "adresse": " 84 RUE DE CHARONNE\n",
        "telephone": "01.43.48.85.96",
        "date_achat": "14/10/2024",
        "heure_achat": "13:10",
        "prix_total": "4.24 €",
        "produits_achetes": [
            {"produit": "FINES BULLES PER ", "quantite": "T1", "prix_unitaire": "1.99 €"},
            {"produit": "J.POUSS.EPIN.SS CR.FRAICHE 190G", "quantite": "T1", "prix_unitaire": "1.25 €"}
        ]
    }

    normalized_json = normalize_json(json_data)
    print(normalized_json)