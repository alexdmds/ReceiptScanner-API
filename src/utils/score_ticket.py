from fuzzywuzzy import fuzz
from scipy.optimize import linear_sum_assignment
import numpy as np

def compare_text(value_ref, value_test):
    """Compare deux textes avec Fuzzy Matching."""
    if value_ref is None or value_test is None:
        return 0  # Aucun score si une des valeurs est absente
    return fuzz.ratio(str(value_ref), str(value_test))

def compare_numeric(value_ref, value_test, tolerance=0.05):
    """
    Correspondance exacte attendue pour les valeurs numériques.
    """
    if value_ref is None or value_test is None:
        return 0
    if value_ref == value_test:
        return 100
    return 0

def compare_exact_text(value_ref, value_test):
    """
    Correspondance exacte attendue pour les valeurs textuelles.
    """
    if value_ref is None or value_test is None:
        return 0
    if value_ref == value_test:
        return 100
    return 0

def compare_list(list_ref, list_test):
    """Compare deux listes d'éléments (produits) en utilisant l'algorithme hongrois pour le meilleur appariement."""
    if not list_ref or not list_test:
        return 0

    # Construire une matrice de similarité (coût inversé)
    similarity_matrix = []
    for ref_item in list_ref:
        row = []
        for test_item in list_test:
            similarity = fuzz.ratio(ref_item['produit'], test_item.get('produit', ''))
            row.append(100 - similarity)  # Inverser pour minimiser
        similarity_matrix.append(row)

    similarity_matrix = np.array(similarity_matrix)

    # Résoudre le problème d'appariement optimal
    row_ind, col_ind = linear_sum_assignment(similarity_matrix)

    # Calculer le score total basé sur les champs texts et les quantités

    # Pondérations par catégorie
    weights = {
        "produit": 40,
        "quantite": 20,
        "prix_unitaire": 40
    }

    nb_max = max(len(list_ref), len(list_test)) # Nombre maximal d'éléments

    total_score = 0
    for i, j in zip(row_ind, col_ind):
        ref_item = list_ref[i]
        test_item = list_test[j]
        print(ref_item, test_item)
        max_score = sum(weights.values())

        # Comparer chaque champ
        for key, weight in weights.items():
            if key == "produit":
                total_score += compare_text(ref_item.get(key), test_item.get(key)) * (weight / max_score)
            elif key in {"quantite", "prix_unitaire"}:
                total_score += compare_numeric(ref_item.get(key), test_item.get(key)) * (weight / max_score)

    return round(total_score/nb_max, 2)

def score_ticket(json_ref, json_test):
    """
    Évalue un JSON généré en le comparant à un JSON de référence.
    
    Args:
        json_ref (dict): JSON de référence attendu.
        json_test (dict): JSON généré à évaluer.
        
    Returns:
        float: Score entre 0 et 100 indiquant la qualité du JSON généré.
    """
    # Pondérations par catégorie
    weights = {
        "nom_magasin": 10,
        "adresse": 10,
        "telephone": 10,
        "date_achat": 10,
        "heure_achat": 10,
        "prix_total": 10,
        "produits_achetes": 40
    }

    total_score = 0
    max_score = sum(weights.values())

    # Comparer chaque champ
    for key, weight in weights.items():
        if key in {"nom_magasin", "adresse", "telephone"}:
            total_score += compare_text(json_ref.get(key), json_test.get(key)) * (weight / 100)
        elif key in {"date_achat", "heure_achat"}:
            total_score += compare_exact_text(json_ref.get(key), json_test.get(key)) * (weight / 100)
        elif key == "prix_total":
            total_score += compare_numeric(json_ref.get(key), json_test.get(key)) * (weight / 100)
        elif key == "produits_achetes":
            total_score += compare_list(json_ref.get(key), json_test.get(key)) * (weight / 100)

    # Calculer le score final
    final_score = (total_score / max_score) * 100
    return round(final_score, 2)




if __name__ == "__main__":
    # Exemple d'utilisation
    json_ref = {
        "nom_magasin": "Super U",
        "adresse": "1 Rue de la République, 75001 Paris",
        "telephone": "01 23 45 67 89",
        "date_achat": "2021-06-01",
        "heure_achat": "14:30",
        "prix_total": 42.95,
        "produits_achetes": [
            {"produit": "Pain", "quantite": 1, "prix_unitaire": 1.20},
            {"produit": "Beurre", "quantite": 1, "prix_unitaire": 2.30},
            {"produit": "Lait", "quantite": 6, "prix_unitaire": 0.80}
        ]
    }

    json_test = {
        "nom_magasin": "",
        "adresse": "1 Rue de",
        "telephone": "01 9",
        "date_achat": "2",
        "heure_achat": "",
        "prix_total": 42.5,
        "produits_achetes": [
            {"produit": "P", "quantite": 1, "prix_unitaire": 1.0},
            {"produit": "lt", "quantite": 1, "prix_unitaire": 1.00}
        ]
    }

    print(score_ticket(json_ref, json_test))  # Output: 100.0