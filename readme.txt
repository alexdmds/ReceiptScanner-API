## Structure JSON en sortie de `process_ticket`

Cette section décrit le format JSON retourné par l'API après le traitement d'un ticket de caisse.

### Exemple d'entrée JSON :

{
  "image_url": "https://firebasestorage.googleapis.com/v0/b/c-troop-70f68.appspot.com/o/tickets%2F2e457f8c-7ce5-41b4-9611-b9cda354379b6090562080649726705.jpg?alt=media&token=2304f8ba-1faa-450b-8c38-ae6f24fcbacb"
}


### Exemple de sortie JSON :

```json
{
  "estTicket": 1,
  "nom_magasin": "FRANPRIX",
  "adresse": "23 BLD DE GRENELLE",
  "telephone": "01.45.79.19.34",
  "date_achat": "06-02-2024",
  "heure_achat": "09:16",
  "prix_total": "20.38Eur",
  "produits_achetes": [
    {
      "produit": "VIENNOIS.A.GAS",
      "quantite": "1",
      "prix_unitaire": "1.50Eur"
    },
    {
      "produit": "LESS.24D.ECO 4",
      "quantite": "1",
      "prix_unitaire": "8.59Eur"
    },
    {
      "produit": "LAITUE 200G MF",
      "quantite": "1",
      "prix_unitaire": "1.89Eur"
    },
    {
      "produit": "FINDUS POMMES",
      "quantite": "1",
      "prix_unitaire": "3.85Eur"
    },
    {
      "produit": "CHIPOLATAS X6",
      "quantite": "1",
      "prix_unitaire": "4.55Eur"
    }
  ]
}