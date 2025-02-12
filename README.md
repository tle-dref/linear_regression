# ft_linear_regression

Ce projet consiste à implémenter un algorithme de régression linéaire simple pour prédire le prix d'une voiture en fonction de son kilométrage.

## Description

Le programme est divisé en deux parties :
- Un programme d'apprentissage qui calcule les paramètres θ₀ et θ₁ de l'équation `prix = θ₀ + θ₁ × kilométrage`
- Un programme de prédiction qui utilise ces paramètres pour estimer le prix d'une voiture

## Prérequis

- Python 3.7+
- Bibliothèques requises :
  - numpy
  - matplotlib (pour la visualisation)
  - pandas (pour la lecture des données)


### Programme d'apprentissage

```bash
python3 train.py
```

Ce programme :
1. Lit les données d'entraînement depuis data.csv
2. Normalise les données
3. Applique la descente de gradient pour trouver les meilleurs paramètres θ₀ et θ₁
4. Sauvegarde ces paramètres dans un fichier
5. Affiche la ligne de régression et les données sur un graphique

### Programme de prédiction

```bash
python3 predict.py
```

Ce programme :
1. Charge les paramètres θ₀ et θ₁ précédemment calculés
2. Demande à l'utilisateur d'entrer un kilométrage
3. Prédit et affiche le prix estimé de la voiture

## Formules mathématiques

### Fonction hypothèse
h(x) = θ₀ + θ₁x

### Fonction de coût
J(θ₀,θ₁) = (1/2m) × Σ(h(xᵢ) - yᵢ)²

### Descente de gradient
θ₀ = θ₀ - α × (1/m) × Σ(h(xᵢ) - yᵢ)
θ₁ = θ₁ - α × (1/m) × Σ(h(xᵢ) - yᵢ) × xᵢ

où :
- m est le nombre d'exemples d'entraînement
- α est le taux d'apprentissage
- xᵢ est le kilométrage de l'exemple i
- yᵢ est le prix réel de l'exemple i

## Normalisation des données

Pour améliorer la convergence, les données sont normalisées selon la formule :
x_norm = (x - moyenne) / écart-type

