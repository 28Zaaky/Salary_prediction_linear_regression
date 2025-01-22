# Régression Linéaire - Prédiction de Salaire  

## Description  
Ce projet implémente un modèle de régression linéaire simple pour prédire les salaires en fonction du nombre d'années d'expérience. L'objectif est d'étudier la relation entre ces deux variables et d'entraîner un modèle capable de prédire de nouveaux salaires.  

## Fonctionnalités  
- Chargement des données depuis un fichier CSV (`Salary_Data.csv`).  
- Visualisation des données pour comprendre la relation entre expérience et salaire.  
- Division des données en ensembles d'entraînement et de test.  
- Création et entraînement d'un modèle de régression linéaire.  
- Évaluation des performances du modèle (MAE, MSE, RMSE, R²).  
- Visualisation des résultats (valeurs réelles et prédictions).  
- Prédiction pour de nouvelles données d'années d'expérience.  

## Données  
Le fichier de données `Salary_Data.csv` contient deux colonnes :  
- **YearsExperience** : Nombre d'années d'expérience (variable indépendante).  
- **Salary** : Salaire associé (variable dépendante).  

## Visualisation  
Les données sont visualisées à l'aide de graphiques :  
- Nuage de points pour représenter la relation entre expérience et salaire.  
- Ligne de régression pour illustrer les prédictions du modèle.  

## Métriques d'Évaluation  
- **MAE** (Erreur absolue moyenne)  
- **MSE** (Erreur quadratique moyenne)  
- **RMSE** (Racine carrée de l'erreur quadratique moyenne)  
- **R²** (Coefficient de détermination, proportion de variance expliquée par le modèle)  

## Prérequis  
Avant de lancer le projet, assurez-vous d'avoir les dépendances suivantes installées :  
- Python 3
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  
