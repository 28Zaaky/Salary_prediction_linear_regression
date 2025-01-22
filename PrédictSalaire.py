# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Chargement des données
# On charge les données de salaire en fonction de l'expérience (fichier CSV) dans un DataFrame
data = pd.read_csv('Salary_Data.csv')

# Affichage des premières lignes des données pour un aperçu rapide
print(data.head(), "\n")

# Affichage des informations générales sur le DataFrame (types de colonnes, valeurs manquantes, etc.)
print(data.info())

# Séparation des variables explicatives (X) et de la variable cible (y)
# Ici, X représente l'expérience en années (variable indépendante), et y représente le salaire (variable dépendante)
X = data[['YearsExperience']]  # Notez que X est une matrice à 2 dimensions (même si une seule colonne)
y = data['Salary']

# Visualisation des données
# On crée un graphique en nuage de points (scatter plot) pour visualiser la relation entre l'expérience et le salaire
plt.scatter(X, y)
plt.xlabel('Years of Experience')  # Légende pour l'axe des X (années d'expérience)
plt.ylabel('Salary')  # Légende pour l'axe des Y (salaire)
plt.title('Salary vs. Years of Experience')  # Titre du graphique
plt.show()

# Division des données en ensemble d'entraînement et de test
# 1/3 des données seront utilisées pour tester le modèle, et 2/3 pour l'entraînement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Création du modèle de régression linéaire
# On initialise un modèle de régression linéaire simple
regressor = LinearRegression()

# Entraînement du modèle sur l'ensemble d'entraînement (données X_train et y_train)
regressor.fit(X_train, y_train)

# Affichage des coefficients du modèle de régression linéaire
# Le coefficient indique l'impact d'une unité d'augmentation de X (expérience) sur y (salaire)
# L'intercept représente la valeur de y (salaire) quand X (années d'expérience) est 0
print('Coefficient:', regressor.coef_)
print('Intercept:', regressor.intercept_)

# Tracé de la ligne de régression
# On trace la ligne de régression (prédiction) par-dessus le nuage de points
ordonne = np.linspace(0, 15, 1000)  # Génération de points sur l'axe des X entre 0 et 15 (expérience en années)
plt.scatter(X, y)  # Scatter plot des points réels
plt.plot(ordonne, regressor.predict(ordonne.reshape(-1, 1)), color='r')  # Ligne rouge de la régression
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Regression Line')
plt.show()

# Prédiction des salaires sur l'ensemble de test
# On utilise les données X_test pour prédire les salaires associés (y_predict)
y_predict = regressor.predict(X_test)

# Évaluation du modèle
# On calcule les erreurs du modèle en utilisant différentes métriques
print('MAE:', metrics.mean_absolute_error(y_test, y_predict))  # MAE (Mean Absolute Error) : erreur absolue moyenne
print('MSE:', metrics.mean_squared_error(y_test, y_predict))  # MSE (Mean Squared Error) : erreur quadratique moyenne
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))  # RMSE : racine carrée de l'erreur quadratique moyenne
print('R²:', metrics.r2_score(y_test, y_predict))  # R² : proportion de la variance expliquée par le modèle

# Visualisation des résultats de test (données réelles vs prédictions)
# On compare visuellement les salaires réels et prédits avec un graphique
plt.scatter(X_test, y_test, color='blue', label='Actual')  # Scatter des salaires réels (en bleu)
plt.scatter(X_test, y_predict, color='red', label='Predicted')  # Scatter des salaires prédits (en rouge)
plt.plot(ordonne, regressor.predict(ordonne.reshape(-1, 1)), color='r')  # Ligne de régression en rouge
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Actual vs Predicted Salaries')
plt.legend()  # Ajout de la légende pour distinguer les valeurs réelles et prédites
plt.show()

# Prédiction de nouveaux salaires
# On peut prédire des salaires pour des années d'expérience non présentes dans l'ensemble initial
NewData = [[5], [4]]  # Exemple : prédiction pour 5 ans et 4 ans d'expérience
NewTest = pd.DataFrame(NewData, columns=['YearsExperience'])  # Conversion en DataFrame
y_new = regressor.predict(NewTest)  # Prédiction des salaires correspondants
print('Predicted Salaries for new data:', y_new)  # Affichage des résultats
