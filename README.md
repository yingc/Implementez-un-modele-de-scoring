# P7_Implementez-un-modele-de-scoring

Scoring de risque crédit
Projet réalisé dans le cadre de la formation Data Scientist d'OpenClassrooms (titre RNCP niveau 7).
Contexte
La société financière « Prêt à dépenser » propose des crédits à la consommation à des personnes ayant peu ou pas d'historique de prêt. Elle a besoin :
d'un modèle qui estime la probabilité qu'un client ne rembourse pas son crédit ;
d'un tableau de bord qui permette aux chargés de relation client d'expliquer la décision au client, en particulier en cas de refus.
Données
Données publiques Home Credit Default Risk (Kaggle) :
307 507 clients dans le jeu d'entraînement ;
798 variables après agrégation des différentes tables (demandes de crédit, historique auprès d'autres organismes, crédits précédents, paiements) ;
classes très déséquilibrées : environ 8 % de clients en défaut de paiement.
La préparation des variables s'appuie sur le kernel Kaggle public LightGBM with Simple Features.
Démarche
Préparation des données : jointure et agrégation des tables, traitement des valeurs aberrantes et manquantes.
Comparaison de modèles sur un échantillon de 20 % des données (score AUC) :
Modèle
AUC
Référence aléatoire (Dummy)
0,51
Arbre de décision
0,55
Random Forest
0,71
AdaBoost
0,75
CatBoost
0,77
Modèle retenu : LightGBM, entraîné sur l'ensemble des données et évalué par validation croisée à 10 plis.
Déséquilibre des classes : essai de rééquilibrage avec SMOTE.
Seuil de décision métier : un défaut non détecté (faux négatif) coûte plus cher qu'un refus à tort (faux positif). Le seuil est choisi en minimisant la fonction de coût 10 × faux négatifs + faux positifs, ce qui conduit à un seuil d'environ 0,17 au lieu de 0,5.
Interprétabilité : importance globale des variables et explication de chaque décision individuelle avec SHAP.
Résultats
AUC de 0,79 en validation croisée à 10 plis pour le modèle LightGBM.
AUC de 0,78 avec rééquilibrage SMOTE, qui n'améliore donc pas le modèle.
Un seuil de décision abaissé à 0,17 pour limiter les défauts non détectés.
Tableau de bord
L'application Streamlit (app.py) permet, pour un client donné :
d'afficher la décision d'accord ou de refus du crédit ;
de visualiser les variables qui ont le plus pesé dans la décision (graphique SHAP) ;
de comparer le client à des clients similaires (regroupement K-Means) ;
de consulter la description de chaque variable.
Lancer l'application en local :
pip install -r requirements.txt
streamlit run app.py
​
Contenu du dépôt
Fichier
Rôle
P7_01_kenel_model.ipynb
Préparation des variables et modèle LightGBM
P7_02_autre_modele_20%.ipynb
Comparaison de modèles, seuil métier, SHAP
P7_3_preprocessing_app.ipynb
Préparation des données et du modèle allégé pour l'application
app.py
Tableau de bord Streamlit
LightGBMC.joblib
Modèle entraîné
Outils
Python, pandas, scikit-learn, LightGBM, CatBoost, SHAP, Streamlit.
Pistes d'amélioration
Choisir le seuil de décision sur un jeu de validation que le modèle n'a jamais vu pendant l'entraînement.
Réduire le nombre de variables pour alléger le modèle et faciliter son explication.
![image](https://user-images.githubusercontent.com/6079611/201625677-b18337d1-60d9-4567-b670-15a21bc65ecb.png)
