import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from IPython.display import display

def optimize_hyperparameters(model, param_grid, X_train, y_train, cv=4, n_iter=15, 
                            scoring='accuracy', use_random=True, verbose=1):
    """
    Optimise les hyperparamètres d'un modèle avec GridSearch ou RandomizedSearch.
    
    Parameters:
    -----------
    model : estimator object
        Le modèle à optimiser
    param_grid : dict
        Grille de paramètres à tester
    X_train : array-like
        Données d'entraînement
    y_train : array-like
        Cibles d'entraînement
    cv : int, default=5
        Nombre de folds pour la validation croisée
    n_iter : int, default=20
        Nombre d'itérations pour RandomizedSearchCV
    scoring : str, default='accuracy'
        Métrique de scoring
    use_random : bool, default=True
        Utiliser RandomizedSearchCV (True) ou GridSearchCV (False)
    verbose : int, default=1
        Niveau de verbosité
        
    Returns:
    --------
    dict
        Dictionnaire contenant le meilleur modèle, les meilleurs paramètres et le temps d'optimisation
    """
    start_time = time.time()
    
    if use_random:
        search = RandomizedSearchCV(
            model, 
            param_distributions=param_grid, 
            n_iter=n_iter,
            cv=cv, 
            scoring=scoring, 
            n_jobs=-1,
            verbose=verbose,
            random_state=42
        )
    else:
        search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            scoring=scoring, 
            n_jobs=-1,
            verbose=verbose
        )
    
    search.fit(X_train, y_train)
    optimization_time = time.time() - start_time
    
    print(f"Meilleurs paramètres: {search.best_params_}")
    print(f"Meilleur score de validation croisée: {search.best_score_:.4f}")
    print(f"Temps d'optimisation: {optimization_time:.2f} secondes")
    
    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "optimization_time": optimization_time
    }

def evaluate_model_performance(model, X_train, y_train, X_test, y_test, cv=5):
    """
    Évalue les performances d'un modèle sur les données d'entraînement et de test.
    
    Parameters:
    -----------
    model : estimator object
        Le modèle à évaluer
    X_train, y_train : arrays
        Données d'entraînement
    X_test, y_test : arrays
        Données de test
    cv : int, default=5
        Nombre de folds pour la validation croisée
        
    Returns:
    --------
    dict
        Dictionnaire contenant les métriques de performance
    """
    results = {}
    
    # Cross-validation sur les données d'entraînement
    start_time = time.time()
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_time = time.time() - start_time
    
    # Entraînement sur l'ensemble complet
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Évaluation sur l'ensemble de test
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time
    
    # Métriques de performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Stocker les résultats
    results = {
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std_accuracy": cv_scores.std(),
        "test_accuracy": accuracy,
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "cv_time": cv_time,
        "train_time": train_time,
        "pred_time": pred_time,
        "total_time": cv_time + train_time + pred_time,
        "report": report
    }
    
    # Afficher les résultats
    print(f"Performances du modèle:")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1 Score pondéré: {report['weighted avg']['f1-score']:.4f}")
    print(f"Temps d'entraînement: {train_time:.2f}s, Prédiction: {pred_time:.2f}s")
    
    return results, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, figsize=(8, 6), title="Matrice de confusion"):
    """
    Affiche une matrice de confusion stylisée.
    
    Parameters:
    -----------
    y_true : array-like
        Vraies étiquettes
    y_pred : array-like
        Prédictions du modèle
    class_names : list, default=None
        Noms des classes
    normalize : bool, default=False
        Normaliser la matrice
    figsize : tuple, default=(8, 6)
        Taille de la figure
    title : str, default="Matrice de confusion"
        Titre du graphique
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Classe prédite')
    plt.ylabel('Classe réelle')
    plt.tight_layout()
    plt.show()

def learning_curve_plot(model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Trace la courbe d'apprentissage d'un modèle.
    
    Parameters:
    -----------
    model : estimator object
        Le modèle à évaluer
    X_train, y_train : arrays
        Données d'entraînement
    cv : int, default=5
        Nombre de folds pour la validation croisée
    train_sizes : array, default=np.linspace(0.1, 1.0, 10)
        Fractions de l'ensemble d'entraînement à utiliser
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, train_sizes=train_sizes,
        scoring='accuracy', n_jobs=-1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Tracer la courbe d'apprentissage
    plt.figure(figsize=(10, 6))
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation croisée")
    
    plt.title("Courbe d'apprentissage")
    plt.xlabel("Taille de l'ensemble d'entraînement")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def create_results_summary(results_tracker, group_by=None, top_n=10):
    """
    Crée un résumé des résultats du tracker.
    
    Parameters:
    -----------
    results_tracker : DataFrame
        DataFrame contenant les résultats
    group_by : list, default=None
        Colonnes pour regrouper les résultats
    top_n : int, default=10
        Nombre de résultats à afficher
        
    Returns:
    --------
    DataFrame
        Résumé des résultats
    """
    if results_tracker.empty:
        print("Aucun résultat à résumer.")
        return None
    
    summary = results_tracker.copy()
    
    # Sélectionner les colonnes d'intérêt
    summary_cols = ["dataset", "dataset_description", "model", "normalize_by_row", 
                   "features_added", "accuracy", "f1_weighted", "total_time"]
    
    for col in summary_cols:
        if col not in summary.columns:
            summary_cols.remove(col)
    
    summary = summary[summary_cols]
    
    # Trier par accuracy décroissante
    summary = summary.sort_values("accuracy", ascending=False)
    
    # Limiter aux top_n résultats
    summary = summary.head(top_n)
    
    # Afficher le résumé
    print(f"Top {top_n} résultats:")
    display(summary)
    
    # Si group_by est spécifié, créer un résumé groupé
    if group_by is not None:
        valid_group_by = [col for col in group_by if col in summary.columns]
        if valid_group_by:
            grouped = summary.groupby(valid_group_by).agg({
                'accuracy': ['mean', 'std', 'max'],
                'f1_weighted': ['mean', 'std', 'max'],
                'total_time': ['mean', 'min']
            }).reset_index()
            
            print(f"\nRésumé groupé par {', '.join(valid_group_by)}:")
            display(grouped)
    
    return summary

def plot_results_comparison(results_tracker, x_axis='model', y_axis='accuracy', hue=None, top_n=15):
    """
    Visualise une comparaison des résultats.
    
    Parameters:
    -----------
    results_tracker : DataFrame
        DataFrame contenant les résultats
    x_axis : str, default='model'
        Colonne à utiliser pour l'axe x
    y_axis : str, default='accuracy'
        Colonne à utiliser pour l'axe y
    hue : str, default=None
        Colonne à utiliser pour la couleur
    top_n : int, default=15
        Nombre de résultats à afficher
    """
    if results_tracker.empty:
        print("Aucun résultat à visualiser.")
        return
    
    # Trier les résultats
    sorted_results = results_tracker.sort_values(y_axis, ascending=False).head(top_n)
    
    plt.figure(figsize=(14, 8))
    
    if hue is not None and hue in sorted_results.columns:
        sns.barplot(x=x_axis, y=y_axis, hue=hue, data=sorted_results)
    else:
        sns.barplot(x=x_axis, y=y_axis, data=sorted_results)
    
    plt.title(f'Comparaison des performances ({y_axis})')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()