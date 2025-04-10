import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

def analyze_feature_importance(model, X, feature_names=None, method='builtin', top_n=20):
    """
    Analyse l'importance des features pour un modèle donné.
    
    Parameters:
    -----------
    model : Modèle entraîné avec une méthode feature_importances_ ou coef_
    X : DataFrame ou array, données d'entrée
    feature_names : Liste des noms de features (si X n'est pas un DataFrame)
    method : Méthode d'analyse ('builtin', 'permutation')
    top_n : Nombre de features importantes à afficher
    
    Returns:
    --------
    DataFrame contenant les scores d'importance pour chaque feature
    """
    if feature_names is None:
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Déterminer la méthode d'importance selon le modèle
    if method == 'builtin':
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim > 1:
                importance = np.abs(model.coef_).mean(axis=0)
            else:
                importance = np.abs(model.coef_)
        else:
            print("Le modèle ne possède pas d'attribut feature_importances_ ou coef_, utilisation de permutation_importance.")
            method = 'permutation'
    
    # Utiliser permutation importance si nécessaire
    if method == 'permutation':
        try:
            perm_importance = permutation_importance(model, X, random_state=42, n_repeats=10)
            importance = perm_importance.importances_mean
        except Exception as e:
            print(f"Erreur lors du calcul de permutation_importance: {e}")
            return pd.DataFrame()
    
    # Créer DataFrame avec les importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Afficher les résultats
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
    plt.title(f'Top {top_n} Features selon {method.capitalize()} Importance')
    plt.tight_layout()
    plt.show()
    
    return importance_df

def analyze_with_shap(model, X, feature_names=None, max_display=20, sample_size=500):
    """
    Analyse le modèle avec SHAP pour interpréter les prédictions.
    
    Parameters:
    -----------
    model : Modèle entraîné
    X : DataFrame ou array, données d'entrée
    feature_names : Liste des noms de features (si X n'est pas un DataFrame)
    max_display : Nombre maximum de features à afficher
    sample_size : Taille de l'échantillon à utiliser (pour les grands datasets)
    
    Returns:
    --------
    Tuple contenant les valeurs SHAP et l'explainer
    """
    try:
        import shap
    except ImportError:
        print("La bibliothèque SHAP n'est pas installée. Exécutez 'pip install shap' pour l'installer.")
        return None, None
    
    if feature_names is None and hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
    
    # Échantillonner si nécessaire
    if sample_size and sample_size < len(X):
        if hasattr(X, 'sample'):
            X_sample = X.sample(sample_size, random_state=42)
        else:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
    else:
        X_sample = X
    
    try:
        # Choisir l'explainer approprié selon le type de modèle
        if hasattr(model, 'feature_importances_'):  # Tree-based models (XGBoost, RandomForest)
            explainer = shap.TreeExplainer(model)
        else:
            # Pour les autres modèles, utiliser KernelExplainer avec un sous-échantillon
            background = shap.sample(X_sample, min(100, len(X_sample)))
            explainer = shap.KernelExplainer(model.predict, background)
        
        # Calculer les valeurs SHAP
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values, explainer
    
    except Exception as e:
        print(f"Erreur lors de l'analyse SHAP: {e}")
        return None, None

def plot_shap_summary(shap_values, X, feature_names=None, max_display=20, plot_type="bar"):
    """
    Visualise le résumé des valeurs SHAP.
    
    Parameters:
    -----------
    shap_values : Valeurs SHAP calculées par analyze_with_shap
    X : DataFrame ou array, données d'entrée
    feature_names : Liste des noms de features (si X n'est pas un DataFrame)
    max_display : Nombre maximum de features à afficher
    plot_type : Type de visualisation ('bar', 'dot', ou 'violin')
    """
    try:
        import shap
    except ImportError:
        print("La bibliothèque SHAP n'est pas installée. Exécutez 'pip install shap' pour l'installer.")
        return
    
    if feature_names is not None and hasattr(X, 'columns'):
        X_display = X.copy()
        X_display.columns = feature_names
    else:
        X_display = X
    
    plt.figure(figsize=(12, 10))
    
    if plot_type == "bar":
        shap.summary_plot(shap_values, X_display, plot_type="bar", max_display=max_display, show=False)
        plt.title('Importance des Features selon SHAP (Bar Plot)')
    elif plot_type == "violin":
        shap.summary_plot(shap_values, X_display, plot_type="violin", max_display=max_display, show=False)
        plt.title('Importance des Features selon SHAP (Violin Plot)')
    else:  # dot plot est le défaut
        shap.summary_plot(shap_values, X_display, max_display=max_display, show=False)
        plt.title('Impact des Features sur la Prédiction selon SHAP')
    
    plt.tight_layout()
    plt.show()

def plot_shap_dependence(shap_values, X, feature_idx, interaction_idx=None, feature_names=None):
    """
    Visualise la dépendance entre une feature et ses valeurs SHAP.
    
    Parameters:
    -----------
    shap_values : Valeurs SHAP calculées par analyze_with_shap
    X : DataFrame ou array, données d'entrée
    feature_idx : Index ou nom de la feature à analyser
    interaction_idx : Index ou nom de la feature d'interaction (optionnel)
    feature_names : Liste des noms de features (si X n'est pas un DataFrame)
    """
    try:
        import shap
    except ImportError:
        print("La bibliothèque SHAP n'est pas installée. Exécutez 'pip install shap' pour l'installer.")
        return
    
    if feature_names is not None and hasattr(X, 'columns'):
        X_display = X.copy()
        X_display.columns = feature_names
    else:
        X_display = X
    
    plt.figure(figsize=(10, 8))
    
    if interaction_idx is not None:
        shap.dependence_plot(feature_idx, shap_values, X_display, interaction_index=interaction_idx, show=False)
        if isinstance(feature_idx, str) and isinstance(interaction_idx, str):
            plt.title(f'Dépendance SHAP: {feature_idx} avec interaction {interaction_idx}')
        else:
            plt.title(f'Dépendance SHAP: Feature {feature_idx} avec interaction {interaction_idx}')
    else:
        shap.dependence_plot(feature_idx, shap_values, X_display, show=False)
        if isinstance(feature_idx, str):
            plt.title(f'Dépendance SHAP: {feature_idx}')
        else:
            plt.title(f'Dépendance SHAP: Feature {feature_idx}')
    
    plt.tight_layout()
    plt.show()

def plot_lime_explanation(model, X, instance_idx, feature_names=None, num_features=10, class_names=None):
    """
    Visualise l'explication LIME pour une instance spécifique.
    
    Parameters:
    -----------
    model : Modèle entraîné
    X : DataFrame ou array, données d'entrée
    instance_idx : Index de l'instance à expliquer
    feature_names : Liste des noms de features (si X n'est pas un DataFrame)
    num_features : Nombre de features à inclure dans l'explication
    class_names : Noms des classes pour les modèles de classification
    """
    try:
        import lime
        from lime import lime_tabular
    except ImportError:
        print("La bibliothèque LIME n'est pas installée. Exécutez 'pip install lime' pour l'installer.")
        return
    
    if feature_names is None and hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    # Créer l'explainer LIME
    explainer = lime_tabular.LimeTabularExplainer(
        X_array,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True
    )
    
    # Obtenir l'instance à expliquer
    if isinstance(X, pd.DataFrame):
        instance = X.iloc[instance_idx].values
    else:
        instance = X[instance_idx]
    
    # Générer l'explication
    if hasattr(model, 'predict_proba'):
        explanation = explainer.explain_instance(
            instance, 
            model.predict_proba,
            num_features=num_features
        )
    else:
        explanation = explainer.explain_instance(
            instance, 
            model.predict,
            num_features=num_features
        )
    
    # Visualiser l'explication
    plt.figure(figsize=(10, 8))
    explanation.as_pyplot_figure()
    plt.title(f'Explication LIME pour l\'instance {instance_idx}')
    plt.tight_layout()
    plt.show()
    
    return explanation