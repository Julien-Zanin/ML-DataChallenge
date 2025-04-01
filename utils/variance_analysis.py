import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from time import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from utils.data_registry import DATASETS
from utils.features import add_features

from IPython.display import display

 
def evaluate_feature_sets(dataset_key='raw', model_key='xgboost_baseline'):
    """
    Évaluer différents ensembles de features.
    
    Parameters:
    -----------
    dataset_key : str
        Clé du dataset dans DATASETS
    model_key : str
        Clé du modèle à utiliser
    """
    from utils.experiment_runner import run_experiment
    
    print(f"\nÉvaluation des ensembles de features sur {dataset_key} avec {model_key}...")
    
    # Définir différents ensembles de features
    feature_sets_to_test = [
        None,  # Toutes les features
        ["basic_stats"],  # Statistiques de base
        ["technical"],  # Indicateurs techniques
        ["basic_stats", "technical"]  # Combinaison
    ]
    
    results = []
    
    # Tester chaque ensemble de features
    for feature_set in feature_sets_to_test:
        try:
            print(f"\nTest avec feature_set = {feature_set}")
            result = run_experiment(dataset_key=dataset_key, model_key=model_key, 
                                   add_feat=True, feature_sets=feature_set)
            
            set_name = "Toutes" if feature_set is None else ', '.join(feature_set)
            print(f"Accuracy avec {set_name}: {result['accuracy']:.4f}")
            
            results.append({
                'feature_set': set_name,
                'accuracy': result['accuracy'],
                'f1_weighted': result['f1_weighted'],
                'train_time': result['train_time'],
                'total_time': result['total_time'],
                'result': result
            })
            
            # Ajouter également le résultat sans feature engineering pour comparaison
            if feature_set is None:  # Comparer avec le cas sans features uniquement une fois
                result_no_feat = run_experiment(dataset_key=dataset_key, model_key=model_key, 
                                             add_feat=False)
                print(f"Accuracy sans features: {result_no_feat['accuracy']:.4f}")
                
                results.append({
                    'feature_set': 'Aucune (baseline)',
                    'accuracy': result_no_feat['accuracy'],
                    'f1_weighted': result_no_feat['f1_weighted'],
                    'train_time': result_no_feat['train_time'],
                    'total_time': result_no_feat['total_time'],
                    'result': result_no_feat
                })
        
        except Exception as e:
            print(f"Erreur lors de l'évaluation avec feature_set = {feature_set}: {e}")
    
    # Créer un DataFrame des résultats
    results_df = pd.DataFrame([{
        'Feature Set': r['feature_set'],
        'Accuracy': r['accuracy'],
        'F1 Score': r['f1_weighted'],
        'Train Time (s)': r['train_time'],
        'Total Time (s)': r['total_time']
    } for r in results])
    
    # Trier par précision décroissante
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    # Afficher le tableau des résultats
    print("\nRésumé des performances avec différents ensembles de features:")
    display(results_df)
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Feature Set', y='Accuracy', data=results_df)
    plt.title(f'Impact des différents ensembles de features ({dataset_key}, {model_key})')
    plt.xlabel('Ensemble de features')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Visualiser le compromis performance/temps
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Total Time (s)'], results_df['Accuracy'], s=100)
    
    # Ajouter des annotations pour chaque point
    for i, row in results_df.iterrows():
        plt.annotate(row['Feature Set'], 
                    (row['Total Time (s)'], row['Accuracy']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    
    plt.title('Compromis performance/temps pour différents ensembles de features')
    plt.xlabel('Temps total (secondes)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return results

def analyze_feature_importance(dataset_key='raw', use_precomputed_features=True):
    """
    Analyser l'importance des features avec XGBoost.
    
    Parameters:
    -----------
    dataset_key : str
        Clé du dataset dans DATASETS
    use_precomputed_features : bool
        Utiliser le dataset avec features préalculées
    """
    from xgboost import XGBClassifier
    
    try:
        # Déterminer quelle dataset utiliser
        actual_key = f"{dataset_key}_with_features" if use_precomputed_features else dataset_key
        
        if actual_key not in DATASETS:
            if use_precomputed_features:
                print(f"Dataset avec features préalculées '{actual_key}' introuvable. Utilisation de '{dataset_key}' à la place.")
                actual_key = dataset_key
            else:
                raise ValueError(f"Dataset '{dataset_key}' introuvable dans le registre.")
        
        # Charger le dataset
        dataset_info = DATASETS[actual_key]
        X_train = pd.read_csv(dataset_info['train'])
        
        # Mapping pour la variable cible
        mapping = {-1: 0, 0: 1, 1: 2}
        y_train = X_train["reod"].replace(mapping)
        
        # Extraire les features
        non_feature_cols = ['ID', 'day', 'equity', 'reod']
        feature_cols = [col for col in X_train.columns if col not in non_feature_cols]
        
        X_features = X_train[feature_cols].fillna(0)
        
        # Entraîner un modèle XGBoost
        model = XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)
        model.fit(X_features, y_train)
        
        # Récupérer l'importance des features
        importance = model.feature_importances_
        
        # Créer un DataFrame pour visualisation
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Visualiser les 20 features les plus importantes
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Top 20 des features les plus importantes selon XGBoost')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.show()
        
        # Afficher les résultats
        print("\nTop 20 des features les plus importantes:")
        display(importance_df.head(20))
        
        return importance_df, model
    
    except Exception as e:
        print(f"Erreur lors de l'analyse de l'importance des features: {e}")
        import traceback
        traceback.print_exc()  # Affiche la trace complète pour déboguer
        return None, None
    
    
def select_best_features(feature_importance, threshold=0.01, max_features=20):
    """
    Sélectionner les meilleures features basées sur leur importance.
    
    Parameters:
    -----------
    feature_importance : DataFrame
        DataFrame avec les colonnes 'Feature' et 'Importance'
    threshold : float
        Seuil minimal d'importance
    max_features : int
        Nombre maximal de features à sélectionner
    """
    if feature_importance is None:
        print("Aucune information d'importance disponible.")
        return None
    
    # Filtrer par seuil d'importance
    filtered_features = feature_importance[feature_importance['Importance'] > threshold]
    
    # Limiter au nombre maximal de features
    best_features = filtered_features.head(max_features)['Feature'].tolist()
    
    print(f"Sélection de {len(best_features)} features avec importance > {threshold}:")
    for i, feature in enumerate(best_features):
        importance = feature_importance[feature_importance['Feature'] == feature]['Importance'].values[0]
        print(f"{i+1}. {feature}: {importance:.6f}")
    
    return best_features

def summarize_feature_analysis(pca_results, correlation_results, feature_importance, best_features):
    """
    Résumer les principales découvertes de l'analyse des features.
    """
    print("Principales découvertes de l'analyse des features:")
    
    # Insights de la PCA
    if pca_results and all(pca_results):
        _, explained_variance, cumulative_variance, important_features = pca_results
        
        # Nombre de composantes pour différents seuils de variance
        thresholds = [0.7, 0.8, 0.9, 0.95]
        for threshold in thresholds:
            n_components = np.argmax(cumulative_variance >= threshold) + 1
            print(f"- {n_components} composantes principales expliquent {threshold*100:.0f}% de la variance")
        
        # Variables importantes dans la première composante
        if important_features and 0 in important_features:
            top_features = important_features[0][:3]  # Top 3 de la première composante
            print(f"- Les variables les plus importantes dans la première composante sont: {[f[0] for f in top_features]}")
    
    # Insights des corrélations
    if correlation_results and all(correlation_results):
        _, strong_correlations, target_corr = correlation_results
        
        if strong_correlations:
            # Nombre de paires fortement corrélées
            print(f"- {len(strong_correlations)} paires de variables sont fortement corrélées (r > 0.8)")
            
            # Top 3 des paires les plus corrélées
            top_corr = strong_correlations[:3]
            for var1, var2, corr in top_corr:
                print(f"  • {var1} et {var2} ont une corrélation de {corr:.4f}")
        
        if target_corr is not None:
            # Variables les plus corrélées avec la cible
            top_pos = target_corr.head(3).index.tolist()
            top_neg = target_corr.tail(3).index.tolist()
            
            print(f"- Variables les plus positivement corrélées avec la cible: {top_pos}")
            print(f"- Variables les plus négativement corrélées avec la cible: {top_neg}")
    
    # Insights de l'importance des features
    if feature_importance is not None:
        top_features = feature_importance.head(5)['Feature'].tolist()
        print(f"- Les 5 features les plus importantes selon XGBoost sont: {top_features}")
        
        # Contribution cumulée des top features
        top_10_importance = feature_importance.head(10)['Importance'].sum()
        print(f"- Les 10 features les plus importantes représentent {top_10_importance*100:.2f}% de l'importance totale")
    
    # Meilleures features sélectionnées
    if best_features:
        print(f"- {len(best_features)} features ont été sélectionnées pour les modèles optimisés")
        
        # Catégoriser les features sélectionnées
        base_rendements = [f for f in best_features if f.startswith('r') and f[1:].isdigit()]
        derived_stats = [f for f in best_features if f.startswith('r_')]
        
        print(f"  • {len(base_rendements)} rendements de base")
        print(f"  • {len(derived_stats)} features dérivées")
    
    print("\nRecommandations pour la modélisation:")
    if best_features:
        print(f"1. Utiliser les {len(best_features)} features sélectionnées pour les modèles optimisés")
    
    if pca_results and all(pca_results):
        n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1
        print(f"2. Envisager une réduction de dimensionnalité avec PCA ({n_components_90} composantes)")
    
    if correlation_results and all(correlation_results):
        print("3. Éliminer les variables fortement corrélées pour réduire la multicolinéarité")
    
    print("4. Explorer des modèles qui gèrent bien les features fortement corrélées (XGBoost, Régularisation)")
