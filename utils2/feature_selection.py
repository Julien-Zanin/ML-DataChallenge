import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from tqdm import tqdm  # For progress bars
# Optional for visualization:
# from matplotlib_venn import venn2, venn3

def find_important_features(df, target_col='reod', threshold=0.05):
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in DataFrame")
        return []
    
    # Calculate correlations with target
    feature_cols = [col for col in df.columns if col != target_col and col not in ['ID']]
    correlations = df[feature_cols].corrwith(df[target_col])
    
    # Filter features based on threshold
    important_features = correlations[correlations.abs() > threshold].sort_values(ascending=False)
    
    print(f"Found {len(important_features)} features with correlation > {threshold}")
    print("\nTop positively correlated features:")
    print(important_features.head(10))
    print("\nTop negatively correlated features:")
    print(important_features.tail(10))
    
    return important_features.index.tolist()

def optimize_feature_count(X, y, model_factory, feature_ranking, 
                          n_features_range=range(5, 101, 5),
                          scale_method='standard',
                          cv=5):
    """
    Trouve le nombre optimal de features à utiliser pour maximiser la performance du modèle.
    
    Arguments:
        X: DataFrame contenant toutes les features
        y: Series contenant la variable cible
        model_factory: Fonction qui crée une nouvelle instance du modèle
        feature_ranking: Liste des features ordonnées par importance décroissante
        n_features_range: Plage du nombre de features à tester
        scale_method: Méthode de normalisation ('standard', 'robust', 'quantile', None)
        cv: Nombre de folds pour la validation croisée
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from tqdm import tqdm
    
    # Sélectionner le bon scaler selon la méthode demandée
    if scale_method == 'standard':
        scaler = StandardScaler()
    elif scale_method == 'robust':
        scaler = RobustScaler()
    elif scale_method == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal')
    else:
        scaler = None
    
    # Stocker les résultats pour chaque nombre de features
    results = []
    
    # Tester différents nombres de features
    for n_features in tqdm(n_features_range):
        # Sélectionner les top n features
        selected_features = feature_ranking[:n_features]
        X_selected = X[selected_features]
        
        # Créer le pipeline avec ou sans scaler
        if scaler:
            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', model_factory())
            ])
        else:
            pipeline = model_factory()
        
        # Évaluer la performance avec validation croisée
        cv_scores = cross_val_score(pipeline, X_selected, y, cv=cv, scoring='accuracy')
        
        # Stocker le résultat
        results.append({
            'n_features': n_features,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'features': selected_features
        })
    
    # Convertir en DataFrame pour faciliter l'analyse
    results_df = pd.DataFrame(results)
    
    # Trouver le nombre optimal de features
    best_result = results_df.loc[results_df['mean_cv_score'].idxmax()]
    optimal_n_features = int(best_result['n_features'])
    best_score = best_result['mean_cv_score']
    
    print(f"Nombre optimal de features: {optimal_n_features}")
    print(f"Score de validation croisée: {best_score:.4f} ± {best_result['std_cv_score']:.4f}")
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 6))
    plt.errorbar(results_df['n_features'], results_df['mean_cv_score'], 
                 yerr=results_df['std_cv_score'], fmt='o-')
    plt.axvline(x=optimal_n_features, color='r', linestyle='--', 
                label=f'Optimal: {optimal_n_features} features')
    plt.xlabel('Nombre de features')
    plt.ylabel('Score de validation croisée (accuracy)')
    plt.title('Impact du nombre de features sur la performance du modèle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return {
        'optimal_n_features': optimal_n_features,
        'best_score': best_score,
        'optimal_features': best_result['features'],
        'results': results_df
    }
    
def select_features_with_various_methods(X, y, methods=['correlation', 'f_value', 'mutual_info', 'pca']):
    """
    Select features using various methods and compare the results.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    methods : list
        List of feature selection methods to use
        
    Returns:
    --------
    dict
        Dictionary with selected features for each method
    """
    results = {}
    feature_names = X.columns.tolist()
    
    # Method 1: Correlation with target
    if 'correlation' in methods:
        # Convert target to numeric if needed
        y_numeric = pd.to_numeric(y, errors='coerce')
        
        # Calculate correlation with target
        correlations = pd.Series(
            [np.corrcoef(X[col].values, y_numeric.values)[0, 1] for col in feature_names],
            index=feature_names
        )
        
        # Get top features by absolute correlation
        abs_corr = correlations.abs().sort_values(ascending=False)
        top_corr_features = abs_corr.index.tolist()
        
        results['correlation'] = {
            'features': top_corr_features,
            'scores': abs_corr.values,
            'top_positive': correlations.nlargest(10),
            'top_negative': correlations.nsmallest(10)
        }
        
        # Visualize correlation results
        plt.figure(figsize=(12, 6))
        correlations.abs().sort_values().tail(20).plot(kind='barh')
        plt.title('Top 20 Features by Absolute Correlation with Target')
        plt.tight_layout()
        plt.show()
    
    # Method 2: ANOVA F-value
    if 'f_value' in methods:
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        # Get scores and p-values
        f_scores = pd.Series(selector.scores_, index=feature_names)
        p_values = pd.Series(selector.pvalues_, index=feature_names)
        
        # Get top features by F-score
        top_f_features = f_scores.sort_values(ascending=False).index.tolist()
        
        results['f_value'] = {
            'features': top_f_features,
            'scores': f_scores.sort_values(ascending=False).values,
            'p_values': p_values[top_f_features]
        }
        
        # Visualize F-score results
        plt.figure(figsize=(12, 6))
        f_scores.sort_values().tail(20).plot(kind='barh')
        plt.title('Top 20 Features by F-score')
        plt.tight_layout()
        plt.show()
    
    # Method 3: Mutual Information
    if 'mutual_info' in methods:
        selector = SelectKBest(score_func=mutual_info_classif, k='all')
        selector.fit(X, y)
        
        # Get scores
        mi_scores = pd.Series(selector.scores_, index=feature_names)
        
        # Get top features by mutual information
        top_mi_features = mi_scores.sort_values(ascending=False).index.tolist()
        
        results['mutual_info'] = {
            'features': top_mi_features,
            'scores': mi_scores.sort_values(ascending=False).values
        }
        
        # Visualize mutual information results
        plt.figure(figsize=(12, 6))
        mi_scores.sort_values().tail(20).plot(kind='barh')
        plt.title('Top 20 Features by Mutual Information')
        plt.tight_layout()
        plt.show()
    
    # Method 4: PCA
    if 'pca' in methods:
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=min(X.shape[1], 100))  # Limit to 100 components max
        pca.fit(X_scaled)
        
        # Get explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Find number of components needed for different variance thresholds
        variance_thresholds = [0.7, 0.8, 0.9, 0.95]
        components_needed = {}
        
        for threshold in variance_thresholds:
            n_components = np.argmax(cumulative_variance >= threshold) + 1
            components_needed[threshold] = n_components
        
        results['pca'] = {
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'components_needed': components_needed
        }
        
        # Visualize PCA results
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        
        # Add lines for thresholds
        for threshold in variance_thresholds:
            plt.axhline(y=threshold, color='r', linestyle='--')
            plt.text(len(cumulative_variance) * 0.8, threshold + 0.01, 
                    f"{threshold*100}%: {components_needed[threshold]} components")
        
        plt.tight_layout()
        plt.show()
    
    # Compare feature selection methods
    if len(methods) > 1:
        common_methods = [m for m in methods if m != 'pca']
        
        if len(common_methods) > 1:
            method_pairs = [(i, j) for i in range(len(common_methods)) for j in range(i+1, len(common_methods))]
            
            for i, j in method_pairs:
                method1, method2 = common_methods[i], common_methods[j]
                
                # Get top features from each method
                top_features1 = results[method1]['features'][:50]  # Top 50 features
                top_features2 = results[method2]['features'][:50]
                
                # Find common features
                common_features = set(top_features1).intersection(set(top_features2))
                
                print(f"Overlap between {method1} and {method2} (top 50): {len(common_features)} features")
                print(f"Common features: {sorted(list(common_features)[:10])}...")
    
    return results

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


# Méthode 1: Sélection par corrélation avec la cible
def select_by_correlation(X, y, top_n=None, plot=True):
    # Convertir la cible en valeurs numériques si nécessaire
    y_numeric = pd.to_numeric(y, errors='coerce')
    feature_names = X.columns
    
    # Calculer la corrélation de chaque feature avec la cible
    correlations = pd.Series(
        [np.corrcoef(X[col].values, y_numeric.values)[0, 1] for col in feature_names],
        index=feature_names
    )
    
    # Trier par corrélation absolue (positive ou négative)
    abs_corr = correlations.abs().sort_values(ascending=False)
    
    # Sélectionner le nombre de features demandé
    if top_n:
        selected_features = abs_corr.head(top_n).index.tolist()
    else:
        selected_features = abs_corr.index.tolist()
    
    # Visualiser les résultats si demandé
    if plot:
        plt.figure(figsize=(12, 6))
        abs_corr.head(20).plot(kind='barh')
        plt.title('Top 20 features par corrélation absolue avec la cible')
        plt.xlabel('Corrélation absolue')
        plt.tight_layout()
        plt.show()
        
        # Afficher les corrélations positives et négatives
        print("Top corrélations positives:")
        print(correlations.nlargest(10))
        print("\nTop corrélations négatives:")
        print(correlations.nsmallest(10))
    
    return {
        'selected_features': selected_features,
        'correlations': correlations,
        'abs_correlations': abs_corr
    }

# Méthode 2: Sélection par test ANOVA F-value
def select_by_f_value(X, y, top_n=None, plot=True):
    feature_names = X.columns
    
    # Appliquer le test F de l'ANOVA
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    
    # Récupérer les scores et p-values
    f_scores = pd.Series(selector.scores_, index=feature_names)
    p_values = pd.Series(selector.pvalues_, index=feature_names)
    
    # Trier par score F décroissant
    sorted_features = f_scores.sort_values(ascending=False)
    
    # Sélectionner le nombre de features demandé
    if top_n:
        selected_features = sorted_features.head(top_n).index.tolist()
    else:
        selected_features = sorted_features.index.tolist()
    
    # Visualiser les résultats si demandé
    if plot:
        plt.figure(figsize=(12, 6))
        sorted_features.head(20).plot(kind='barh')
        plt.title('Top 20 features par F-score (ANOVA)')
        plt.xlabel('F-score')
        plt.tight_layout()
        plt.show()
        
        # Afficher les p-values pour voir la significativité statistique
        print("Top features par F-score avec leurs p-values:")
        for feature in sorted_features.head(10).index:
            print(f"{feature}: F-score = {f_scores[feature]:.2f}, p-value = {p_values[feature]:.6f}")
    
    return {
        'selected_features': selected_features,
        'f_scores': f_scores,
        'p_values': p_values
    }

# Méthode 3: Sélection par information mutuelle
def select_by_mutual_info(X, y, top_n=None, plot=True):
    feature_names = X.columns
    
    # Appliquer la sélection par information mutuelle
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    selector.fit(X, y)
    
    # Récupérer les scores
    mi_scores = pd.Series(selector.scores_, index=feature_names)
    
    # Trier par score décroissant
    sorted_features = mi_scores.sort_values(ascending=False)
    
    # Sélectionner le nombre de features demandé
    if top_n:
        selected_features = sorted_features.head(top_n).index.tolist()
    else:
        selected_features = sorted_features.index.tolist()
    
    # Visualiser les résultats si demandé
    if plot:
        plt.figure(figsize=(12, 6))
        sorted_features.head(20).plot(kind='barh')
        plt.title('Top 20 features par information mutuelle')
        plt.xlabel('Score d\'information mutuelle')
        plt.tight_layout()
        plt.show()
        
        print("Top features par information mutuelle:")
        print(sorted_features.head(10))
    
    return {
        'selected_features': selected_features,
        'mi_scores': mi_scores
    }

# Fonction pour comparer les résultats des différentes méthodes
# def compare_feature_selection_methods(results_dict, top_n=20):
    
def optimize_feature_count(X, y, model_factory, feature_ranking, 
                          n_features_range=range(5, 101, 5),
                          scale_method='standard',
                          cv=5):
    """
    Trouve le nombre optimal de features à utiliser pour maximiser la performance du modèle.
    
    Arguments:
        X: DataFrame contenant toutes les features
        y: Series contenant la variable cible
        model_factory: Fonction qui crée une nouvelle instance du modèle
        feature_ranking: Liste des features ordonnées par importance décroissante
        n_features_range: Plage du nombre de features à tester
        scale_method: Méthode de normalisation ('standard', 'robust', 'quantile', None)
        cv: Nombre de folds pour la validation croisée
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from tqdm import tqdm
    
    # Sélectionner le bon scaler selon la méthode demandée
    if scale_method == 'standard':
        scaler = StandardScaler()
    elif scale_method == 'robust':
        scaler = RobustScaler()
    elif scale_method == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal')
    else:
        scaler = None
    
    # Stocker les résultats pour chaque nombre de features
    results = []
    
    # Tester différents nombres de features
    for n_features in tqdm(n_features_range):
        # Sélectionner les top n features
        selected_features = feature_ranking[:n_features]
        X_selected = X[selected_features]
        
        # Créer le pipeline avec ou sans scaler
        if scaler:
            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', model_factory())
            ])
        else:
            pipeline = model_factory()
        
        # Évaluer la performance avec validation croisée
        cv_scores = cross_val_score(pipeline, X_selected, y, cv=cv, scoring='accuracy')
        
        # Stocker le résultat
        results.append({
            'n_features': n_features,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'features': selected_features
        })
    
    # Convertir en DataFrame pour faciliter l'analyse
    results_df = pd.DataFrame(results)
    
    # Trouver le nombre optimal de features
    best_result = results_df.loc[results_df['mean_cv_score'].idxmax()]
    optimal_n_features = int(best_result['n_features'])
    best_score = best_result['mean_cv_score']
    
    print(f"Nombre optimal de features: {optimal_n_features}")
    print(f"Score de validation croisée: {best_score:.4f} ± {best_result['std_cv_score']:.4f}")
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 6))
    plt.errorbar(results_df['n_features'], results_df['mean_cv_score'], 
                 yerr=results_df['std_cv_score'], fmt='o-')
    plt.axvline(x=optimal_n_features, color='r', linestyle='--', 
                label=f'Optimal: {optimal_n_features} features')
    plt.xlabel('Nombre de features')
    plt.ylabel('Score de validation croisée (accuracy)')
    plt.title('Impact du nombre de features sur la performance du modèle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return {
        'optimal_n_features': optimal_n_features,
        'best_score': best_score,
        'optimal_features': best_result['features'],
        'results': results_df
    }