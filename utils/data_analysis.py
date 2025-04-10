import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.manifold import TSNE
from utils.data_registry import DATASETS
from utils.feature_engineering import add_features
from IPython.display import display



def compare_normalization_impact(original_df, normalized_df, sample_cols=None):

    # Get common columns between both DataFrames
    common_cols = [col for col in original_df.columns if col in normalized_df.columns]
    
    # Select sample columns if not provided
    if sample_cols is None:
        # Focus on rendement columns
        rendement_cols = [col for col in common_cols if col.startswith('r') and col[1:].isdigit()]
        sample_cols = np.random.choice(rendement_cols, min(5, len(rendement_cols)), replace=False)
    
    # Create comparison plots
    for col in sample_cols:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Original data distribution
        sns.histplot(original_df[col].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f'Original: {col}')
        axes[0].axvline(original_df[col].mean(), color='r', linestyle='--', 
                         label=f'Mean: {original_df[col].mean():.2f}')
        axes[0].legend()
        
        # Normalized data distribution
        sns.histplot(normalized_df[col].dropna(), kde=True, ax=axes[1])
        axes[1].set_title(f'Normalized: {col}')
        axes[1].axvline(normalized_df[col].mean(), color='r', linestyle='--',
                        label=f'Mean: {normalized_df[col].mean():.2f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    # Compare correlation structures
    print("Analyzing correlation structure changes...")
    
    # Calculate correlation matrices
    orig_corr = original_df[sample_cols].corr()
    norm_corr = normalized_df[sample_cols].corr()
    
    # Plot them side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(orig_corr, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[0])
    axes[0].set_title('Original Correlation Matrix')
    
    sns.heatmap(norm_corr, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1])
    axes[1].set_title('Normalized Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate the difference in correlation matrices
    diff_corr = norm_corr - orig_corr
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Change in Correlation Matrix (Normalized - Original)')
    plt.tight_layout()
    plt.show()
    
def analyze_distributions(datasets, selected_cols=None):
    """
    Analyse les distributions des rendements pour différentes stratégies d'imputation.
    
    Parameters:
    -----------
    datasets : dict
        Dictionnaire contenant les datasets
    selected_cols : list
        Liste des colonnes à analyser (si None, sélectionne quelques colonnes représentatives)
    """
    if not datasets:
        print("Aucun dataset à analyser.")
        return
    
    # Sélectionner quelques colonnes représentatives si non spécifiées
    if selected_cols is None:
        selected_cols = ['r0', 'r10', 'r25', 'r40', 'r52']
    
    # Comparer les distributions pour chaque colonne sélectionnée
    for col in selected_cols:
        plt.figure(figsize=(15, 10))
        
        # Créer un subplot par stratégie d'imputation
        n_strategies = len(datasets)
        rows = (n_strategies + 1) // 2  # Arrondir vers le haut
        cols = 2 if n_strategies > 1 else 1
        
        for i, (strategy, data) in enumerate(datasets.items()):
            plt.subplot(rows, cols, i+1)
            
            if col in data['train'].columns:
                # Filtrer les valeurs aberrantes pour une meilleure visualisation
                values = data['train'][col].dropna()
                
                # Calculer les limites pour filtrer les valeurs extrêmes
                q1, q3 = values.quantile([0.01, 0.99])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Filtrer pour l'affichage (mais conserver les stats originales)
                filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
                
                # Tracer l'histogramme avec KDE
                sns.histplot(filtered_values, kde=True, bins=50)
                
                # Ajouter des informations statistiques
                mean_val = values.mean()
                median_val = values.median()
                std_val = values.std()
                missing = values.isna().sum() / len(data['train']) * 100
                
                plt.axvline(x=mean_val, color='r', linestyle='--', label=f'Moyenne: {mean_val:.2f}')
                plt.axvline(x=median_val, color='g', linestyle='--', label=f'Médiane: {median_val:.2f}')
                
                plt.title(f'{col} - {strategy} ({data["description"]})')
                plt.xlabel('Rendement (points de base)')
                plt.ylabel('Fréquence')
                plt.legend()
                
                # Ajouter des statistiques sur le graphique
                textstr = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}\nMissing: {missing:.2f}%'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
                        verticalalignment='top', bbox=props)
            else:
                plt.text(0.5, 0.5, f"La colonne {col} n'existe pas dans ce dataset",
                        horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.show()

def compare_column_stats(datasets,rend_cols=None):
    """
    Compare les statistiques des colonnes entre les différentes stratégies d'imputation.
    """
    if not datasets:
        print("Aucun dataset à analyser.")
        return
    
    # Extraire le premier dataset pour obtenir la liste des colonnes de rendement
    first_dataset = next(iter(datasets.values()))['train']
    rendement_cols = [col for col in first_dataset.columns if col.startswith('r') and col[1:].isdigit()]
    
    # Créer un DataFrame pour stocker les statistiques
    stats_data = []
    
    for strategy, data in datasets.items():
        train_df = data['train']
        
        for col in rendement_cols:
            if col in train_df.columns:
                values = train_df[col].dropna()
                stats = {
                    'Stratégie': strategy,
                    'Colonne': col,
                    'Moyenne': values.mean(),
                    'Médiane': values.median(),
                    'Écart-type': values.std(),
                    'Min': values.min(),
                    'Max': values.max(),
                    'Skewness': values.skew(),
                    'Kurtosis': values.kurtosis(),
                    'Missing (%)': train_df[col].isna().sum() / len(train_df) * 100
                }
                stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    
    # Créer des tableaux pivots pour faciliter la comparaison
    pivot_mean = stats_df.pivot(index='Colonne', columns='Stratégie', values='Moyenne')
    pivot_median = stats_df.pivot(index='Colonne', columns='Stratégie', values='Médiane')
    pivot_std = stats_df.pivot(index='Colonne', columns='Stratégie', values='Écart-type')
    pivot_skew = stats_df.pivot(index='Colonne', columns='Stratégie', values='Skewness')
    pivot_kurt = stats_df.pivot(index='Colonne', columns='Stratégie', values='Kurtosis')
    pivot_missing = stats_df.pivot(index='Colonne', columns='Stratégie', values='Missing (%)')
    
    # Analyser l'impact de l'imputation sur la moyenne
    plt.figure(figsize=(14, 7))
    
    # Sélectionner quelques colonnes représentatives
    if rend_cols is  None : 
        sample_cols = ['r0', 'r1', 'r2', 'r3','r4', 'r5','r6', 'r10', 'r25', 'r40','r50','r51', 'r52']
    else : 
        sample_cols = rend_cols
        
    sample_pivot_mean = pivot_mean.loc[sample_cols]
    
    # Créer un heatmap pour visualiser les différences de moyenne
    sns.heatmap(sample_pivot_mean, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Comparaison des moyennes par stratégie d\'imputation')
    plt.tight_layout()
    plt.show()
    
    # Analyser l'impact sur l'écart-type
    plt.figure(figsize=(14, 7))
    sample_pivot_std = pivot_std.loc[sample_cols]
    sns.heatmap(sample_pivot_std, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Comparaison des écarts-types par stratégie d\'imputation')
    plt.tight_layout()
    plt.show()
    
    # Analyser l'impact sur la skewness (asymétrie)
    plt.figure(figsize=(14, 7))
    sample_pivot_skew = pivot_skew.loc[sample_cols]
    sns.heatmap(sample_pivot_skew, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Comparaison de l\'asymétrie (skewness) par stratégie d\'imputation')
    plt.tight_layout()
    plt.show()
    
    return stats_df, {
        'mean': pivot_mean,
        'median': pivot_median,
        'std': pivot_std,
        'skew': pivot_skew,
        'kurt': pivot_kurt,
        'missing': pivot_missing
    }

   
def analyze_normalization(datasets, selected_strategy='ffbf', selected_cols=None):
    """
    Analyse l'impact des différentes méthodes de normalisation.
    
    Parameters:
    -----------
    datasets : dict
        Dictionnaire contenant les datasets
    selected_strategy : str
        Stratégie d'imputation à analyser
    selected_cols : list
        Liste des colonnes à analyser (si None, sélectionne quelques colonnes représentatives)
    """
    if selected_strategy not in datasets:
        print(f"La stratégie {selected_strategy} n'est pas disponible.")
        return
    
    dataset = datasets[selected_strategy]['train']
    
    # Sélectionner quelques colonnes représentatives si non spécifiées
    if selected_cols is None:
        selected_cols = ['r0', 'r10', 'r25', 'r40', 'r52']
    
    # Extraire toutes les colonnes de rendement
    rendement_cols = [col for col in dataset.columns if col.startswith('r') and col[1:].isdigit()]
    
    # Sous-ensemble des données pour les colonnes de rendement
    X_rendements = dataset[rendement_cols].fillna(0)
    
    # Appliquer différentes méthodes de normalisation
    # 1. StandardScaler (normalisation Z-score par colonne)
    standard_scaler = StandardScaler()
    X_standard = pd.DataFrame(
        standard_scaler.fit_transform(X_rendements),
        columns=rendement_cols,
        index=X_rendements.index
    )
    
    # 2. RobustScaler (utilise la médiane et l'IQR, plus robuste aux valeurs aberrantes)
    robust_scaler = RobustScaler()
    X_robust = pd.DataFrame(
        robust_scaler.fit_transform(X_rendements),
        columns=rendement_cols,
        index=X_rendements.index
    )
    
    # 3. QuantileTransformer (transformation vers une distribution normale)
    quantile_transformer = QuantileTransformer(output_distribution='normal')
    X_quantile = pd.DataFrame(
        quantile_transformer.fit_transform(X_rendements),
        columns=rendement_cols,
        index=X_rendements.index
    )
    
    # 4. Normalisation par ligne (personnalisée)
    def normalize_rows(df):
        # Pour chaque ligne, soustraire la moyenne et diviser par l'écart-type
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    
    X_row_normalized = normalize_rows(X_rendements)
    
    # Regrouper les données normalisées
    normalized_data = {
        'Original': X_rendements,
        'StandardScaler': X_standard,
        'RobustScaler': X_robust,
        'QuantileTransformer': X_quantile,
        'Normalisation par ligne': X_row_normalized
    }
    
    # Comparer les distributions avant et après normalisation
    for col in selected_cols:
        plt.figure(figsize=(15, 12))
        
        for i, (method, data) in enumerate(normalized_data.items()):
            plt.subplot(3, 2, i+1)
            
            if col in data.columns:
                sns.histplot(data[col], kde=True, bins=50)
                plt.title(f'{col} - {method}')
                plt.xlabel('Valeur normalisée')
                plt.ylabel('Fréquence')
                
                # Ajouter des statistiques
                mean_val = data[col].mean()
                median_val = data[col].median()
                std_val = data[col].std()
                
                plt.axvline(x=mean_val, color='r', linestyle='--', label=f'Moyenne: {mean_val:.2f}')
                plt.axvline(x=median_val, color='g', linestyle='--', label=f'Médiane: {median_val:.2f}')
                plt.legend()
                
                # Statistiques sur le graphique
                textstr = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
                        verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()
    
    # Analyser l'impact sur les statistiques
    normalization_stats = []
    
    for method, data in normalized_data.items():
        for col in selected_cols:
            stats = {
                'Méthode': method,
                'Colonne': col,
                'Moyenne': data[col].mean(),
                'Médiane': data[col].median(),
                'Écart-type': data[col].std(),
                'Skewness': data[col].skew(),
                'Kurtosis': data[col].kurtosis()
            }
            normalization_stats.append(stats)
    
    normalization_df = pd.DataFrame(normalization_stats)
    
    # Afficher les statistiques dans un tableau pivot
    pivot_normalization = normalization_df.pivot_table(
        index='Colonne',
        columns='Méthode',
        values=['Moyenne', 'Écart-type', 'Skewness', 'Kurtosis']
    )
    
    print("\nStatistiques après normalisation:")
    display(pivot_normalization)
    
    # Visualiser la corrélation entre les variables après normalisation
    plt.figure(figsize=(15, 15))
    
    for i, (method, data) in enumerate(normalized_data.items()):
        plt.subplot(3, 2, i+1)
        
        # Sélectionner un sous-ensemble des colonnes pour la visualisation
        sample_cols = selected_cols
        corr_matrix = data[sample_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title(f'Corrélation après {method}')
        plt.tight_layout()
    
    plt.show()
    
    return normalized_data, normalization_df

def analyze_normalized_dataset(df, feature_groups=None):
    """
    Analyze a normalized dataset, focusing on different groups of features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
    feature_groups : dict, optional
        Dictionary mapping feature group names to lists of columns to analyze
        
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    if feature_groups is None:
        # Auto-detect feature groups
        feature_groups = {
            'rendements': [col for col in df.columns if col.startswith('r') and col[1:].isdigit()],
            'basic_stats': [col for col in df.columns if col.startswith('r_') and not any(x in col for x in ['roll', 'momentum'])],
            'roll_features': [col for col in df.columns if 'roll' in col],
            'momentum_features': [col for col in df.columns if 'momentum' in col]
        }
    
    results = {}
    
    # 1. Basic statistics for each feature group
    stats = {}
    for group_name, columns in feature_groups.items():
        if not columns:
            continue
            
        group_df = df[columns]
        stats[group_name] = {
            'mean': group_df.mean().describe(),
            'std': group_df.std().describe(),
            'missing': group_df.isna().sum().sum(),
            'n_features': len(columns)
        }
    
    results['stats'] = stats
    
    # 2. Correlation with target (if available)
    if 'reod' in df.columns:
        target_correlations = {}
        for group_name, columns in feature_groups.items():
            correlations = df[columns].corrwith(df['reod'])
            target_correlations[group_name] = {
                'strongest_positive': correlations.nlargest(3),
                'strongest_negative': correlations.nsmallest(3),
                'mean_abs_corr': correlations.abs().mean()
            }
        
        results['target_correlations'] = target_correlations
    
    # 3. Plot distribution of feature values for each group
    fig_dict = {}
    for group_name, columns in feature_groups.items():
        if not columns:
            continue
            
        # Take a sample of columns if there are too many
        sample_cols = columns[:5] if len(columns) > 5 else columns
        
        fig, axes = plt.subplots(len(sample_cols), 1, figsize=(12, 3*len(sample_cols)))
        fig.suptitle(f'Distribution of {group_name} features')
        
        if len(sample_cols) == 1:
            axes = [axes]  # Make it iterable when there's only one subplot
            
        for i, col in enumerate(sample_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'{col} (mean={df[col].mean():.2f}, std={df[col].std():.2f})')
            axes[i].axvline(df[col].mean(), color='r', linestyle='--')
        
        plt.tight_layout()
        fig_dict[group_name] = fig
    
    results['figures'] = fig_dict
    
    # 4. Feature correlations within groups
    correlation_matrices = {}
    for group_name, columns in feature_groups.items():
        if len(columns) < 2:  # Need at least 2 features for correlation
            continue
            
        # Take a sample if there are too many columns
        sample_cols = columns[:10] if len(columns) > 10 else columns
        corr_matrix = df[sample_cols].corr()
        correlation_matrices[group_name] = corr_matrix
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   annot=True if len(sample_cols) <= 10 else False, 
                   fmt='.2f', square=True)
        plt.title(f'Correlation Matrix: {group_name}')
        plt.tight_layout()
        
    results['correlation_matrices'] = correlation_matrices
    
    return results

def perform_pca_analysis(dataset_key='raw', sample_size=10000):
    """
    Effectuer une analyse PCA sur un dataset spécifié.
    
    Parameters:
    -----------
    dataset_key : str
        Clé du dataset dans DATASETS
    sample_size : int
        Taille de l'échantillon pour l'analyse
    """
    print(f"\nAnalyse PCA sur le dataset {dataset_key}...")
    
    try:
        # Charger le dataset
        start_time = time()
        dataset_info = DATASETS[dataset_key]
        X_train = pd.read_csv(dataset_info['train'])
        load_time = time() - start_time
        print(f"Temps de chargement: {load_time:.2f} secondes")
        
        # Échantillonner les données si nécessaire
        if sample_size and len(X_train) > sample_size:
            X_sample = X_train.sample(sample_size, random_state=42)
        else:
            X_sample = X_train
        
        # Extraire les features (colonnes de rendement)
        rendement_cols = [col for col in X_sample.columns if col.startswith('r') and col[1:].isdigit()]
        X_rendements = X_sample[rendement_cols].fillna(0)
        
        # Standardiser les données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_rendements)
        
        # Appliquer PCA
        start_time = time()
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        pca_time = time() - start_time
        print(f"Temps d'exécution PCA: {pca_time:.2f} secondes")
        
        # Analyser la variance expliquée
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Déterminer le nombre de composantes pour différents seuils de variance expliquée
        thresholds = [0.7, 0.8, 0.9, 0.95, 0.99]
        components_needed = {}
        
        for threshold in thresholds:
            n_components = np.argmax(cumulative_variance >= threshold) + 1
            components_needed[threshold] = n_components
            print(f"Composantes nécessaires pour {threshold*100}% de variance expliquée: {n_components}")
        
        # Visualiser la variance expliquée
        plt.figure(figsize=(18, 12))
        plt.subplot(2, 1, 1)
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        plt.title('Variance expliquée par composante')
        plt.xlabel('Composante')
        plt.ylabel('Variance expliquée')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
        plt.title('Variance expliquée cumulée')
        plt.xlabel('Nombre de composantes')
        plt.ylabel('Variance expliquée cumulée')
        plt.grid(True)
        
        # Ajouter des lignes horizontales pour les seuils
        for threshold in thresholds:
            plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
            plt.text(len(cumulative_variance) * 0.7, threshold - 0.02, f'{threshold*100}%')
        
        plt.tight_layout()
        plt.show()
        
        # Examiner les 2 premières composantes principales
        if 'reod' in X_sample.columns:
            plt.figure(figsize=(10, 8))
            colors = {-1: 'red', 0: 'blue', 1: 'green'}
            for label in sorted(X_sample['reod'].unique()):
                idx = X_sample['reod'] == label
                plt.scatter(X_pca[idx, 0], X_pca[idx, 1], 
                            c=colors.get(label, 'gray'), 
                            label=f'Classe {label}', 
                            alpha=0.6)
            
            plt.title('Projection PCA sur les 2 premières composantes')
            plt.xlabel('Composante principale 1')
            plt.ylabel('Composante principale 2')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        # Analyser les loadings (contributions) des variables d'origine
        loadings = pca.components_
        
        # Afficher les contributions des 10 premières composantes
        n_display = min(10, len(loadings))
        
        plt.figure(figsize=(15, 50))
        for i in range(n_display):
            plt.subplot(n_display, 1, i+1)
            plt.bar(range(len(rendement_cols)), loadings[i], alpha=0.7)
            plt.title(f'Loadings - Composante {i+1} (Variance: {explained_variance[i]:.4f})')
            plt.xlabel('Variable d\'origine (index)')
            plt.ylabel('Contribution')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Trouver les variables les plus contributives
        most_important_features = {}
        
        for i in range(n_display):
            component = loadings[i]
            sorted_indices = np.argsort(np.abs(component))[::-1]
            top_indices = sorted_indices[:5]  # Top 5 features
            
            most_important_features[i] = [
                (rendement_cols[idx], component[idx]) for idx in top_indices
            ]
        
        # Afficher les variables les plus contributives
        print("\nVariables les plus contributives par composante:")
        for comp, features in most_important_features.items():
            print(f"\nComposante {comp+1}:")
            for feature, loading in features:
                print(f"  {feature}: {loading:.4f}")
        
        return pca, explained_variance, cumulative_variance, most_important_features
    
    except Exception as e:
        print(f"Erreur lors de l'analyse PCA: {e}")
        return None, None, None, None
    
def perform_tsne_analysis(dataset_key='raw', sample_size=5000, perplexity=30, n_iter=1000):
    """
    Effectuer une analyse t-SNE sur un dataset spécifié.
    
    Parameters:
    -----------
    dataset_key : str
        Clé du dataset dans DATASETS
    sample_size : int
        Taille de l'échantillon pour l'analyse
    perplexity : float
        Paramètre de perplexité pour t-SNE
    n_iter : int
        Nombre d'itérations
    """
    print(f"\nAnalyse t-SNE sur le dataset {dataset_key}...")
    
    try:
        # Charger le dataset
        start_time = time()
        dataset_info = DATASETS[dataset_key]
        X_train = pd.read_csv(dataset_info['train'])
        load_time = time() - start_time
        print(f"Temps de chargement: {load_time:.2f} secondes")
        
        # Échantillonner les données (t-SNE est très gourmand en ressources)
        if sample_size and len(X_train) > sample_size:
            X_sample = X_train.sample(sample_size, random_state=42)
        else:
            X_sample = X_train
        
        # Extraire les features (colonnes de rendement)
        rendement_cols = [col for col in X_sample.columns if col.startswith('r') and col[1:].isdigit()]
        X_rendements = X_sample[rendement_cols].fillna(0)
        
        # Standardiser les données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_rendements)
        
        # Appliquer t-SNE
        start_time = time()
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        tsne_time = time() - start_time
        print(f"Temps d'exécution t-SNE: {tsne_time:.2f} secondes")
        
        # Visualiser les résultats
        if 'reod' in X_sample.columns:
            plt.figure(figsize=(10, 8))
            colors = {-1: 'red', 0: 'blue', 1: 'green'}
            for label in sorted(X_sample['reod'].unique()):
                idx = X_sample['reod'] == label
                plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], 
                            c=colors.get(label, 'gray'), 
                            label=f'Classe {label}', 
                            alpha=0.6)
            
            plt.title(f't-SNE - Perplexité: {perplexity}, Itérations: {n_iter}')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        return X_tsne, X_sample['reod'] if 'reod' in X_sample.columns else None
    
    except Exception as e:
        print(f"Erreur lors de l'analyse t-SNE: {e}")
        return None, None

def analyze_correlations(dataset_key='raw', sample_size=50000, correlation_threshold=0.8):
    """
    Analyser les corrélations entre les features.
    
    Parameters:
    -----------
    dataset_key : str
        Clé du dataset dans DATASETS
    sample_size : int
        Taille de l'échantillon pour l'analyse
    correlation_threshold : float
        Seuil pour considérer une corrélation comme forte
    """
    print(f"\nAnalyse des corrélations sur le dataset {dataset_key}...")
    
    try:
        # Charger le dataset
        start_time = time()
        dataset_info = DATASETS[dataset_key]
        X_train = pd.read_csv(dataset_info['train'])
        load_time = time() - start_time
        print(f"Temps de chargement: {load_time:.2f} secondes")
        
        # Échantillonner les données si nécessaire
        if sample_size and len(X_train) > sample_size:
            X_sample = X_train.sample(sample_size, random_state=42)
        else:
            X_sample = X_train
        
        # Extraire les features (colonnes de rendement)
        rendement_cols = [col for col in X_sample.columns if col.startswith('r') and col[1:].isdigit()]
        X_rendements = X_sample[rendement_cols].fillna(0)
        
        # Calculer la matrice de corrélation
        start_time = time()
        corr_matrix = X_rendements.corr()
        corr_time = time() - start_time
        print(f"Temps de calcul des corrélations: {corr_time:.2f} secondes")
        
        # Visualiser la matrice de corrélation
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Matrice de corrélation des rendements')
        plt.tight_layout()
        plt.show()
        
        # Identifier les paires de variables fortement corrélées
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        strong_correlations = [(col1, col2, corr_matrix.loc[col1, col2]) 
                              for col1 in upper_tri.index 
                              for col2 in upper_tri.columns 
                              if abs(upper_tri.loc[col1, col2]) > correlation_threshold]
        
        # Trier par force de corrélation
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Afficher les corrélations fortes
        print(f"\nPaires de variables avec corrélation > {correlation_threshold}:")
        correlation_df = pd.DataFrame(strong_correlations, columns=['Variable 1', 'Variable 2', 'Corrélation'])
        display(correlation_df.head(20))
        
        # Corrélation avec la variable cible
        if 'reod' in X_sample.columns:
            target_corr = X_rendements.corrwith(X_sample['reod'])
            target_corr = target_corr.sort_values(ascending=False)
            
            plt.figure(figsize=(12, 8))
            target_corr.plot(kind='bar')
            plt.title('Corrélation entre les rendements et la variable cible')
            plt.xlabel('Variable')
            plt.ylabel('Corrélation avec reod')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            # Afficher les variables les plus corrélées avec la cible
            print("\nVariables les plus corrélées positivement avec la cible:")
            display(target_corr.head(10))
            
            print("\nVariables les plus corrélées négativement avec la cible:")
            display(target_corr.tail(10))
        
        return corr_matrix, strong_correlations, target_corr if 'reod' in X_sample.columns else None
    
    except Exception as e:
        print(f"Erreur lors de l'analyse des corrélations: {e}")
        return None, None, None


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

