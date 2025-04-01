import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from IPython.display import display

def visualize_results(results_df):
    """
    Visualize experiment results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with experiment results
    """
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Compare accuracy across datasets for each model
    plt.figure(figsize=(14, 8))
    sns.barplot(x="dataset", y="accuracy", hue="model", data=results_df)
    plt.title("Model Accuracy by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("accuracy_by_dataset.png")
    
    # 2. Compare accuracy with and without feature engineering
    plt.figure(figsize=(14, 8))
    sns.barplot(x="model", y="accuracy", hue="features_added", data=results_df)
    plt.title("Impact of Feature Engineering")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("feature_engineering_impact.png")
    
    # 3. Heatmap of best combinations
    pivot_df = results_df.pivot_table(
        values="accuracy", 
        index="dataset", 
        columns="model", 
        aggfunc="max"
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Best Accuracy for Each Dataset-Model Combination")
    plt.tight_layout()
    plt.savefig("best_combinations_heatmap.png")
    
    # 4. Feature importance analysis (if applicable)
    # This would need to be implemented separately for each model type
    
    plt.close('all')


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
    if rend_cols is not None : 
        sample_cols = ['r0', 'r10', 'r25', 'r40', 'r52']
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