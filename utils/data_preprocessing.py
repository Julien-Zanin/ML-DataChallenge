
import pandas as pd
import numpy as np
from utils.data_registry import DATASETS, add_dataset_with_features
from utils.features import add_features
from sklearn.impute import KNNImputer


def precompute_datasets_with_features():
    """
    Précompute et sauvegarde les datasets avec features ajoutées.
    """
    for dataset_key in ['raw', 'ffbf', 'bfff', 'interp', 'knn', 'mice']:
        if dataset_key in DATASETS:
            try:
                print(f"\nTraitement du dataset {dataset_key}...")
                
                # Charger le dataset
                dataset_info = DATASETS[dataset_key]
                X_train = pd.read_csv(dataset_info["train"])
                X_test = pd.read_csv(dataset_info["test"])
                
                # Trier par ID
                X_train = X_train.sort_values(by="ID")
                X_test = X_test.sort_values(by="ID")
                
                # Ajouter toutes les features
                print("Ajout des features...")
                X_train_with_features = add_features(X_train)
                X_test_with_features = add_features(X_test)
                
                # Sauvegarder
                output_train_path = f"processed_data/preprocessed/X_train_{dataset_key}_with_features.csv"
                output_test_path = f"processed_data/preprocessed/X_test_{dataset_key}_with_features.csv"
                
                X_train_with_features.to_csv(output_train_path, index=False)
                X_test_with_features.to_csv(output_test_path, index=False)
                
                print(f"Datasets sauvegardés: {output_train_path}, {output_test_path}")
                
                # Ajouter au registre
                new_key = f"{dataset_key}_with_features"
                new_description = f"{dataset_info['description']} avec features ajoutées"
                add_dataset_with_features(new_key, output_train_path, output_test_path, new_description)
                
            except Exception as e:
                print(f"Erreur lors du traitement du dataset {dataset_key}: {e}")
    
    print("\nRegistre DATASETS mis à jour:")
    for key, info in DATASETS.items():
        if "_with_features" in key:
            print(f"- {key}: {info['description']}")

def forward_fill_imputation(df, columns_to_fill=None, axis=1):
    """Forward fill imputation for DataFrame."""
    df_filled = df.copy(deep=True)
    
    if columns_to_fill is not None:
        # Apply forward fill to specified columns only
        df_filled[columns_to_fill] = df_filled[columns_to_fill].ffill(axis=axis)
    else:
        # Apply forward fill to entire DataFrame
        df_filled = df_filled.ffill(axis=axis)
    
    return df_filled


def backward_fill_imputation(df, columns_to_fill=None, axis=1):
    """Backward fill imputation for DataFrame."""
    df_filled = df.copy(deep=True)
    
    if columns_to_fill is not None:
        # Apply backward fill to specified columns only
        df_filled[columns_to_fill] = df_filled[columns_to_fill].bfill(axis=axis)
    else:
        # Apply backward fill to entire DataFrame
        df_filled = df_filled.bfill(axis=axis)
    
    return df_filled


def mixed_directional_fill(df, columns_to_fill=None, fill_method='ffill_then_bfill', axis=1):
    """Combined forward/backward fill imputation."""
    df_filled = df.copy(deep=True)
    
    if fill_method == 'ffill_then_bfill':
        df_filled = forward_fill_imputation(df_filled, columns_to_fill, axis)
        df_filled = backward_fill_imputation(df_filled, columns_to_fill, axis)
    elif fill_method == 'bfill_then_ffill':
        df_filled = backward_fill_imputation(df_filled, columns_to_fill, axis)
        df_filled = forward_fill_imputation(df_filled, columns_to_fill, axis)
    else:
        raise ValueError("fill_method must be either 'ffill_then_bfill' or 'bfill_then_ffill'")
    
    return df_filled


def linear_interpolation(df, columns_to_fill=None, axis=1):
    """Linear interpolation with edge filling."""
    df_filled = df.copy(deep=True)
    
    if columns_to_fill is not None:
        # Apply interpolation to specified columns
        df_filled[columns_to_fill] = df_filled[columns_to_fill].interpolate(method='linear', axis=axis)
        # Fill remaining NaNs at edges
        df_filled[columns_to_fill] = df_filled[columns_to_fill].ffill(axis=axis).bfill(axis=axis)
    else:
        # Apply interpolation to entire DataFrame
        df_filled = df_filled.interpolate(method='linear', axis=axis)
        df_filled = df_filled.ffill(axis=axis).bfill(axis=axis)
    
    return df_filled


def impute_group(group, features_cols):
    # Select only feature columns
    group_features = group[features_cols]
    
    # Standardization at group level
    group_features_std, group_stats = standardize(group_features)
    
    # Imputation on standardized group
    imputer_group = KNNImputer(n_neighbors=5)
    group_imputed_std = imputer_group.fit_transform(group_features_std)
    
    # Create DataFrame with matching columns from the imputation result
    # Use the actual columns from the imputation result rather than features_cols
    group_imputed_std_df = pd.DataFrame(
        group_imputed_std, 
        index=group.index
    )
    
    # Assign the actual column names from the original features
    group_imputed_std_df.columns = group_features.columns
    
    # Inversion of standardization
    group_imputed = inverse_standardize(group_imputed_std_df, group_stats)
    
    # Replace imputed columns in the original group
    group_result = group.copy()
    group_result[features_cols] = group_imputed
    
    return group_result

def standardize(df):
    df_std = df.copy()
    stats = {}
    
    for col in df.columns: 
        mean = df[col].mean()
        std = df[col].std()
        stats[col] = {"mean":mean,"std":std}
        df_std[col] = (df[col] - mean) / std
    return df_std, stats

def inverse_standardize(df_std, stats):
    df_origin = df_std.copy()
    
    for col in df_std.columns:
        mean, std= stats[col]["mean"], stats[col]["std"]
        df_origin[col] = df_std[col] * std + mean
    return df_origin


def normalize_rendements_by_row(df):
    """
    Normalise uniquement les colonnes de rendement par ligne.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Le dataframe contenant les données
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe avec les rendements normalisés
    """
    df_normalized = df.copy(deep=True)
    
    # Identifier les colonnes de rendement (r0 à r52)
    rendement_cols = [col for col in df.columns if col.startswith('r') and col[1:].isdigit()]
    
    if rendement_cols:
        # Extraire le sous-dataframe des rendements
        rendements = df[rendement_cols]
        
        # Calculer moyenne et écart-type par ligne
        row_means = rendements.mean(axis=1)
        row_stds = rendements.std(axis=1)
        
        # Normaliser les rendements (avec epsilon pour éviter la division par zéro)
        normalized_rendements = rendements.sub(row_means, axis=0).div(row_stds + 1e-8, axis=0)
        
        # Remplacer les NaN par 0
        normalized_rendements.fillna(0, inplace=True)
        
        # Mettre à jour le dataframe original avec les rendements normalisés
        df_normalized[rendement_cols] = normalized_rendements
    
    return df_normalized
