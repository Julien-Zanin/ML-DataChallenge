import pandas as pd
import numpy as np

def add_features(df,features_sets = None):
    """
    Add features to the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    feature_sets : list of str, optional
        List of feature sets to add (default: all)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added features
    """
    result_df = df.copy(deep=True)
    
    available_features_sets = {
        "basic_stats": True,
        "technical": True,
    }
    
    if features_sets is not None:
        for key in available_features_sets:
           available_features_sets[key] = key in features_sets
    
    # Sélectionner uniquement les colonnes r0 à r52
    r_cols = [col for col in df.columns if col.startswith('r') and col[1:].isdigit()]
    
    # Vérifier que les colonnes existent et contiennent des données
    if len(r_cols) == 0:
        print("Avertissement: Aucune colonne de rendement trouvée.")
        return result_df
        
    # Vérifier si les données sont bien numériques
    for col in r_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Conversion de la colonne {col} en numérique")
            result_df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if available_features_sets["basic_stats"]:
        # Calculer les statistiques de base
        result_df["r_mean"] = result_df[r_cols].mean(axis=1)
        result_df["r_std"] = result_df[r_cols].std(axis=1)
        result_df["r_min"] = result_df[r_cols].min(axis=1)
        result_df["r_max"] = result_df[r_cols].max(axis=1)
        result_df["r_sum"] = result_df[r_cols].sum(axis=1)
        
        # Compter les valeurs positives/négatives/zéro
        result_df["r_pos_count"] = (result_df[r_cols] > 0).sum(axis=1)
        result_df["r_neg_count"] = (result_df[r_cols] < 0).sum(axis=1) 
        result_df["r_zero_count"] = (result_df[r_cols] == 0).sum(axis=1)
        
        # Somme des valeurs positives/négatives
        result_df['r_pos_sum'] = result_df[r_cols].apply(lambda x: x[x > 0].sum(), axis=1)
        result_df['r_neg_sum'] = result_df[r_cols].apply(lambda x: x[x < 0].sum(), axis=1)
        
    if available_features_sets["technical"]:
        # Calculer les moyennes mobiles
        for window in [5, 10, 20]:
            if window < len(r_cols):
                result_df[f"r_roll_mean_{window}"] = result_df[r_cols].apply(
                    lambda row: row.rolling(window=window).mean().iloc[-1], axis=1)
                result_df[f"r_roll_std_{window}"] = result_df[r_cols].apply(
                    lambda row: row.rolling(window=window).std().iloc[-1], axis=1)
    
        if len(r_cols) > 10:
            result_df["r_momentum_5"] = result_df[r_cols].apply(
                lambda row: row.iloc[-1] - row.iloc[-6] if not pd.isna(row.iloc[-1]) and not pd.isna(row.iloc[-6]) else np.nan, 
                axis=1)
            result_df["r_momentum_10"] = result_df[r_cols].apply(
                lambda row: row.iloc[-1] - row.iloc[-11] if not pd.isna(row.iloc[-1]) and not pd.isna(row.iloc[-11]) else np.nan, 
                axis=1)
    
    return result_df