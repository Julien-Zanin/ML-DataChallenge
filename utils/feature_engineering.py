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

def add_financial_features(df):
    """
    Ajoute des features financières avancées au dataframe.
    """
    result_df = df.copy(deep=True)
    
    # Identifier les colonnes de rendement
    r_cols = [col for col in df.columns if col.startswith('r') and col[1:].isdigit()]
    
    if len(r_cols) == 0:
        print("Aucune colonne de rendement trouvée.")
        return result_df
    
    # 1. Volatilité réalisée (sur différentes fenêtres)
    for window in [5, 10, 20]:
        if window < len(r_cols):
            result_df[f"volatility_{window}"] = result_df[r_cols].apply(
                lambda row: row.rolling(window=window).std().iloc[-1], axis=1)
    
    # 2. Ratio de Sharpe simplifié (rendement / volatilité)
    if 'r_mean' in result_df.columns and 'r_std' in result_df.columns:
        result_df['sharpe_ratio'] = result_df['r_mean'] / (result_df['r_std'] + 1e-8)
    
    # 3. Skewness (asymétrie de la distribution des rendements)
    result_df['r_skewness'] = result_df[r_cols].apply(
        lambda row: row.skew(), axis=1)
    
    # 4. Kurtosis (poids des extrêmes dans la distribution)
    result_df['r_kurtosis'] = result_df[r_cols].apply(
        lambda row: row.kurtosis(), axis=1)
    
    # 5. RSI (Relative Strength Index) sur les rendements
    for window in [5, 14]:
        if window < len(r_cols):
            # Convertir en série pour utiliser rolling
            def calculate_rsi(row, period=window):
                prices = pd.Series(row.values)
                deltas = prices.diff()
                seed = deltas[:period+1]
                up = seed[seed >= 0].sum() / period
                down = -seed[seed < 0].sum() / period
                rs = up / (down + 1e-8)
                return 100 - (100 / (1 + rs))
            
            result_df[f'rsi_{window}'] = result_df[r_cols].apply(calculate_rsi, axis=1)
    
    # 6. MACD (Moving Average Convergence Divergence)
    if len(r_cols) > 26:
        def calculate_macd(row):
            prices = pd.Series(row.values)
            ema12 = prices.ewm(span=12, adjust=False).mean()
            ema26 = prices.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            return macd.iloc[-1]
        
        result_df['macd'] = result_df[r_cols].apply(calculate_macd, axis=1)
    
    # 7. Ratio Nombre de rendements positifs / nombre de rendements négatifs
    if 'r_pos_count' in result_df.columns and 'r_neg_count' in result_df.columns:
        result_df['pos_neg_ratio'] = result_df['r_pos_count'] / (result_df['r_neg_count'] + 1e-8)
    
    return result_df
