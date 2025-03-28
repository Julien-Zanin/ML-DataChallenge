import pandas as pd

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
        "technical" : True,
    }
    
    if features_sets is not None :
        for key in available_features_sets:
           available_features_sets[key] = key in features_sets
    
    r_cols = [col for col in df.column if col.startswith('r_') and col[1:].isdigit()]
    
    if available_features_sets["basic_stats"]:
        result_df["r_mean"] = df[r_cols].mean(axis=1)
        result_df["r_std"] = df[r_cols].std(axis=1)
        result_df["r_min"] = df[r_cols].min(axis=1)
        result_df["r_max"] = df[r_cols].max(axis=1)
        result_df["r_sum"] = df[r_cols].sum(axis=1)
        
        #Compter les valeurs Positive/Negative/Zero
        result_df["r_pos_count"] = (df[r_cols] > 0).sum(axis=1)
        result_df["r_neg_count"] = (df[r_cols] < 0).sum(axis=1) 
        result_df["r_zero_count"] = (df[r_cols] == 0).sum(axis=1)
        
        #Somme des valeurs positives / négatives 
        result_df['r_pos_sum'] = df[r_cols].apply(lambda x: x[x > 0].sum(), axis=1)
        result_df['r_neg_sum'] = df[r_cols].apply(lambda x: x[x < 0].sum(), axis=1)
        
    if available_features_sets["technical"]:
        #MOn récupère la moyenne et l'écart type mobile sur les 5, 10 et 20 dernières valeurs (seulement la dernière valeur)
        for window in [5,10,20]:
            if window < len(r_cols):
                result_df[f"r_roll_mean_{window}"] = df[r_cols].apply(lambda row : row.rolling(window=window).mean().iloc[-1], axis=1)
                result_df[f"r_roll_std_{window}"] = df[r_cols].apply(lambda row : row.rolling(window=window).std().iloc[-1], axis=1)
    
        if len(r_cols) > 10:
            result_df["r_momentum_5"] = df[r_cols].apply( lambda row : row.iloc[-6] - row.iloc[-1], axis=1)
            result_df["r_momentum_10"] = df[r_cols].apply(lambda row : row.iloc[-1] - row.iloc[-11], axis=1)
    
    return result_df    
 
