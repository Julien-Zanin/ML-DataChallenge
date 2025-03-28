import pandas as pd 

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

