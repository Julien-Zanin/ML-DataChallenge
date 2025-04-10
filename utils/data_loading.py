from utils.data_registry import DATASETS
from utils.data_preprocessing import normalize_rendements_by_row
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

def load_datasets(strategies=['raw', 'ffbf', 'bfff', 'interp', 'mice',"knn"]):
    """
    Charge directement les datasets préformés.
    
    Parameters:
    -----------
    strategies : list
        Liste des stratégies d'imputation à charger
    
    Returns:
    --------
    dict
        Dictionnaire contenant les datasets pour chaque stratégie
    """
    datasets = {}
    
    for key in strategies:
        if key in DATASETS:
            try:
                print(f"\nChargement du dataset {key}...")
                start_time = time()
                
                # Chargement des données
                train_path = DATASETS[key]['train']
                test_path = DATASETS[key]['test']
                
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                
                # Trier par ID pour garantir la cohérence
                train_df = train_df.sort_values(by="ID")
                test_df = test_df.sort_values(by="ID")
                
                datasets[key] = {
                    'train': train_df,
                    'test': test_df,
                    'description': DATASETS[key]['description']
                }
                
                load_time = time() - start_time
                print(f"Temps de chargement: {load_time:.2f} secondes")
                
                # Afficher les dimensions et le nombre de valeurs manquantes
                print(f"Dimensions train: {train_df.shape}, test: {test_df.shape}")
                
                # Vérifier les valeurs manquantes
                train_na = train_df.isna().sum().sum()
                test_na = test_df.isna().sum().sum()
                
                print(f"Valeurs manquantes - train: {train_na}, test: {test_na}")
                
                # Vérifier la distribution des classes (si 'reod' est présent)
                if 'reod' in train_df.columns:
                    print("\nDistribution des classes dans le dataset d'entraînement:")
                    reod_counts = train_df['reod'].value_counts(normalize=True) * 100
                    for cls, pct in reod_counts.items():
                        print(f"  Classe {cls}: {pct:.2f}%")
                
            except Exception as e:
                print(f"Erreur lors du chargement du dataset {key}: {e}")
    
    return datasets

def load_dataset_for_analysis(dataset_key, normalize=False):
    """
    Charge un dataset du registre DATASETS pour l'analyse.
    
    Parameters:
    -----------
    dataset_key : str
        Clé du dataset dans le registre
    normalize : bool
        Appliquer la normalisation par ligne
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    # Vérifier si le dataset existe dans le registre
    if dataset_key not in DATASETS:
        raise ValueError(f"Dataset '{dataset_key}' non trouvé dans le registre.")
    
    # Charger le dataset
    dataset_info = DATASETS[dataset_key]
    X_train = pd.read_csv(dataset_info["train"])
    X_test = pd.read_csv(dataset_info["test"])
    
    # Extraire les labels
    y_train = X_train["reod"].copy()
    y_test = X_test["reod"].copy()
    
    # Extraire les features (tout sauf ID et reod)
    X_train_feat = X_train.drop(["ID", "reod"], axis=1, errors='ignore')
    X_test_feat = X_test.drop(["ID", "reod"], axis=1, errors='ignore')
    
    # Appliquer normalisation si demandé
    if normalize:
        X_train_feat = normalize_rendements_by_row(X_train_feat)
        X_test_feat = normalize_rendements_by_row(X_test_feat)
    
    # Gérer les valeurs manquantes
    X_train_feat = X_train_feat.fillna(0)
    X_test_feat = X_test_feat.fillna(0)
    
    return X_train_feat, y_train, X_test_feat, y_test



def load_dataset_without_day_id(dataset_key, normalize=False):
    """
    Charge un dataset sans les variables day et ID
    
    Parameters:
    -----------
    dataset_key : str
        Clé du dataset dans DATASETS
    normalize : bool
        Appliquer normalisation par ligne
        
    Returns:
    --------
    tuple (X_train, y_train, X_test, y_test)
    """
    # Charger le dataset complet
    X_train_full, y_train, X_test_full, y_test = load_dataset_for_analysis(dataset_key, normalize)
    
    # Retirer day et ID
    cols_to_remove = ['day', 'ID']
    X_train = X_train_full.drop(columns=[col for col in cols_to_remove if col in X_train_full.columns])
    X_test = X_test_full.drop(columns=[col for col in cols_to_remove if col in X_test_full.columns])
    
    print(f"Données chargées sans day et ID - Dimensions: {X_train.shape}")
    return X_train, y_train, X_test, y_test

def select_financial_and_rendements(X_train, X_test):
    """
    Sélectionne toutes les features financières importantes et tous les rendements,
    mais retire day et ID
    
    Parameters:
    -----------
    X_train, X_test : DataFrames
        Données d'entraînement et de test
        
    Returns:
    --------
    X_train_selected, X_test_selected : DataFrames avec features sélectionnées
    """
    # Identifier les colonnes de rendement
    rendement_cols = [col for col in X_train.columns if col.startswith('r') and col[1:].isdigit()]
    
    # Identifier les features financières importantes (basées sur l'analyse précédente)
    financial_features = [
        'equity',
        'pos_ratio', 
        'neg_ratio',
        'momentum',
        'sharpe_ratio',
        'volatility_20', 
        'volatility_30',
        'volatility_10',
        'trend_slope'
    ]
    
    # Sélectionner les colonnes qui existent effectivement dans le DataFrame
    financial_features = [f for f in financial_features if f in X_train.columns]
    
    # Colonnes à retirer (ID et day)
    cols_to_remove = ['ID', 'day']
    cols_to_keep = rendement_cols + financial_features
    
    # Retirer les colonnes à exclure
    cols_to_keep = [col for col in cols_to_keep if col not in cols_to_remove]
    
    # Sélectionner les colonnes
    X_train_selected = X_train[cols_to_keep]
    X_test_selected = X_test[cols_to_keep]
    
    print(f"Sélection de {len(cols_to_keep)} features ({len(rendement_cols)} rendements + {len(financial_features)} financières)")
    print(f"Features financières: {financial_features}")
    
    return X_train_selected, X_test_selected
