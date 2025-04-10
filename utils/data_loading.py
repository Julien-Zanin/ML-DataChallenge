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