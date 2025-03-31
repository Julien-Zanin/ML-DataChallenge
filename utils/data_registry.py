DATASETS = {
    "raw" : {
        "train" : r"processed_data\X_train_70.csv",
        "test"  : r"processed_data\X_test_70.csv",
        "description" : "Données brutes"
    },
    "ffbf" : {
        "train" : r"processed_data\X_train_ffbf.csv",
        "test"  : r"processed_data\X_test_ffbf.csv",
        "description" : "Données forward filled puis backward"
    },
    "bfff" : {
        "train" : r"processed_data\X_train_bfff.csv",
        "test"  : r"processed_data\X_test_bfff.csv",
        "description" : "Données backward filled puis forward"
    },
    "interp" : {
        "train" : r"processed_data\X_train_interp.csv",
        "test"  : r"processed_data\X_test_interp.csv",
        "description" : "Données interpolation linéaire puis bffff"
    },
    "mice" : {
        "train" : r"processed_data/X_train_mice.csv",
        "test"  : r"processed_data/X_test_mice.csv",
        "description" : "Données MICE imputer puis bfff"
    },
    "knn" : { 
        "train" : r"processed_data/X_train_knn.csv",
        "test"  : r"processed_data/X_test_knn.csv",
        "description" : "Données knn imputer puis bfff"
    }
}

def add_dataset_with_features(dataset_key, train_path, test_path, description):
    """
    Ajoute un dataset avec features au registre DATASETS.
    
    Parameters:
    -----------
    dataset_key : str
        Clé unique pour le dataset
    train_path : str
        Chemin vers le fichier d'entraînement
    test_path : str
        Chemin vers le fichier de test
    description : str
        Description du dataset
    
    Returns:
    --------
    bool
        True si l'ajout a été effectué avec succès, False sinon
    """
    if dataset_key in DATASETS:
        print(f"Avertissement: Le dataset {dataset_key} existe déjà et sera écrasé.")
    
    DATASETS[dataset_key] = {
        "train": train_path,
        "test": test_path,
        "description": description
    }
    
    return True