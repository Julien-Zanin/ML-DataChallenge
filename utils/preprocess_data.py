
import pandas as pd
from utils.data_registry import DATASETS, add_dataset_with_features
from utils.features import add_features

def precompute_datasets_with_features():
    """
    Précompute et sauvegarde les datasets avec features ajoutées.
    """
    for dataset_key in ['raw', 'ffbf', 'bfff', 'interp', 'mice']:
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
                output_train_path = f"processed_data/X_train_{dataset_key}_with_features.csv"
                output_test_path = f"processed_data/X_test_{dataset_key}_with_features.csv"
                
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

if __name__ == "__main__":
    precompute_datasets_with_features()