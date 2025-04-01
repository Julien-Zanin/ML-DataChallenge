import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from time import time

from utils.data_registry import DATASETS
from utils.features import add_features
from utils.benchmarks import get_models

import matplotlib.pyplot as plt
import seaborn as sns

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

def run_experiment(dataset_key, model_key, add_feat=True, feature_sets=None, normalize_by_row=False, use_precomputed_features=False,scaler=StandardScaler()):
    """
    Run an experiment for a specific dataset and model.
    
    Parameters:
    -----------
    dataset_key : str
        Key for the dataset in the DATASETS dictionary
    model_key : str
        Key for the model in the get_models() dictionary
    add_feat : bool
        Whether to add engineered features
    feature_sets : list of str or None
        Feature sets to add if add_feat is True
    normalize_by_row : bool
        Whether to normalize rendement features by row
    use_precomputed_features : bool
        Whether to use a dataset that already has precomputed features
        
    Returns:
    --------
    dict
        Results dictionary with metrics
    """
    # Mapping for class labels
    mapping = {-1: 0, 0: 1, 1: 2}
    inverse_mapping = {0: -1, 1: 0, 2: 1}
    
    # Determine actual dataset key
    actual_key = f"{dataset_key}_features" if use_precomputed_features else dataset_key
    
    if actual_key not in DATASETS:
        if use_precomputed_features:
            print(f"Dataset avec features préalculées '{actual_key}' introuvable. Utilisation de '{dataset_key}' à la place.")
            actual_key = dataset_key
        else:
            raise ValueError(f"Dataset '{dataset_key}' introuvable dans le registre.")
    
    # Load datasets
    start_load_time = time()
    dataset_info = DATASETS[actual_key]
    X_train = pd.read_csv(dataset_info["train"])
    X_test = pd.read_csv(dataset_info["test"])
    load_time = time() - start_load_time
    
    # Extract labels
    y_train = X_train["reod"].replace(mapping)
    y_test = X_test["reod"].replace(mapping)
    
    # Conserver seulement ID et reod comme colonnes non-features
    non_feature_cols = ['ID', 'reod']
    
    # Add engineered features if requested (and not already precomputed)
    if add_feat and not use_precomputed_features:
        X_train = add_features(X_train, feature_sets)
        X_test = add_features(X_test, feature_sets)
    
    # Extract features (tout sauf ID et reod)
    feature_cols = [col for col in X_train.columns if col not in non_feature_cols]
    
    # Prepare data without fillna yet
    X_train_feat = X_train[feature_cols]
    X_test_feat = X_test[feature_cols]
    
    # Apply row normalization if requested (only to rendement columns)
    if normalize_by_row:
        print("Application de la normalisation par ligne des rendements...")
        X_train_feat = normalize_rendements_by_row(X_train_feat)
        X_test_feat = normalize_rendements_by_row(X_test_feat)
    
    # Check for missing values
    train_na_count = X_train_feat.isna().sum().sum()
    test_na_count = X_test_feat.isna().sum().sum()
    
    if train_na_count > 0 or test_na_count > 0:
        print(f"Attention: Valeurs manquantes détectées - Train: {train_na_count}, Test: {test_na_count}")
        print("Application de fillna(0)...")
        X_train_feat = X_train_feat.fillna(0)
        X_test_feat = X_test_feat.fillna(0)
    else:
        print("Aucune valeur manquante détectée.")
    
    # Get model
    models = get_models()
    model_info = models[model_key]
    model = model_info["model"]
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])
    
    # Train model
    start_train_time = time()
    pipeline.fit(X_train_feat, y_train)
    train_time = time() - start_train_time
    
    # Make predictions
    start_pred_time = time()
    y_pred = pipeline.predict(X_test_feat)
    pred_time = time() - start_pred_time
    
    # Map predictions back to original labels
    y_pred_orig = pd.Series(y_pred).replace(inverse_mapping)
    y_test_orig = y_test.replace(inverse_mapping)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_orig, y_pred_orig)
    report = classification_report(y_test_orig, y_pred_orig, output_dict=True)
    
    # Store results
    results = {
        "dataset": dataset_key,
        "dataset_description": dataset_info["description"],
        "model": model_key,
        "model_description": model_info["description"],
        "features_added": add_feat or use_precomputed_features,
        "feature_sets": feature_sets,
        "normalize_by_row": normalize_by_row,
        "accuracy": accuracy,
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "class_-1_precision": report.get("-1", {}).get("precision", 0),
        "class_0_precision": report.get("0", {}).get("precision", 0),
        "class_1_precision": report.get("1", {}).get("precision", 0),
        "class_-1_recall": report.get("-1", {}).get("recall", 0),
        "class_0_recall": report.get("0", {}).get("recall", 0),
        "class_1_recall": report.get("1", {}).get("recall", 0),
        "report": report,
        "load_time": load_time,
        "train_time": train_time,
        "pred_time": pred_time,
        "total_time": load_time + train_time + pred_time
    }
    
    return results

def display_experiment_result(result):
    """Display detailed results for a single experiment."""
    print(f"Dataset: {result['dataset']} ({result['dataset_description']})")
    print(f"Model: {result['model']} ({result['model_description']})")
    print(f"Features Added: {result['features_added']}")
    print(f"Feature Sets: {result['feature_sets']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Weighted F1-Score: {result['f1_weighted']:.4f}")
    print(f"Class-wise Performance:")
    print(f"  Class -1: Precision = {result['class_-1_precision']:.4f}, Recall = {result['class_-1_recall']:.4f}")
    print(f"  Class  0: Precision = {result['class_0_precision']:.4f}, Recall = {result['class_0_recall']:.4f}")
    print(f"  Class  1: Precision = {result['class_1_precision']:.4f}, Recall = {result['class_1_recall']:.4f}")
    print(f"Timing Information:")
    print(f"  Data Loading: {result['load_time']:.2f} seconds")
    print(f"  Training: {result['train_time']:.2f} seconds")
    print(f"  Prediction: {result['pred_time']:.2f} seconds")
    print(f"  Total Time: {result['total_time']:.2f} seconds")
    
    # Create a confusion matrix visualization
    cm = np.array([
        [result['report']['-1']['support'] * result['class_-1_recall'], 
         result['report']['-1']['support'] * (1-result['class_-1_recall']) * result['class_0_recall'] / (result['class_0_recall'] + result['class_1_recall']),
         result['report']['-1']['support'] * (1-result['class_-1_recall']) * result['class_1_recall'] / (result['class_0_recall'] + result['class_1_recall'])],
        [result['report']['0']['support'] * (1-result['class_0_recall']) * result['class_-1_recall'] / (result['class_-1_recall'] + result['class_1_recall']),
         result['report']['0']['support'] * result['class_0_recall'],
         result['report']['0']['support'] * (1-result['class_0_recall']) * result['class_1_recall'] / (result['class_-1_recall'] + result['class_1_recall'])],
        [result['report']['1']['support'] * (1-result['class_1_recall']) * result['class_-1_recall'] / (result['class_-1_recall'] + result['class_0_recall']),
         result['report']['1']['support'] * (1-result['class_1_recall']) * result['class_0_recall'] / (result['class_-1_recall'] + result['class_0_recall']),
         result['report']['1']['support'] * result['class_1_recall']]
    ])        
    # Normalize confusion matrix to show percentages
    cm_normalized = cm / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=['Predicted -1', 'Predicted 0', 'Predicted 1'],
                yticklabels=['Actual -1', 'Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix - {result["model"]} on {result["dataset"]}')
    plt.tight_layout()
    plt.show()
    

def add_result(results_tracker, result):
    """Add a result to the results tracker DataFrame."""
    return pd.concat([results_tracker, pd.DataFrame([result])], ignore_index=True)