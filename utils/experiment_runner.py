import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from time import time
from utils.data_registry import DATASETS
from utils.feature_engineering import add_features
from utils.benchmarks import get_models
# For analyze_feature_importance
from xgboost import XGBClassifier
from IPython.display import display

from utils.data_preprocessing import normalize_rendements_by_row

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
    
def add_result(results_tracker, result):
    """Add a result to the results tracker DataFrame."""
    return pd.concat([results_tracker, pd.DataFrame([result])], ignore_index=True)

def evaluate_feature_sets(dataset_key='raw', model_key='xgboost_baseline'):
    """
    Évaluer différents ensembles de features.
    
    Parameters:
    -----------
    dataset_key : str
        Clé du dataset dans DATASETS
    model_key : str
        Clé du modèle à utiliser
    """
    from utils.experiment_runner import run_experiment
    
    print(f"\nÉvaluation des ensembles de features sur {dataset_key} avec {model_key}...")
    
    # Définir différents ensembles de features
    feature_sets_to_test = [
        None,  # Toutes les features
        ["basic_stats"],  # Statistiques de base
        ["technical"],  # Indicateurs techniques
        ["basic_stats", "technical"]  # Combinaison
    ]
    
    results = []
    
    # Tester chaque ensemble de features
    for feature_set in feature_sets_to_test:
        try:
            print(f"\nTest avec feature_set = {feature_set}")
            result = run_experiment(dataset_key=dataset_key, model_key=model_key, 
                                   add_feat=True, feature_sets=feature_set)
            
            set_name = "Toutes" if feature_set is None else ', '.join(feature_set)
            print(f"Accuracy avec {set_name}: {result['accuracy']:.4f}")
            
            results.append({
                'feature_set': set_name,
                'accuracy': result['accuracy'],
                'f1_weighted': result['f1_weighted'],
                'train_time': result['train_time'],
                'total_time': result['total_time'],
                'result': result
            })
            
            # Ajouter également le résultat sans feature engineering pour comparaison
            if feature_set is None:  # Comparer avec le cas sans features uniquement une fois
                result_no_feat = run_experiment(dataset_key=dataset_key, model_key=model_key, 
                                             add_feat=False)
                print(f"Accuracy sans features: {result_no_feat['accuracy']:.4f}")
                
                results.append({
                    'feature_set': 'Aucune (baseline)',
                    'accuracy': result_no_feat['accuracy'],
                    'f1_weighted': result_no_feat['f1_weighted'],
                    'train_time': result_no_feat['train_time'],
                    'total_time': result_no_feat['total_time'],
                    'result': result_no_feat
                })
        
        except Exception as e:
            print(f"Erreur lors de l'évaluation avec feature_set = {feature_set}: {e}")
    
    # Créer un DataFrame des résultats
    results_df = pd.DataFrame([{
        'Feature Set': r['feature_set'],
        'Accuracy': r['accuracy'],
        'F1 Score': r['f1_weighted'],
        'Train Time (s)': r['train_time'],
        'Total Time (s)': r['total_time']
    } for r in results])
    
    # Trier par précision décroissante
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    # Afficher le tableau des résultats
    print("\nRésumé des performances avec différents ensembles de features:")
    display(results_df)
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Feature Set', y='Accuracy', data=results_df)
    plt.title(f'Impact des différents ensembles de features ({dataset_key}, {model_key})')
    plt.xlabel('Ensemble de features')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Visualiser le compromis performance/temps
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Total Time (s)'], results_df['Accuracy'], s=100)
    
    # Ajouter des annotations pour chaque point
    for i, row in results_df.iterrows():
        plt.annotate(row['Feature Set'], 
                    (row['Total Time (s)'], row['Accuracy']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    
    plt.title('Compromis performance/temps pour différents ensembles de features')
    plt.xlabel('Temps total (secondes)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return results

def analyze_feature_importance(dataset_key='raw', use_precomputed_features=True):
    """
    Analyser l'importance des features avec XGBoost.
    
    Parameters:
    -----------
    dataset_key : str
        Clé du dataset dans DATASETS
    use_precomputed_features : bool
        Utiliser le dataset avec features préalculées
    """
    from xgboost import XGBClassifier
    
    try:
        # Déterminer quelle dataset utiliser
        actual_key = f"{dataset_key}_with_features" if use_precomputed_features else dataset_key
        
        if actual_key not in DATASETS:
            if use_precomputed_features:
                print(f"Dataset avec features préalculées '{actual_key}' introuvable. Utilisation de '{dataset_key}' à la place.")
                actual_key = dataset_key
            else:
                raise ValueError(f"Dataset '{dataset_key}' introuvable dans le registre.")
        
        # Charger le dataset
        dataset_info = DATASETS[actual_key]
        X_train = pd.read_csv(dataset_info['train'])
        
        # Mapping pour la variable cible
        mapping = {-1: 0, 0: 1, 1: 2}
        y_train = X_train["reod"].replace(mapping)
        
        # Extraire les features
        non_feature_cols = ['ID', 'day', 'equity', 'reod']
        feature_cols = [col for col in X_train.columns if col not in non_feature_cols]
        
        X_features = X_train[feature_cols].fillna(0)
        
        # Entraîner un modèle XGBoost
        model = XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)
        model.fit(X_features, y_train)
        
        # Récupérer l'importance des features
        importance = model.feature_importances_
        
        # Créer un DataFrame pour visualisation
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Visualiser les 20 features les plus importantes
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Top 20 des features les plus importantes selon XGBoost')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.show()
        
        # Afficher les résultats
        print("\nTop 20 des features les plus importantes:")
        display(importance_df.head(20))
        
        return importance_df, model
    
    except Exception as e:
        print(f"Erreur lors de l'analyse de l'importance des features: {e}")
        import traceback
        traceback.print_exc()  # Affiche la trace complète pour déboguer
        return None, None
    
    