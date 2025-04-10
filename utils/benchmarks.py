from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_models():
    models={
        "xgboost_baseline": {
            "model" : XGBClassifier(
                objective ="multi:softmax",
                num_class = 3,
                random_state=42
            ),
            "description": "XGBoost de base"
        },
        "xgboost_tuned" : {
            "model" : XGBClassifier(
                objective ="multi:softmax",
                num_class = 3,
                random_state=42,
                n_estimators=300,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.5,
                colsample_bytree=0.5
            ),
            "description": "XGBoost avec paramètres "
            
        },
        "rf_baseline": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            "description": "Baseline Random Forest model"
        },
        "logistic": {
            "model": LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            ),
            "description": "Multinomial Logistic Regression"
        }
    }
    return models 

def get_unsupervised_models():
    """
    Renvoie un dictionnaire de modèles non supervisés
    
    Returns:
    --------
    dict
        Dictionnaire contenant les modèles non supervisés
    """
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    
    models = {
        "kmeans": {
            "model": KMeans(n_clusters=3, random_state=42, n_init=10),
            "description": "K-Means Clustering"
        },
        "dbscan": {
            "model": DBSCAN(eps=0.5, min_samples=5),
            "description": "DBSCAN Clustering"
        },
        "agg_clustering": {
            "model": AgglomerativeClustering(n_clusters=3),
            "description": "Agglomerative Clustering"
        }
    }
    
    return models

def get_supervised_advanced_models():
    """
    Renvoie un dictionnaire de modèles supervisés avancés
    
    Returns:
    --------
    dict
        Dictionnaire contenant les modèles supervisés avancés
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    
    models = {
        "random_forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "description": "Random Forest Classifier"
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "description": "Gradient Boosting Classifier"
        },
        "svm": {
            "model": SVC(probability=True, random_state=42),
            "description": "Support Vector Machine"
        },
        "xgboost_tuned": {
            "model": XGBClassifier(objective="multi:softmax", num_class=3, n_estimators=200, 
                                  max_depth=5, learning_rate=0.1, random_state=42),
            "description": "XGBoost optimisé"
        }
    }
    
    return models