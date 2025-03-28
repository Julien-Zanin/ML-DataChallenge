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

