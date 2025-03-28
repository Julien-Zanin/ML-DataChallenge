from utils.data_standardize import standardize, inverse_standardize
from sklearn.impute import KNNImputer
import pandas as pd 


def impute_group(group, features_cols):
    # Select only feature columns
    group_features = group[features_cols]
    
    # Standardization at group level
    group_features_std, group_stats = standardize(group_features)
    
    # Imputation on standardized group
    imputer_group = KNNImputer(n_neighbors=5)
    group_imputed_std = imputer_group.fit_transform(group_features_std)
    
    # Create DataFrame with matching columns from the imputation result
    # Use the actual columns from the imputation result rather than features_cols
    group_imputed_std_df = pd.DataFrame(
        group_imputed_std, 
        index=group.index
    )
    
    # Assign the actual column names from the original features
    group_imputed_std_df.columns = group_features.columns
    
    # Inversion of standardization
    group_imputed = inverse_standardize(group_imputed_std_df, group_stats)
    
    # Replace imputed columns in the original group
    group_result = group.copy()
    group_result[features_cols] = group_imputed
    
    return group_result