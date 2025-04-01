import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from IPython.display import display


def visualize_results(results_df):
    """
    Visualize experiment results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with experiment results
    """
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Compare accuracy across datasets for each model
    plt.figure(figsize=(14, 8))
    sns.barplot(x="dataset", y="accuracy", hue="model", data=results_df)
    plt.title("Model Accuracy by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("accuracy_by_dataset.png")
    
    # 2. Compare accuracy with and without feature engineering
    plt.figure(figsize=(14, 8))
    sns.barplot(x="model", y="accuracy", hue="features_added", data=results_df)
    plt.title("Impact of Feature Engineering")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("feature_engineering_impact.png")
    
    # 3. Heatmap of best combinations
    pivot_df = results_df.pivot_table(
        values="accuracy", 
        index="dataset", 
        columns="model", 
        aggfunc="max"
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Best Accuracy for Each Dataset-Model Combination")
    plt.tight_layout()
    plt.savefig("best_combinations_heatmap.png")
    
    # 4. Feature importance analysis (if applicable)
    # This would need to be implemented separately for each model type
    
    plt.close('all')
