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
    "knn" : { 
        "train" : r"processed_data/X_train_knn.csv",
        "test"  : r"processed_data/X_test_knn.csv",
        "description" : "Données knn imputer puis bfff"
    }
}
