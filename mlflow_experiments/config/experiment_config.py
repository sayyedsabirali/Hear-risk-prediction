SMART_COMBINATIONS = [
    {
        "name": "basic_standard",
        "duplicate_handling": "keep_first",
        "numerical_missing": "mean", 
        "categorical_missing": "mode",
        "outlier_handling": {"method": "iqr", "threshold": 1.5},  # FIXED: consistent parameter names
        "skewness_reduction": "none",
        "encoding": "onehot",
        "scaling": "standard"
    },
    {
        "name": "advanced_robust", 
        "duplicate_handling": "keep_first",
        "numerical_missing": "median",
        "categorical_missing": "unknown",
        "outlier_handling": {"method": "winsorize", "limits": [0.05, 0.05]},  # FIXED
        "skewness_reduction": "log",
        "encoding": "onehot",
        "scaling": "robust"
    },
    {
        "name": "knn_target_encoding",
        "duplicate_handling": "keep_first",
        "numerical_missing": "knn",
        "categorical_missing": "unknown", 
        "outlier_handling": {"method": "capping", "limits": [0.05, 0.95]},  # FIXED: removed duplicate 'method'
        "skewness_reduction": "yeojohnson",
        "encoding": "target",
        "scaling": "standard"
    },
    {
        "name": "simple_fast",
        "duplicate_handling": "keep_first",
        "numerical_missing": "mean",
        "categorical_missing": "mode",
        "outlier_handling": {"method": "none"},  # FIXED
        "skewness_reduction": "none",
        "encoding": "label", 
        "scaling": "minmax"
    },
    {
        "name": "no_scaling_for_trees",
        "duplicate_handling": "keep_first",
        "numerical_missing": "median",
        "categorical_missing": "mode",
        "outlier_handling": {"method": "iqr", "threshold": 1.5},  # FIXED
        "skewness_reduction": "none",
        "encoding": "onehot",
        "scaling": "none"
    },
    {
        "name": "heavy_preprocessing",
        "duplicate_handling": "drop_all", 
        "numerical_missing": "knn",
        "categorical_missing": "mode",
        "outlier_handling": {"method": "winsorize", "limits": [0.05, 0.05]},  # FIXED
        "skewness_reduction": "yeojohnson",
        "encoding": "onehot",
        "scaling": "standard"
    },
    {
        "name": "linear_models_optimized",
        "duplicate_handling": "keep_first",
        "numerical_missing": "mean",
        "categorical_missing": "unknown",
        "outlier_handling": {"method": "iqr", "threshold": 1.5},  # FIXED
        "skewness_reduction": "yeojohnson", 
        "encoding": "onehot",
        "scaling": "standard"
    },
    {
        "name": "tree_models_optimized",
        "duplicate_handling": "keep_first",
        "numerical_missing": "median",
        "categorical_missing": "unknown",
        "outlier_handling": {"method": "none"},  # FIXED
        "skewness_reduction": "none",
        "encoding": "label",
        "scaling": "none"
    },
    {
        "name": "drop_missing_heavy",
        "duplicate_handling": "keep_first", 
        "numerical_missing": "drop",
        "categorical_missing": "drop",
        "outlier_handling": {"method": "iqr", "threshold": 1.5},  # FIXED
        "skewness_reduction": "log",
        "encoding": "onehot",
        "scaling": "standard"
    },
    {
        "name": "minimal_preprocessing",
        "duplicate_handling": "keep_first",
        "numerical_missing": "mean",
        "categorical_missing": "mode", 
        "outlier_handling": {"method": "none"},  # FIXED
        "skewness_reduction": "none",
        "encoding": "onehot",
        "scaling": "none"
    }
]