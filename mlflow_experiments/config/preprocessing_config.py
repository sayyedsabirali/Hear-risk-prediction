# All possible combinations for testing
PREPROCESSING_CONFIG = {
    "missing_value_strategies": [
        {"numerical": "drop", "categorical": "drop"},  # Baseline
        {"numerical": "mean", "categorical": "mode"},
        {"numerical": "median", "categorical": "mode"}, 
        {"numerical": "knn", "categorical": "mode"},
        {"numerical": "mean", "categorical": "unknown"},
        {"numerical": "median", "categorical": "unknown"},
    ],
    
    "outlier_handling_methods": [
        {"method": "remove", "threshold": 1.5},
        {"method": "clipper", "threshold": 1.5},
        {"method": "clipper", "threshold": 2.0},
        {"method": "clipper", "threshold": 3.0},
        {"method": "none", "threshold": 1.5}
    ],
    
    "skewness_reduction": [
        {"method": "log"},
        {"method": "boxcox"},
        {"method": "yeojohnson"},
        {"method": "none"}
    ],
    
    "scaling_methods": [
        {"method": "standard"},
        {"method": "minmax"},
        {"method": "robust"},
        {"method": "none"}
    ],
    
#     "models": [
#         {"name": "xgboost", "params": {"n_estimators": 100, "max_depth": 6, "random_state": 42}},
#         {"name": "random_forest", "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}},
#         {"name": "logistic_regression", "params": {"max_iter": 1000, "random_state": 42}}
#     ]
# }

    "models": [
        {"name": "lightgbm", "params": {"n_estimators": 100, "max_depth": 6, "random_state": 42, "verbose": -1}},
        {"name": "catboost", "params": {"iterations": 100, "depth": 6, "random_state": 42, "verbose": False}},
        {"name": "ada_boost", "params": {"n_estimators": 100, "random_state": 42}},
        {"name": "extra_trees", "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}},
        {"name": "svm", "params": {"kernel": "rbf", "random_state": 42}},
    ]
}

# Generate all combinations (max 500)
def generate_combinations(max_combinations=500):
    import itertools
    import random
    
    # Get all possible combinations
    all_combinations = list(itertools.product(
        PREPROCESSING_CONFIG["missing_value_strategies"],
        PREPROCESSING_CONFIG["outlier_handling_methods"],
        PREPROCESSING_CONFIG["skewness_reduction"],
        PREPROCESSING_CONFIG["scaling_methods"],
        PREPROCESSING_CONFIG["models"]
    ))
    
    # If more than max_combinations, sample randomly
    if len(all_combinations) > max_combinations:
        print(f"🎲 Randomly sampling {max_combinations} combinations from {len(all_combinations)} total")
        all_combinations = random.sample(all_combinations, max_combinations)
    
    # Convert to the required format
    combinations_list = []
    for i, (missing, outlier, skewness, scaling, model) in enumerate(all_combinations):
        combo_name = f"combo_{i+1:03d}"
        
        combinations_list.append({
            "name": combo_name,
            "missing": missing,
            "outlier": outlier,
            "skewness": skewness,
            "scaling": scaling,
            "model": model
        })
    
    return combinations_list

# Generate combinations for testing
COMBINATIONS_TO_TEST = generate_combinations(500)