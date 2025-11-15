"""Configuration for preprocessing and model combinations"""

# Preprocessing strategies
MISSING_VALUE_STRATEGIES = [
    {"numerical": "drop", "categorical": "drop"},
    {"numerical": "mean", "categorical": "mode"},
    {"numerical": "median", "categorical": "mode"}, 
    {"numerical": "knn", "categorical": "mode"},
    {"numerical": "mean", "categorical": "unknown"},
    {"numerical": "median", "categorical": "unknown"},
]

OUTLIER_METHODS = [
    {"method": "remove", "threshold": 1.5},
    {"method": "cap", "threshold": 1.5},
    {"method": "cap", "threshold": 2.0},
    {"method": "cap", "threshold": 3.0},
    {"method": "winsorize", "threshold": 1.5},
    {"method": "none", "threshold": 1.5}
]

SKEWNESS_METHODS = [
    {"method": "log"},
    {"method": "boxcox"},
    {"method": "yeojohnson"},
    {"method": "none"}
]

SCALING_METHODS = [
    {"method": "standard"},
    {"method": "minmax"},
    {"method": "robust"},
    {"method": "none"}
]

# Model configurations
MODELS = [
    {"name": "xgboost", "params": {"n_estimators": 100, "max_depth": 6, "random_state": 42, "eval_metric": "logloss"}},
    {"name": "lightgbm", "params": {"n_estimators": 100, "max_depth": 6, "random_state": 42, "verbose": -1}},
    {"name": "catboost", "params": {"iterations": 100, "depth": 6, "random_state": 42, "verbose": False}},
    {"name": "random_forest", "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}},
    {"name": "logistic_regression", "params": {"max_iter": 1000, "random_state": 42}},
    {"name": "gradient_boosting", "params": {"n_estimators": 100, "max_depth": 6, "random_state": 42}},
    {"name": "ada_boost", "params": {"n_estimators": 100, "random_state": 42}},
    {"name": "extra_trees", "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}},
    {"name": "svm", "params": {"kernel": "rbf", "random_state": 42, "probability": True}},
    {"name": "knn", "params": {"n_neighbors": 5}},
    {"name": "decision_tree", "params": {"max_depth": 10, "random_state": 42}},
    {"name": "naive_bayes", "params": {}}
]

def generate_combinations(max_combinations=100, random_seed=42):
    """Generate RANDOM preprocessing + model combinations"""
    import itertools
    import random
    
    # Set seed for reproducibility
    random.seed(random_seed)
    
    # Generate ALL possible combinations
    all_combos = list(itertools.product(
        MISSING_VALUE_STRATEGIES,
        OUTLIER_METHODS,
        SKEWNESS_METHODS,
        SCALING_METHODS,
        MODELS
    ))
    
    total_possible = len(all_combos)
    print(f"📊 Total possible combinations: {total_possible}")
    
    # Randomly sample combinations
    if len(all_combos) > max_combinations:
        print(f"🎲 Randomly sampling {max_combinations} combinations...")
        all_combos = random.sample(all_combos, max_combinations)
    else:
        print(f"✅ Using all {len(all_combos)} combinations")
        # Even if using all, shuffle them for random order
        random.shuffle(all_combos)
    
    # Format combinations
    combinations = []
    for i, (missing, outlier, skewness, scaling, model) in enumerate(all_combos):
        combinations.append({
            'id': i + 1,
            'name': f"combo_{i+1:03d}",
            'missing': missing,
            'outlier': outlier,
            'skewness': skewness,
            'scaling': scaling,
            'model': model
        })
    
    return combinations


def generate_stratified_combinations(n_per_model=10, random_seed=42):
    """
    Generate BALANCED random combinations - ensuring each model gets tested equally
    
    Args:
        n_per_model: Number of random preprocessing combinations per model
        random_seed: Random seed for reproducibility
    """
    import random
    
    random.seed(random_seed)
    
    combinations = []
    combo_id = 1
    
    print(f"🎯 Generating {n_per_model} random combinations for each of {len(MODELS)} models")
    
    # For each model, generate random preprocessing combinations
    for model in MODELS:
        print(f"  ⚙️  Model: {model['name']} - generating {n_per_model} combinations")
        
        for _ in range(n_per_model):
            # Randomly select one option from each preprocessing category
            missing = random.choice(MISSING_VALUE_STRATEGIES)
            outlier = random.choice(OUTLIER_METHODS)
            skewness = random.choice(SKEWNESS_METHODS)
            scaling = random.choice(SCALING_METHODS)
            
            combinations.append({
                'id': combo_id,
                'name': f"combo_{combo_id:03d}",
                'missing': missing,
                'outlier': outlier,
                'skewness': skewness,
                'scaling': scaling,
                'model': model
            })
            
            combo_id += 1
    
    # Shuffle all combinations for random execution order
    random.shuffle(combinations)
    
    print(f"✅ Total combinations generated: {len(combinations)}")
    
    return combinations


def get_full_grid_combinations():
    """
    Generate COMPLETE grid of all possible combinations (NO randomization)
    Warning: This can generate thousands of combinations!
    """
    import itertools
    
    all_combos = list(itertools.product(
        MISSING_VALUE_STRATEGIES,
        OUTLIER_METHODS,
        SKEWNESS_METHODS,
        SCALING_METHODS,
        MODELS
    ))
    
    total = len(all_combos)
    print(f"⚠️  WARNING: Full grid will generate {total} combinations!")
    
    combinations = []
    for i, (missing, outlier, skewness, scaling, model) in enumerate(all_combos):
        combinations.append({
            'id': i + 1,
            'name': f"combo_{i+1:04d}",
            'missing': missing,
            'outlier': outlier,
            'skewness': skewness,
            'scaling': scaling,
            'model': model
        })
    
    return combinations