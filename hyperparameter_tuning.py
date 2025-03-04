from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def get_param_grids():
    """Define parameter grids for each model"""
    param_grids = {
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Neural Network': {
            'model': MLPClassifier(random_state=42, max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10, 100],
                'solver': ['lbfgs', 'newton-cg', 'sag'],
                'multi_class': ['auto', 'ovr', 'multinomial']
            }
        }
    }
    return param_grids

def tune_hyperparameters(X, y, model_name, param_grid):
    """Perform hyperparameter tuning using GridSearchCV"""
    grid_search = GridSearchCV(
        param_grid['model'],
        param_grid['params'],
        cv=10,  # Using 10-fold CV for tuning
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_
    } 
    
def compare_feature_sets(X_with_product, X_without_product, y, model_name, param_grid):
    """
    Compare model performance with and without product type features
    
    Parameters:
    -----------
    X_with_product : array-like
        Feature matrix including product type
    X_without_product : array-like
        Feature matrix excluding product type
    y : array-like
        Target variable
    model_name : str
        Name of the model to use
    param_grid : dict
        Parameter grid for the specified model
        
    Returns:
    --------
    dict
        Dictionary containing results for both feature sets
    """
    print(f"\nTuning {model_name} with product type features...")
    results_with_product = tune_hyperparameters(X_with_product, y, model_name, param_grid)
    
    print(f"\nTuning {model_name} without product type features...")
    results_without_product = tune_hyperparameters(X_without_product, y, model_name, param_grid)
    
    return {
        'with_product_type': results_with_product,
        'without_product_type': results_without_product
    }

def print_comparison_results(results, model_name):
    """
    Print detailed comparison results
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for both feature sets
    model_name : str
        Name of the model being compared
    """
    print(f"\n{'='*50}")
    print(f"Comparison Results for {model_name}")
    print(f"{'='*50}")
    
    print("\nResults with product type:")
    print(f"Best score: {results['with_product_type']['best_score']:.4f}")
    print(f"Best parameters: {results['with_product_type']['best_params']}")
    
    print("\nResults without product type:")
    print(f"Best score: {results['without_product_type']['best_score']:.4f}")
    print(f"Best parameters: {results['without_product_type']['best_params']}")
    
    # Calculate performance difference
    diff = results['with_product_type']['best_score'] - results['without_product_type']['best_score']
    print(f"\nPerformance difference (with - without): {diff:.4f}")
    
    # Provide recommendation
    if abs(diff) < 0.02:
        print("\nRecommendation: Consider using the model without product type for better generalization,")
        print("as the performance difference is minimal (<2%).")
    elif diff > 0:
        print("\nRecommendation: Consider using the model with product type,")
        print("as it shows significantly better performance.")
    else:
        print("\nRecommendation: Use the model without product type,")
        print("as it shows better performance and provides better generalization.")

def run_all_models_comparison(X_with_product, X_without_product, y):
    """
    Run comparison for all models
    
    Parameters:
    -----------
    X_with_product : array-like
        Feature matrix including product type
    X_without_product : array-like
        Feature matrix excluding product type
    y : array-like
        Target variable
        
    Returns:
    --------
    dict
        Dictionary containing results for all models
    """
    param_grids = get_param_grids()
    all_results = {}
    
    for model_name in param_grids.keys():
        print(f"\n{'#'*70}")
        print(f"Running comparison for {model_name}")
        print(f"{'#'*70}")
        
        results = compare_feature_sets(
            X_with_product,
            X_without_product,
            y,
            model_name,
            param_grids[model_name]
        )
        
        print_comparison_results(results, model_name)
        all_results[model_name] = results
    
    return all_results