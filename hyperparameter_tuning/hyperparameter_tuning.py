from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
    

# def get_param_grids():
#     """Define parameter grids for each model"""
#     param_grids = {
#         'SVM': {
#             'model': SVC(random_state=42),
#             'params': {
#                 'C': [0.1, 1, 10, 100],
#                 'kernel': ['rbf', 'linear'],
#                 'gamma': ['scale', 'auto', 0.1, 0.01]
#             }
#         },
#         'Random Forest': {
#             'model': RandomForestClassifier(random_state=42),
#             'params': {
#                 'n_estimators': [100, 200, 300],
#                 'max_depth': [None, 10, 20, 30],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4]
#             }
#         },
#         'KNN': {
#             'model': KNeighborsClassifier(),
#             'params': {
#                 'n_neighbors': [3, 5, 7, 9],
#                 'weights': ['uniform', 'distance'],
#                 'metric': ['euclidean', 'manhattan']
#             }
#         },
#         'Neural Network': {
#             'model': MLPClassifier(random_state=42, max_iter=1000),
#             'params': {
#                 'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
#                 'activation': ['relu', 'tanh'],
#                 'alpha': [0.0001, 0.001, 0.01],
#                 'learning_rate': ['constant', 'adaptive']
#             }
#         },
#         'Logistic Regression': {
#             'model': LogisticRegression(random_state=42, max_iter=1000),
#             'params': {
#                 'C': [0.1, 1, 10, 100],
#                 'solver': ['lbfgs', 'newton-cg', 'sag'],
#                 'multi_class': ['auto', 'ovr', 'multinomial']
#             }
#         }
#     }
#     return param_grids

# def tune_hyperparameters(X, y, model_name, param_grid):
#     """Perform hyperparameter tuning using GridSearchCV"""
#     grid_search = GridSearchCV(
#         param_grid['model'],
#         param_grid['params'],
#         cv=10,
#         scoring='accuracy',
#         n_jobs=-1,
#         verbose=1
#     )
    
#     grid_search.fit(X, y)
    
#     return {
#         'best_params': grid_search.best_params_,
#         'best_score': grid_search.best_score_,
#         'best_model': grid_search.best_estimator_,
#         'cv_results': grid_search.cv_results_
#     }

# def compare_feature_sets(X_with_product, X_without_product, y, model_name, param_grid):
#     """Compare model performance with and without product type features"""
#     print(f"\nTuning {model_name} with product type features...")
#     results_with_product = tune_hyperparameters(X_with_product, y, model_name, param_grid)
    
#     print(f"\nTuning {model_name} without product type features...")
#     results_without_product = tune_hyperparameters(X_without_product, y, model_name, param_grid)
    
#     return {
#         'with_product_type': results_with_product,
#         'without_product_type': results_without_product
#     }






def get_param_grids():
    """Define parameter grids/distributions for each model"""
    # For models using RandomizedSearchCV, we can use distributions
    # For models using GridSearchCV, we need discrete values
    
    param_grids = {
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'C': loguniform(0.1, 100),
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'] + list(loguniform(0.001, 0.1).rvs(5))
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': randint(50, 300),
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': randint(2, 11),
                'min_samples_leaf': randint(1, 5)
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],  # Discrete values for grid search
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Neural Network': {
            'model': MLPClassifier(
                random_state=42, 
                max_iter=2000,  # Increased from 200 to 2000
                early_stopping=True,  # Add early stopping
                n_iter_no_change=10,  # Stop if no improvement for 10 iterations
                tol=1e-4  # Tolerance for optimization
            ),
            'params': {
                'hidden_layer_sizes': [(50,), (100,)],  # Simpler architectures
                'activation': ['relu', 'tanh'],
                'alpha': uniform(0.0001, 0.01),
                'learning_rate_init': uniform(0.001, 0.01),  # Add learning rate initialization
                'solver': ['adam', 'sgd']  # Try different solvers
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(
                random_state=42, 
                max_iter=2000,  # Significantly increased from default
                tol=1e-4  # Adjusted tolerance
            ),
            'params': {
                'C': uniform(0.1, 100),
                'solver': ['newton-cg', 'saga', 'sag'],  # Avoid lbfgs which is causing issues
                'multi_class': ['ovr', 'multinomial'],
                'penalty': ['l2', None]  # Try with and without regularization
            }
        }
    }
    return param_grids

def tune_hyperparameters(X, y, model_name, param_grid):
    """Perform hyperparameter tuning using either RandomizedSearchCV or GridSearchCV"""
    # Create stratified CV to maintain perfect class balance in each fold
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) 
    
    # Determine which search method to use based on model complexity
    if model_name in ["Random Forest", "SVM", "Logistic Regression", "Neural Network"]:
        search = RandomizedSearchCV(
            param_grid['model'],
            param_grid['params'],
            n_iter=100,
            cv=cv,
            scoring='accuracy',  # Using accuracy for balanced dataset
            return_train_score=True,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        print(f"Using RandomizedSearchCV with 30 iterations for {model_name}")
    else:
        search = GridSearchCV(
            param_grid['model'],
            param_grid['params'],
            cv=cv,
            scoring='accuracy',  # Using accuracy for balanced dataset
            return_train_score=True,
            n_jobs=-1,
            verbose=1
        )
        print(f"Using GridSearchCV for {model_name}")
    
    # Fit the model
    search.fit(X, y)
    
    # Print detailed information about best parameters
    print(f"\nBest parameters for {model_name}:")
    print(f"Mean CV accuracy: {search.best_score_:.4f}")
    print(f"Parameters: {search.best_params_}")
    
    # Calculate overfitting metric
    best_idx = search.best_index_
    train_score = search.cv_results_['mean_train_score'][best_idx]
    test_score = search.cv_results_['mean_test_score'][best_idx]
    print(f"Train-Test gap: {train_score - test_score:.4f}")
    
    # Return comprehensive results
    return {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'best_model': search.best_estimator_,
        'cv_results': search.cv_results_,
        'train_score': train_score,
        'validation_score': test_score,
        'overfitting_score': train_score - test_score
    }

def compare_feature_sets(X_with_product, X_without_product, y, model_name, param_grid):
    """Compare model performance with and without product type features"""
    print(f"\nTuning {model_name} with product_type...")
    results_with_product = tune_hyperparameters(X_with_product, y, model_name, param_grid)
    
    print(f"\nTuning {model_name} without product_type...")
    results_without_product = tune_hyperparameters(X_without_product, y, model_name, param_grid)
    
    return {
        'with_product_type': results_with_product,
        'without_product_type': results_without_product
    }

def print_comparison_results(results, model_name):
    """Print detailed comparison results"""
    print(f"\n{'='*50}")
    print(f"Comparison Results for {model_name}")
    print(f"{'='*50}")
    
    print("\nResults with product type:")
    print(f"Best parameters: {results['with_product_type']['best_params']}")
    print(f"Mean CV accuracy: {results['with_product_type']['best_score']:.4f}")
    
    print("\nResults without product type:")
    print(f"Best parameters: {results['without_product_type']['best_params']}")
    print(f"Mean CV accuracy: {results['without_product_type']['best_score']:.4f}")
    
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

def save_results(results, filename, with_label, without_label):
    """Save results to JSON file without 'best_params' wrapper"""
    import json
    
    results_summary = {
        with_label: {},
        without_label: {}
    }
    
    print(f"\nModel Performance Summary:")
    print("=" * 80)
    
    for dataset_type in [with_label, without_label]:
        print(f"\n{dataset_type.replace('_', ' ').title()}:")
        print("-" * 40)
        
        for model_name, model_results in results.items():
            key = 'with_product_type' if 'with' in dataset_type else 'without_product_type'
            params = model_results[key]['best_params']
            
            # Store parameters directly without the 'best_params' wrapper
            results_summary[dataset_type][model_name] = params
            
            print(f"\n{model_name}:")
            print(f"Best Parameters: {params}")
    
    with open(filename, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"\nResults have been saved to '{filename}'")
