# grid_search_tuning.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def get_param_grids():
    """Define parameter grids for each model using discrete values for GridSearchCV"""
    
    param_grids = {
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1, 10, 50, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 150, 200, 250],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 8, 10],
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
            'model': MLPClassifier(
                random_state=42, 
                max_iter=2000,
                early_stopping=True,
                n_iter_no_change=10,
                tol=1e-4
            ),
            'params': {
                'hidden_layer_sizes': [(50,), (100,)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.005, 0.01],
                'solver': ['adam', 'sgd']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(
                random_state=42, 
                max_iter=2000,
                tol=1e-4
            ),
            'params': {
                'C': [0.1, 1.0, 10.0, 50.0],
                'solver': ['newton-cg', 'saga', 'sag'],
                'multi_class': ['ovr', 'multinomial'],
                'penalty': ['l2', None]
            }
        }
    }
    return param_grids

def tune_hyperparameters(X, y, model_name, param_grid, cv_splits=10):
    """Perform hyperparameter tuning using GridSearchCV"""
    # Create stratified CV to maintain class balance in each fold
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42) 
    
    # Use GridSearchCV for all models
    search = GridSearchCV(
        param_grid['model'],
        param_grid['params'],
        cv=cv,
        scoring='accuracy',
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
    
    return {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'best_model': search.best_estimator_,
        'cv_results': search.cv_results_,
        'train_score': train_score,
        'validation_score': test_score,
        'overfitting_score': train_score - test_score
    }

def compare_feature_sets(X_with_extra, X_without_extra, y, model_name, param_grid, 
                        with_label='with_extra', without_label='without_extra'):
    """Compare model performance with and without extra features"""
    print(f"\nTuning {model_name} {with_label}...")
    results_with_extra = tune_hyperparameters(X_with_extra, y, model_name, param_grid)
    
    print(f"\nTuning {model_name} {without_label}...")
    results_without_extra = tune_hyperparameters(X_without_extra, y, model_name, param_grid)
    
    return {
        with_label: results_with_extra,
        without_label: results_without_extra
    }

def print_comparison_results(results, model_name, with_label='with_extra', without_label='without_extra'):
    """Print detailed comparison results"""
    print(f"\n{'='*50}")
    print(f"Comparison Results for {model_name}")
    print(f"{'='*50}")
    
    print(f"\nResults {with_label}:")
    print(f"Best parameters: {results[with_label]['best_params']}")
    print(f"Mean CV accuracy: {results[with_label]['best_score']:.4f}")
    
    print(f"\nResults {without_label}:")
    print(f"Best parameters: {results[without_label]['best_params']}")
    print(f"Mean CV accuracy: {results[without_label]['best_score']:.4f}")
    
    # Calculate performance difference
    diff = results[with_label]['best_score'] - results[without_label]['best_score']
    print(f"\nPerformance difference ({with_label} - {without_label}): {diff:.4f}")
    
    # Provide recommendation
    if abs(diff) < 0.02:
        print(f"\nRecommendation: Consider using the model {without_label} for better generalization,")
        print("as the performance difference is minimal (<2%).")
    elif diff > 0:
        print(f"\nRecommendation: Consider using the model {with_label},")
        print("as it shows significantly better performance.")
    else:
        print(f"\nRecommendation: Use the model {without_label},")
        print("as it shows better performance and provides better generalization.")

def run_all_models_comparison(X_with_extra, X_without_extra, y, model_types=None,
                            with_label='with_extra', without_label='without_extra'):
    """Run comparison for all models or specified models"""
    param_grids = get_param_grids()
    
    # If no model types are specified, use all available models
    if model_types is None:
        model_types = list(param_grids.keys())
    
    all_results = {}
    
    for model_name in model_types:
        if model_name not in param_grids:
            print(f"Warning: {model_name} not found in parameter grids. Skipping.")
            continue
            
        print(f"\n{'#'*70}")
        print(f"Running comparison for {model_name}")
        print(f"{'#'*70}")
        
        results = compare_feature_sets(
            X_with_extra,
            X_without_extra,
            y,
            model_name,
            param_grids[model_name],
            with_label,
            without_label
        )
        
        print_comparison_results(results, model_name, with_label, without_label)
        all_results[model_name] = results
    
    return all_results

def save_results(results, filename, with_label='with_extra', without_label='without_extra'):
    """Save results to JSON file"""
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
            params = model_results[dataset_type]['best_params']
            results_summary[dataset_type][model_name] = params
            
            print(f"\n{model_name}:")
            print(f"Best Parameters: {params}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"\nResults have been saved to '{filename}'")