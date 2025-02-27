import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

class BakeryClassification:
    def __init__(self, data_path='augmented_bakery_data.csv'):
        """Initialize the classification analysis"""
        self.data = pd.read_csv(data_path)
        self.classifiers = {
            'SVM': SVC(kernel='rbf', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.scaler = StandardScaler()
        
    def prepare_data(self, task='product_type'):
        """Prepare data for classification"""
        # Get feature columns (gain and phase)
        feature_cols = [col for col in self.data.columns if col.startswith(('gain_', 'phase_'))]
        X = self.data[feature_cols]
        
        # Select target based on task
        if task == 'product_type':
            y = self.data['Product_Type']
        else:  # storage_condition
            y = self.data['Storage_Condition']
            
        return X, y
    
    def plot_confusion_matrix(self, y_true, y_pred, title, classifier_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{classifier_name} - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{classifier_name}_{title.lower().replace(" ", "_")}.png')
        plt.close()
    
    def evaluate_classifier(self, clf, X, y, task, clf_name):
        """Evaluate a single classifier"""
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        
        # Train-test split for detailed evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and predict
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, task, clf_name)
        
        # Generate classification report
        report = classification_report(y_test, y_pred)
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': accuracy,
            'classification_report': report
        }
    
    def run_classification_analysis(self, task='product_type'):
        """Run complete classification analysis"""
        X, y = self.prepare_data(task)
        results = {}
        
        print(f"\nRunning classification analysis for {task}")
        print("=" * 50)
        
        for name, clf in self.classifiers.items():
            print(f"\nEvaluating {name}...")
            results[name] = self.evaluate_classifier(clf, X, y, task, name)
            
            print(f"\nResults for {name}:")
            print(f"Cross-validation scores: {results[name]['cv_scores']}")
            print(f"Mean CV accuracy: {results[name]['cv_mean']:.4f} (±{results[name]['cv_std']:.4f})")
            print(f"Test set accuracy: {results[name]['test_accuracy']:.4f}")
            print("\nClassification Report:")
            print(results[name]['classification_report'])
        
        # Plot comparison of classifier performances
        self.plot_classifier_comparison(results, task)
        
        return results
    
    def plot_classifier_comparison(self, results, task):
        """Plot comparison of classifier performances"""
        clf_names = list(results.keys())
        cv_means = [results[name]['cv_mean'] for name in clf_names]
        cv_stds = [results[name]['cv_std'] for name in clf_names]
        test_accuracies = [results[name]['test_accuracy'] for name in clf_names]
        
        x = np.arange(len(clf_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, cv_means, width, label='CV Accuracy', yerr=cv_stds)
        rects2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy')
        
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Classifier Performance Comparison - {task}')
        ax.set_xticks(x)
        ax.set_xticklabels(clf_names, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'classifier_comparison_{task}.png')
        plt.close()

def main():
    # Initialize analysis
    analyzer = BakeryClassification()
    
    # Run analysis for product type classification
    product_results = analyzer.run_classification_analysis(task='product_type')
    
    # Run analysis for storage condition classification
    storage_results = analyzer.run_classification_analysis(task='storage_condition')
    
    # Save results to file
    with open('classification_results.txt', 'w') as f:
        f.write("Classification Results\n")
        f.write("=====================\n\n")
        
        for task, results in [("Product Type", product_results), 
                            ("Storage Condition", storage_results)]:
            f.write(f"\n{task} Classification\n")
            f.write("-" * 50 + "\n")
            
            for clf_name, clf_results in results.items():
                f.write(f"\n{clf_name}:\n")
                f.write(f"Mean CV Accuracy: {clf_results['cv_mean']:.4f} "
                       f"(±{clf_results['cv_std']:.4f})\n")
                f.write(f"Test Accuracy: {clf_results['test_accuracy']:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(clf_results['classification_report'])
                f.write("\n")

if __name__ == "__main__":
    main()