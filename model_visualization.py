import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison(results, classification_type, save_path=None):
    """
    Plot model comparison
    
    Parameters:
    results: dict containing results for both scenarios
    classification_type: 'product' or 'storage' to determine labels
    save_path: path to save the plot (optional)
    """
    # Get model names and results
    models = list(results[list(results.keys())[0]].keys())
    scenarios = list(results.keys())
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set width of bars and positions
    bar_width = 0.35
    x = np.arange(len(models))
    
    # Set labels based on classification type
    if classification_type == 'product':
        title = 'Model Comparison for Product Type Classification'
        label1 = 'Without Storage Conditions'
        label2 = 'With Storage Conditions'
    else:  # storage
        title = 'Model Comparison for Storage Condition Classification'
        label1 = 'Without Product Type'
        label2 = 'With Product Type'
    
    # Create bars
    bars1 = ax.bar(x - bar_width/2, 
                   [results[scenarios[0]][model]['mean_accuracy'] for model in models],
                   bar_width, 
                   yerr=[results[scenarios[0]][model]['std_accuracy'] for model in models],
                   label=label1,
                   capsize=5)
    
    bars2 = ax.bar(x + bar_width/2, 
                   [results[scenarios[1]][model]['mean_accuracy'] for model in models],
                   bar_width, 
                   yerr=[results[scenarios[1]][model]['std_accuracy'] for model in models],
                   label=label2,
                   capsize=5)
    
    # Customize the plot
    ax.set_ylabel('Mean Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on top of bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    
    # Adjust layout and display
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 