{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "50b65d12-5bdf-43f5-a4db-9fd4015feb69",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#Visualization of Classification Accuracies  \n",
    "def plot_accuracy_comparisons(results_without_storage, results_with_storage):  \n",
    "    \"\"\"Plot accuracy comparisons for models with and without Product Types.\"\"\"  \n",
    "    models = list(results_without_storage.keys())  \n",
    "    accuracies_without = list(results_without_storage.values())  \n",
    "    accuracies_with = list(results_with_storage.values())  \n",
    "    \n",
    "    x = np.arange(len(models))  # the label locations  \n",
    "    width = 0.35  # the width of the bars  \n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(10, 6))  \n",
    "    bars1 = ax.bar(x - width/2, accuracies_without, width, label='Without Product_Type')  \n",
    "    bars2 = ax.bar(x + width/2, accuracies_with, width, label='With Product_Type')  \n",
    "\n",
    "    # Add some text for labels, title and custom x-axis tick labels, etc.  \n",
    "    ax.set_ylabel('Accuracy')  \n",
    "    ax.set_title('Model Accuracies')  \n",
    "    ax.set_xticks(x)  \n",
    "    ax.set_xticklabels(models)  \n",
    "    ax.legend()  \n",
    "\n",
    "    # Autolabel function to display accuracy on top of bars  \n",
    "    def autolabel(bars):  \n",
    "        \"\"\"Attach a text label above each bar in *bars*, displaying its height.\"\"\"  \n",
    "        for bar in bars:  \n",
    "            height = bar.get_height()  \n",
    "            ax.annotate(f'{height:.2f}',  \n",
    "                        xy=(bar.get_x() + bar.get_width() / 2, height),  \n",
    "                        xytext=(0, 3),  # 3 points vertical offset  \n",
    "                        textcoords=\"offset points\",  \n",
    "                        ha='center', va='bottom')  \n",
    "\n",
    "    autolabel(bars1)  \n",
    "    autolabel(bars2)  \n",
    "\n",
    "    plt.ylim(0, 1)  # Set y-axis limit (0 to 1 for accuracy)  \n",
    "    plt.show()  \n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.20"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
