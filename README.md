#BAKERY-Spectra-Classification

#Project Overview
This project focuses on classifying bakery products and their storage conditions using impedance spectroscopy data. By applying various machine learning techniques, we aim to accurately classify bakery products based on their types and storage conditions using frequency-domain measurements.

#Dataset Description

__Measurement Characteristics__
- **Technology**: Impedancemetry with Vector Network Analyzer
- **Acquisition Range**: Frequency from 300 to 900 MHz
- **Features**: 202 features (101 gain values and 101 phase values) per sample

#Sample Details
Bakery Products: 2 types
Bread (A)
Cookies (B)
Storage Conditions: 3 types
Open (1)
Wrapped (2)
Humid environment (3)
Replicates: 10 per combination
Original Sample Count: 60 samples
Data Augmentation
To address the limited dataset size, we implemented data augmentation by adding Gaussian noise:
First Augmentation: Added slight Gaussian noise to gain features of the original samples
Second Augmentation: Added higher Gaussian noise to gain features of the original samples
Final Dataset Size: 180 samples (60 original + 120 augmented)
Classification Tasks
Product Type Classification
Without storage condition as a feature
With storage condition as a feature
Storage Condition Classification
Without product type as a feature
With product type as a feature
Algorithms Implemented
We compared five different supervised classification methods:
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Random Forest
Neural Network
Logistic Regression
Linear Discriminant Analysis (LDA)
BAKERY-Spectra-Classification/
├── hyperparameter_tuning/          # Hyperparameter tuning code and results
│   └── hyperparameter_tuning/      # Implementation details
├── results/                        # Classification results
│   ├── LDA_Product_Classification_results/
│   ├── LDA_Storage_Condition_Classification/
│   ├── LDA4Product_Type_classification/
│   ├── LDA4Storage_Condition_classification/
│   ├── Product_Classification_results/
│   └── Storage_Condition_Classification_results/
├── RawData/                        # Original dataset
├── analysis_summary.txt            # Summary of analysis findings
├── augmentation_analysis.png       # Visualization of data augmentation effects
├── augmented_bakery_data.csv       # Augmented dataset
├── augmented_data_analysis.png     # Analysis of augmented data
├── bakery_classification.ipynb     # Main notebook for classification
├── bakery_classification.py        # Python script version of the notebook
├── classification_results.json     # JSON file with classification results
├── LDA_Product_Classification.ipynb # LDA for product classification
├── LDA_Storage_Classification.ipynb # LDA for storage classification
├── preprocessing.ipynb             # Data preprocessing steps
├── processed_bakery_data.csv       # Preprocessed dataset
├── Product_Type_Classification.ipynb # Product type classification notebook
├── Product_type_analysis_summary.txt # Summary of product classification
├── signal_comparison.png           # Comparison of signal features
└── README.md                       # Project documentation
Methodology
Data Preprocessing
Data Loading: Imported raw data from impedance measurements
Feature Extraction: Extracted gain and phase features from spectral data
Data Augmentation: Augmented data by adding controlled noise
Normalization: Applied standard scaling to normalize features
Hyperparameter Tuning
Implemented grid search and cross-validation to find optimal parameters for each algorithm
Stored best parameters for reuse in main classification tasks
Model Training & Evaluation
Split data into training and testing sets
Trained models using the five different algorithms
Evaluated performance using accuracy, precision, recall, F1-score, and confusion matrices
Conducted comparative analysis between different models
Feature Engineering
Analyzed which parts of the spectra were most informative for classification
Evaluated the impact of different preprocessing techniques on model performance
Investigated the importance of gain vs. phase features
Results
The project provides a comprehensive comparison of the five classification methods for both product type and storage condition classification tasks. Detailed results, including performance metrics and visualizations, are available in the respective result directories.
Requirements
Python 3.x
Scikit-learn
Pandas
NumPy
Matplotlib
TensorFlow/Keras
Jupyter Notebook
How to Run
Clone the repository
Install required dependencies
Run preprocessing.ipynb to prepare the data
Run hyperparameter tuning notebooks to determine optimal parameters
Execute classification notebooks for each task
Review results in the results directory
Conclusion
This project demonstrates the effectiveness of various machine learning approaches for classifying bakery products and their storage conditions using impedance spectroscopy data. The comparative analysis highlights the strengths and weaknesses of each method and provides insights into the most informative spectral features for these classification tasks.