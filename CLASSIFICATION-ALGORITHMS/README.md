# MNIST Handwritten Digit Classification

## Summary

This project implements and compares **6 different ML classification algorithms** for recognizing handwritten digits from the MNIST dataset. The notebook provides a comprehensive analysis of each algorithm's performance, including accuracy, precision, recall, F1-score, and computational efficiency (training and prediction times). The analysis is performed both with and without Principal Component Analysis (PCA) dimensionality reduction to understand the impact of feature reduction on model performance.

### Key Objectives

- Compare multiple classification algorithms on the same dataset
- Evaluate performance metrics (accuracy, precision, recall, F1-score)
- Analyze computational efficiency (training and prediction times)
- Visualize decision trees and random forest structures
- Understand the impact of PCA on model performance

## Features

- ✅ **6 Classification Algorithms**: KNN, Logistic Regression, Naive Bayes, SVM, Random Forest, Decision Tree
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- ✅ **Performance Comparison**: Side-by-side comparison of all algorithms
- ✅ **Tree Visualization**: Visual representation of Decision Trees and Random Forest
- ✅ **Time Analysis**: Training and prediction time measurements
- ✅ **PCA Analysis**: Comparison with and without dimensionality reduction
- ✅ **Data Preprocessing**: Normalization and standardization for optimal performance

## Dataset

- **Dataset**: MNIST (Modified National Institute of Standards and Technology)
- **Total Samples**: 70,000 handwritten digit images
- **Features**: 784 features (28x28 pixel images)
- **Classes**: 10 classes (digits 0-9)
- **Subset Used**: 2,000 samples for analysis
- **Train/Test Split**: 80/20 (1,600 training, 400 testing samples)

## Classification Algorithms

### 1. K-Nearest Neighbors (KNN)
- **Type**: Instance-based learning (lazy learner)
- **Parameters**: `n_neighbors=5`, `weights='uniform'`
- **Characteristics**: No training time, slower prediction

### 2. Logistic Regression
- **Type**: Linear classifier
- **Parameters**: `max_iter=2000`, `solver='lbfgs'`, `multi_class='multinomial'`
- **Characteristics**: Fast training, very fast prediction, good baseline

### 3. Support Vector Machines (SVM)
- **Type**: Kernel-based classifier
- **Parameters**: `kernel='rbf'`, `C=5`, `gamma='scale'`
- **Characteristics**: Excellent accuracy, moderate training time

### 4. Decision Tree
- **Type**: Tree-based classifier
- **Parameters**: `max_depth=20`, `min_samples_split=10`, `criterion='gini'`
- **Characteristics**: Fast prediction, interpretable, prone to overfitting

### 5. Random Forest
- **Type**: Ensemble of Decision Trees
- **Parameters**: `n_estimators=100`, `max_depth=20`, `random_state=42`
- **Characteristics**: Robust, good accuracy, handles overfitting better than single tree

### 6. Naive Bayes (Gaussian)
- **Type**: Probabilistic classifier
- **Parameters**: Default GaussianNB parameters
- **Characteristics**: Fast training and prediction, assumes feature independence

## Performance Summary

### Results Table

| Classifier | Accuracy | Precision | Recall | F1-Score | Train Time (s) | Predict Time (s) |
|------------|----------|-----------|--------|----------|----------------|------------------|
| **Support Vector Machines** | **0.9675** (96.75%) | 0.9675 | 0.9675 | 0.9675 | 0.2743 | 0.1461 |
| **K Nearest Neighbors** | 0.9450 (94.50%) | 0.9450 | 0.9450 | 0.9450 | 0.0307 | 0.0462 |
| **Random Forest** | 0.9425 (94.25%) | 0.9425 | 0.9425 | 0.9425 | 0.3562 | 0.0388 |
| **Logistic Regression** | 0.9350 (93.50%) | 0.9350 | 0.9350 | 0.9350 | 21.6725 | 0.0178 |
| **Decision Trees** | 0.7500 (75.00%) | 0.7500 | 0.7500 | 0.7500 | 0.2300 | 0.0031 |
| **Naive Bayes** | 0.7200 (72.00%) | 0.7200 | 0.7200 | 0.7200 | 0.0854 | 0.0031 |

### Key Insights

🏆 **Best Accuracy**: Support Vector Machines (96.75%)
- Highest accuracy among all algorithms
- Excellent generalization on MNIST dataset
- Moderate training time (0.27s)

⚡ **Fastest Training**: K Nearest Neighbors (0.0307s)
- Lazy learner - no actual training phase
- All computation happens during prediction

🚀 **Fastest Prediction**: Decision Trees (0.0031s)
- Extremely fast inference
- Trade-off: Lower accuracy (75%)

### Performance Analysis

#### Top Performers
1. **SVM (96.75%)**: Best overall accuracy, excellent for image classification
2. **KNN (94.50%)**: Very good accuracy with minimal training time
3. **Random Forest (94.25%)**: Robust ensemble method, good balance

#### Moderate Performers
4. **Logistic Regression (93.50%)**: Good baseline, but slow training (21.67s)
5. **Decision Tree (75.00%)**: Fast but prone to overfitting, lower accuracy

#### Lower Performers
6. **Naive Bayes (72.00%)**: Fast but accuracy limited by independence assumption

### Algorithm Characteristics

| Algorithm | Accuracy | Training Speed | Prediction Speed | Best For |
|-----------|----------|----------------|------------------|----------|
| SVM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | High accuracy needs |
| KNN | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Quick prototyping |
| Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Robust predictions |
| Logistic Regression | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | Linear boundaries |
| Decision Tree | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Interpretability |
| Naive Bayes | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Speed priority |

## Data Preprocessing

### Steps Applied

1. **Data Loading**: MNIST dataset from OpenML (70,000 samples)
2. **Subset Selection**: 2,000 samples from test data for analysis
3. **Normalization**: Pixel values scaled from 0-255 to 0-1 range
4. **Standardization**: StandardScaler applied for distance-based algorithms
5. **Train/Test Split**: 80/20 split with stratification

## Tree Visualization

The notebook includes visualization of:
- **Decision Tree**: Full tree structure with `plot_tree()`
- **Random Forest**: First 3 trees from the ensemble (100 total trees)

## Installation

### Quick Install

Install all required dependencies using the requirements file:

```bash
pip install -r requirements.txt
```

### Manual Install

Alternatively, install packages individually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn python-dotenv ipykernel jupyter
```

### Required Libraries

- `numpy` (>=1.24.0): Numerical computations and array operations
- `pandas` (>=2.0.0): Data manipulation and analysis
- `matplotlib` (>=3.7.0): Plotting and visualization
- `seaborn` (>=0.12.0): Statistical visualizations
- `scikit-learn` (>=1.3.0): Machine learning algorithms and utilities
- `python-dotenv` (>=1.0.0): Environment variable management
- `ipykernel` (>=6.25.0): Jupyter notebook kernel support
- `jupyter` (>=1.0.0): Jupyter notebook interface

### Optional Dependencies

- `tensorflow` (>=2.13.0): Alternative MNIST dataset loading method (fallback option)

## Usage

### Running the Notebook

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Open the notebook**:
   ```bash
   jupyter notebook classification_learning.ipynb
   ```

3. **Run cells sequentially**:
   - Cell 1-2: SSL fix and imports if required.
   - Cell 3-4: Data loading and preprocessing
   - Cell 5+: Algorithm training and evaluation

### Key Sections

1. **Data Loading**: Loads MNIST dataset (with SSL fix for OpenML)
2. **Data Preprocessing**: Normalization
3. **Algorithm Training**: Trains each classifier
4. **Performance Evaluation**: Calculates metrics and creates comparison table
5. **Visualization**: Tree structures and performance charts
6. **PCA Analysis**: Comparison with dimensionality reduction

## Methodology

### Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: Correctness of positive predictions
- **Recall**: Ability to find all positive instances
- **F1-Score**: Harmonic mean of precision and recall

### Training Methodology

- **Stratified Split**: Maintains class distribution in train/test sets
- **Normalization**: Essential for distance-based algorithms (KNN, SVM)
- **Time Measurement**: Separate tracking of training and prediction times

## Key Findings

1. **SVM performs best** on MNIST with 96.75% accuracy
2. **KNN is fastest to train** (lazy learner) with good accuracy
3. **Decision Trees are fastest to predict** but sacrifice accuracy
4. **Random Forest** provides good balance between accuracy and robustness
5. **Logistic Regression** has slow training but fast prediction
6. **Naive Bayes** is fastest overall but has lowest accuracy

## PCA Impact Analysis

For detailed analysis of how Principal Component Analysis (PCA) affects each algorithm's performance, see the dedicated document:

📄 **[PCA Impact Analysis](./PCA_IMPACT_ANALYSIS.md)**

This document provides:
- Comprehensive comparison of accuracy with and without PCA
- Training speed improvements
- Algorithm-specific recommendations
- Detailed impact metrics for each classifier

## File Structure

```
Module-3-Project/
├── classification_learning.ipynb    # Main notebook
├── README.md                        # This file
├── PCA_IMPACT_ANALYSIS.md           # Detailed PCA impact analysis
└── requirements.txt                 # Python package dependencies
```

## Notes

- **Memory Considerations**: Full MNIST dataset (70K samples) requires significant memory
- **Subset Usage**: 2,000 samples used for faster experimentation
- **Reproducibility**: `random_state=42` used throughout for consistent results
- **Tree Depth**: Limited to 4 levels in visualizations for readability

## References

- MNIST Dataset: [OpenML](https://www.openml.org/d/554)
- Scikit-learn Documentation: [sklearn.tree](https://scikit-learn.org/stable/modules/tree.html)
- Scikit-learn Documentation: [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

## License

This project is for educational purposes.

