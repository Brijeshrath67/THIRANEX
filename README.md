# TASK2 — Wine Classification with Random Forest

## Overview

This notebook builds a **Random Forest Classifier** on the cleaned wine dataset (from TASK1) to predict wine cultivar classes (0, 1, 2) based on 13 chemical features.

## Steps Performed

### 1. Imports
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### 2. Data Loading
- Loaded `cleaned_wine_data.csv` (178 rows, 14 columns)
- **Features (X)**: All columns except `target` (13 features)
- **Target (y)**: `target` column (3 classes)

### 3. Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- 80% training (142 samples), 20% testing (36 samples)

### 4. Model Training
- **Algorithm**: `RandomForestClassifier()` with default parameters
- Trained on `X_train`, `y_train`

### 5. Evaluation Results

| Metric | Value |
|---|---|
| **Accuracy** | **1.0 (100%)** |
| Precision (all classes) | 1.0 |
| Recall (all classes) | 1.0 |
| F1-score (all classes) | 1.0 |

Perfect classification across all 3 wine classes.

### 6. Confusion Matrix
- Visualized as a heatmap using seaborn
- All predictions correct — diagonal-only matrix

### 7. Feature Importance

Top 5 most important features for classification:

| Feature | Importance |
|---|---|
| **proline** | 0.1723 |
| **color_intensity** | 0.1693 |
| **flavanoids** | 0.1510 |
| **od280/od315_of_diluted_wines** | 0.1448 |
| **alcohol** | 0.1052 |

- Feature importance visualized as a horizontal bar plot

## Dependencies
- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn
