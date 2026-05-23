# TASK4 — Wine Classification with Model Serialization & Inference

## Overview

This notebook extends the Random Forest classifier from TASK2 with **model serialization** (save/load pipeline) and **inference on new samples**. It trains a classifier on the sklearn wine dataset, saves the trained model, loads it back, and makes predictions on individual wine samples.

## Steps Performed

### 1. Imports
```python
import joblib  # NEW: for model serialization
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

### 2. Data Loading
- Loaded wine dataset via `load_wine()` directly (not from CSV)
- 178 samples, 13 features, 3 target classes

### 3. Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- 80% training (142), 20% testing (36)

### 4. Model Training & Evaluation
- **Algorithm**: `RandomForestClassifier()` with default params
- **Accuracy**: **1.0 (100%)**

### 5. Model Serialization (NEW)

```python
# Save model to disk
joblib.dump(model, "wine_quality_model.pkl")

# Load model back
loaded_model = joblib.load("wine_quality_model.pkl")
```
- Model saved as `wine_quality_model.pkl` for reuse without retraining

### 6. Inference on Samples

| Sample | Features Used | Predicted Class |
|---|---|---|
| `X.iloc[0]` (row index 0) | alcohol=14.23, proline=1065, ... | **class_0** |
| `X.iloc[10]` (row index 10) | alcohol=14.1, proline=1510, ... | **class_0** |

- Both predictions verified against actual labels using `loaded_model.predict()`

### 7. Feature Importance

Top features:
| Feature | Importance |
|---|---|
| **color_intensity** | 0.1926 |
| **flavanoids** | 0.1653 |
| **proline** | 0.1465 |
| **od280/od315_of_diluted_wines** | 0.1404 |
| **alcohol** | 0.1055 |

Visualized as a horizontal bar plot.

## Key Files Produced
- `wine_quality_model.pkl` — serialized Random Forest model

## Dependencies
- Python 3.x
- pandas, numpy, joblib
- matplotlib, seaborn
- scikit-learn
