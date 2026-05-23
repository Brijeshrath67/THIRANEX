# TASK1 — Wine Dataset Exploratory Data Analysis (EDA)

## Overview

This notebook performs an **Exploratory Data Analysis (EDA)** on the classic **Wine dataset** from `sklearn.datasets.load_wine`. The dataset contains chemical analysis results of wines grown in the same region in Italy but derived from three different cultivars (target classes 0, 1, 2).

## Steps Performed

### 1. Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
```

### 2. Data Loading & DataFrame Creation

- Loaded the wine dataset via `load_wine()`
- Created a pandas DataFrame with 13 feature columns and a `target` column
- **Shape**: 178 rows × 14 columns
- All columns are **numeric** (13 `float64`, 1 `int64` for target)
- **No missing values** — every column has 178 non-null entries

### 3. Features

| Feature | Description |
|---|---|
| `alcohol` | Alcohol content |
| `malic_acid` | Malic acid |
| `ash` | Ash |
| `alcalinity_of_ash` | Alcalinity of ash |
| `magnesium` | Magnesium |
| `total_phenols` | Total phenols |
| `flavanoids` | Flavanoids |
| `nonflavanoid_phenols` | Non-flavanoid phenols |
| `proanthocyanins` | Proanthocyanins |
| `color_intensity` | Color intensity |
| `hue` | Hue |
| `od280/od315_of_diluted_wines` | OD280/OD315 of diluted wines |
| `proline` | Proline |
| `target` | Wine class (0, 1, 2) |

### 4. Statistical Summary (`df.describe()`)

Key observations:
- **Alcohol**: ranges from 11.03 to 14.83 (mean ~13.0)
- **Magnesium**: ranges from 70 to 162 (mean ~99.7)
- **Proline**: ranges from 278 to 1680 (highest variance feature, std ~314.9)
- **Target** is well-distributed across 3 classes (mean ~0.94)

### 5. Visualizations

- **Correlation Heatmap**: Shows relationships between all numeric features; some strong positive/negative correlations (e.g., flavanoids and total_phenols)
- **Histograms** (all 14 columns): Distribution shape for each feature
- **Boxplots** (first 5 features): Identifies outliers and spread for alcohol, malic_acid, ash, alcalinity_of_ash, and magnesium
- **Target Countplot**: Visualizes class distribution across the 3 wine cultivars

### 6. Data Export

```python
df.to_csv("cleaned_wine_data.csv", index=False)
```

The cleaned dataset is saved locally for downstream modeling tasks.

## Output

- **`cleaned_wine_data.csv`** — CSV file containing the full cleaned wine dataset

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
