# TASK3 — Wine Dataset Correlation & Target Distribution Analysis

## Overview

This notebook performs an in-depth **statistical analysis** of the wine dataset, focusing on **target distribution** and **feature correlation** to understand relationships between chemical properties and wine classes.

## Steps Performed

### 1. Imports & Data Loading
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_wine_data.csv")
```

- Dataset shape: **178 rows × 14 columns**
- Full statistical summary via `df.describe()`

### 2. Target Class Distribution

| Class | Count |
|---|---|
| **1** | 71 |
| **0** | 59 |
| **2** | 48 |

- Displayed via a countplot — class 1 is the most frequent, class 2 the least.

### 3. Correlation Matrix

Full 14×14 correlation matrix computed and analyzed. Key findings:

**Strongest positive correlations between features:**
| Feature Pair | Correlation |
|---|---|
| flavanoids ↔ total_phenols | **+0.865** |
| flavanoids ↔ od280/od315 | **+0.787** |
| proanthocyanins ↔ flavanoids | **+0.653** |
| proanthocyanins ↔ total_phenols | **+0.612** |

**Strongest negative correlations with target (most discriminative features):**
| Feature | Correlation with target |
|---|---|
| flavanoids | **-0.847** |
| od280/od315_of_diluted_wines | **-0.788** |
| total_phenols | **-0.719** |
| hue | **-0.617** |
| proline | **-0.634** |
| alcalinity_of_ash | **+0.518** |
| malic_acid | **+0.438** |
| nonflavanoid_phenols | **+0.489** |

- Features like **flavanoids**, **od280/od315**, **total_phenols**, and **proline** have strong negative correlation with the target, making them highly discriminative for wine classification.

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
