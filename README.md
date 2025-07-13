# Early Stopping for the Random Forest Classifier

This repository contains the code the upcoming paper on early stopping for the the random forest in a classification setting.


## Abstract

We develop and evaluate early stopping rules for classification tree ensembles and individual decision trees. The methods include both uniform and individual early stopping approaches that monitor training MSE with a threshold to prevent overfitting. Our implementation provides comparison between different early stopping strategies (uniform and individual stopping) and traditional methods, evaluating their performance across various data generating processes and empirical datasets. The early stopping methods achieve competitive classification performance while potentially reducing computational costs and model complexity. In particular, the interplay of depth and the mtry hyperparameter is of interest.

## Quick Start

```python
import numpy as np
import pandas as pd
import sys
sys.path.append('.')
from openml import datasets
from sklearn.model_selection import train_test_split
from src.utils.model_builder import build_rf_clf

# Load real dataset (Pima Indians Diabetes)
dataset = datasets.get_dataset(43582)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
X = X.to_numpy()
y = y.to_numpy()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build Random Forest with early stopping
rf_clf, fit_duration = build_rf_clf(
    X_train=X_train,
    y_train=y_train,
    f_train=None,  
    algorithm="UGES",  # Uniform Early Stopping
    max_features="sqrt",
    n_estimators=50,
    random_state=42
)

# Make predictions
y_pred = rf_clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print(f"Test accuracy: {accuracy:.3f}")

```

## Repository Structure

```

## Repository Structure

```
early_stopping_classification/
├── README.md
├── requirements.txt
├── src/                         # Source code
│   ├── algorithms/              # Early stopping implementations
│   └── utils/                   # Utility functions
├── experiments/                 # Experimental code
│   ├── simulations/             # Simulation studies
│   └── empirical_studies/       # Real data experiments
├── reporting/                   # Results and visualization
│   ├── latex_tables/            # LaTeX table generation
│   ├── mtry_plots/              # Visualization plots
│   └── results/                 # Generated results
└── results/                     # Final outputs
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ratmir-miftachov/early_stopping_classification.git
cd early_stopping_classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Contact

* **Ratmir Miftachov**: contact[at]miftachov.com
