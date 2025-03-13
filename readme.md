# README: XGBoost and Feature Engineering Notebook

## Key Components

### 1. Data Preprocessing

- Dropped unnecessary columns: `'ID'`, `'MONTHS_BALANCE'`, `'FLAG_WORK_PHONE'`
- Created the target variable `'TARGET'` based on the `'STATUS'` column
- Standardized numerical features using `StandardScaler`

### 2. Principal Component Analysis (PCA)

PCA is used to reduce dimensionality while retaining the most important variance in the data. The principal components are calculated as:

```math
Z = XW
```

where:

- \(X\) is the standardized data matrix,
- \(W\) is the matrix of eigenvectors from the covariance matrix of \(X\),
- \(Z\) is the transformed data in the principal component space.

The variance explained by each principal component is given by:

```math
\lambda_i = \frac{\sigma_i^2}{\sum \sigma_j^2}
```

where \(\sigma_i^2\) is the eigenvalue associated with component \(i\).

### 3. Cross-Validation with K-Fold

K-Fold cross-validation ensures robust model evaluation by splitting the data into \(K\) equally sized folds. The model is trained on \(K-1\) folds and tested on the remaining fold, repeated for all folds. The general formula for the average performance metric is:

```math
M_{cv} = \frac{1}{K} \sum_{i=1}^{K} M_i
```

where \(M_i\) is the performance metric on the \(i\)-th fold.

### 4. Model Selection: XGBoost

XGBoost uses gradient boosting to optimize decision trees. The prediction at stage \(t\) is given by:

```math
F_t(x) = F_{t-1}(x) + \eta h_t(x)
```

where:

- \(F_{t-1}(x)\) is the previous prediction,
- \(h_t(x)\) is the new weak learner (decision tree),
- \(\eta\) is the learning rate.

The objective function minimized during training consists of both the loss function and a regularization term:

```math
Obj = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{T} \Omega(h_k)
```

where:

- \(l(y_i, \hat{y}_i)\) is the loss function,
- \(\Omega(h_k)\) is the complexity penalty for the \(k\)-th tree.

### 5. Hyperparameter Optimization with Optuna

Optuna optimizes hyperparameters by searching for the best combination through trials. The optimization process is defined as:

```math
\theta^* = \arg\min_{\theta \in \Theta} f(\theta)
```

where:

- \(\theta\) is a set of hyperparameters,
- \(\Theta\) is the search space,
- \(f(\theta)\) is the evaluation metric (e.g., validation loss).

### 6. SHAP: Feature Importance

SHAP (SHapley Additive exPlanations) explains model predictions by assigning contributions to each feature. The SHAP value for a feature \(i\) is computed as:

```math
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| -1)!}{|F|!} [f(S \cup \{i\}) - f(S)]
```

where:

- \(S\) is a subset of all features \(F\),
- \(f(S)\) is the model prediction with features in \(S\),
- \(\phi_i\) represents the marginal contribution of feature \(i\).

## How to Get the Best XGBoost Model

1. **Prepare Data:** Clean and preprocess data as outlined above.
2. **Perform PCA:** Decide if dimensionality reduction benefits the model.
3. **Use K-Fold Cross-Validation:** Ensure the model generalizes well.
4. **Optimize with Optuna:** Run hyperparameter tuning to find the best settings.
5. **Train Final XGBoost Model:** Use the best hyperparameters to train on full data.
6. **Analyze with SHAP:** Understand feature importance and model decisions.

## Usage

Run the notebook step by step to preprocess data, train the model, and evaluate results. The final model (`xgb_best`) can be used for predictions, and SHAP can be used for interpretability.

## Requirements

Ensure the following libraries are installed:

```bash
pip install numpy pandas scikit-learn xgboost shap optuna matplotlib seaborn
```

## Conclusion

This notebook demonstrates the complete pipeline of data preprocessing, feature selection with PCA, model training using **XGBoost**, hyperparameter tuning with **Optuna**, and interpretability using **SHAP**.