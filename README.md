## Customer Behavior Analysis in an Online Store

**Topic:** analysis of customer behavior in an online store and factors that drive purchases.  
**Dataset:** Shopping Behavior Dataset (≈3900 records, 18 features).  
**Goal:** segment customers by behavioral patterns and build a model that predicts subscription status (`Subscription Status` field).

### Project goals
- **EDA and data cleaning**: inspect dataset structure, check types, missing values, duplicates, feature distributions and correlations (notebook `01_eda.ipynb`).  
- **Customer clustering**: KMeans + DBSCAN, search for optimal parameters and evaluate quality via **Silhouette Score** (notebook `02_clustering.ipynb` and script `main.py`).  
- **Segment interpretation**: build cluster profiles based on numerical and categorical features (notebook `03_cluster_profiles.ipynb`).  
- **Classification (Subscription Status)**: train **RandomForest** and **SVM** models, select the best by **ROC-AUC** and save the final model (notebook `04_classification.ipynb` and script `main.py`).  
- **Visualization**: save key plots (distributions, clusters in PCA space, ROC curves) into `results/figures`.  

### Project architecture
- **`main.py`** – main entry script that:
  - loads raw data from `data/raw/shopping_behavior.csv`;  
  - builds the feature matrix using a shared preprocessing pipeline;  
  - searches for the best `k` for KMeans (PCA(15) + grid over `k` from 2 to 15);  
  - runs a parameter search for DBSCAN on numeric-only features;  
  - trains classification models (RandomForest and SVM) for the `Subscription Status` target (the target column is removed from features);  
  - saves the best model to `models/best_model.joblib`, metrics to `results/reports/metrics_main.json`, and plots to `results/figures`.  

- **`src/`** – modular pipeline code:
  - `data_preprocessing.py` – unified preprocessing:
    - numeric features: `SimpleImputer(strategy="median")` + `StandardScaler`;  
    - categorical features: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`;  
    - exposes `build_preprocess_pipeline` and `fit_transform_preprocess` for re-use in scripts and notebooks.  
  - other modules (`clustering.py`, `modeling.py`, `evaluation.py`, `visualization.py`, `utils.py`) are reserved for gradually moving logic out of `main.py` (currently most logic lives in `main.py` and notebooks).  

- **`notebooks/`** – research notebooks:
  - `01_eda.ipynb` – initial exploratory analysis, distributions and correlations;  
  - `02_clustering.ipynb` – experimental tuning of clustering parameters;  
  - `03_cluster_profiles.ipynb` – interpretation and analysis of resulting segments;  
  - `04_classification.ipynb` – experiments with classification models and feature importance.  

- **`data/`**:
  - `data/raw/` – raw data (`shopping_behavior.csv`, not committed to the repository);  
  - `data/processed/` – possible saved intermediate datasets.  

- **`results/`**:
  - `results/figures/` – all saved plots (Silhouette for KMeans, PCA cluster visualization, ROC curves, correlations, etc.);  
  - `results/reports/` – JSON files with training metrics (`metrics_main.json` and others).  

- **`models/`**:
  - `best_model.joblib` – serialized best classification model (RandomForest or SVM, depending on ROC-AUC).  

### Pipeline logic (high-level)
- **1. Data loading**
  - CSV file `shopping_behavior.csv` is loaded from `data/raw/`.  
  - In the EDA notebook, missing values, duplicates, data types and basic statistics are checked.  

- **2. Feature preprocessing**
  - A common end-to-end preprocessing pipeline is built using `ColumnTransformer` + `Pipeline` from `scikit-learn`.  
  - Numeric features are imputed with the median and scaled via `StandardScaler`.  
  - Categorical features are one-hot encoded with `OneHotEncoder(handle_unknown="ignore")`.  
  - For clustering tasks, `fit_transform_preprocess` can use all features (including `Subscription Status` as categorical), which is acceptable for descriptive segmentation.  

- **3. Clustering (customer segmentation)**
  - **KMeans**:
    - features are first transformed and reduced via **PCA(15)**;  
    - `k` is searched from 2 to 15;  
    - for each `k`, the average **Silhouette Score** is computed and the best `k` is selected;  
    - the script plots and saves `k` vs Silhouette (`kmeans_pca15_silhouette_by_k.png`) as well as a scatter of the first two principal components colored by clusters.  
  - **DBSCAN** (numeric-only):
    - uses numeric columns: `Age`, `Purchase Amount (USD)`, `Review Rating`, `Previous Purchases`;  
    - applies normalization (imputer + scaler);  
    - searches a grid of `eps` and `min_samples`, evaluates number of clusters, noise points, and Silhouette Score (without noise);  
    - logs the best configuration, visualizes it in 2D via PCA and saves the figure.  

- **4. Classification (predicting Subscription Status)**
  - The target is built by `make_target_subscription` in `main.py`:
    - `"Yes" → 1`, `"No" → 0`; for any other value an explicit error is raised.  
  - The `Subscription Status` column is **explicitly removed** from features to avoid target leakage.  
  - Data is split into train/test with `test_size=0.2` and stratification by the target.  
  - Two models are trained:
    - **RandomForestClassifier** with 300 trees and `class_weight="balanced"`;  
    - **SVM (RBF)** with `C=2.0`, `probability=True`, `class_weight="balanced"`.  
  - **ROC-AUC** is computed for both models, the best one is selected and saved:
    - model → `models/best_model.joblib`;  
    - metrics and meta-information → `results/reports/metrics_main.json`;  
    - ROC curve of the best model → `results/figures/best_roc_curve_main.png`.  

### How to run the project
- **Requirements:**
  - Python 3.10+;  
  - `pip` installed.  

- **Setup steps:**
  1. Clone or copy the repository.  
  2. (Recommended) create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux / macOS
```

  3. Install dependencies:

```bash
pip install -r requirements.txt
```

  4. Download the **Shopping Behavior Dataset** (e.g. from Kaggle) and save the file as:

```text
data/raw/shopping_behavior.csv
```

- **Run the main pipeline:**

```bash
python main.py
```

After execution you will get:
- the saved model in `models/best_model.joblib`;  
- metric files in `results/reports/metrics_main.json`;  
- plots in `results/figures/`.  

### How to interpret the results
- **Metrics in `metrics_main.json`:**
  - `kmeans_best_k`, `kmeans_best_silhouette` – final number of clusters and their quality via Silhouette;  
  - `dbscan_clusters`, `dbscan_noise_points`, `dbscan_silhouette_no_noise` – DBSCAN cluster statistics;  
  - `rf_auc`, `svm_auc`, `best_model`, `best_auc` – classification performance and the chosen best model.  
- **Plots in `results/figures`:**
  - Silhouette for KMeans across different `k`;  
  - cluster visualization on the first two principal components;  
  - ROC curve of the best model, correlation heatmap and other diagnostic plots.  
- **Cluster profiles:** notebook `03_cluster_profiles.ipynb` highlights which features differentiate segments (age, average purchase amount, activity, etc.), which can directly feed into marketing hypotheses.  

### Ideas for further improvements (architecture and code)
- **Move logic from `main.py` into `src/` modules:**
  - create dedicated functions/classes for clustering, hyperparameter search, training classifiers, saving artifacts and logging;  
  - this will improve reusability, testability and code clarity.  
- **Configuration via config files:**
  - store paths and model parameters (range of `k`, DBSCAN grid, RF/SVM hyperparameters) in `config.yaml` or `config.json`;  
  - this allows changing experiments without touching the code.  
- **Logging instead of `print`:**
  - use the `logging` module with levels (`INFO`, `DEBUG`, `WARNING`) and log-to-file.  
- **Richer model evaluation:**
  - add `classification_report`, confusion matrix and PR curves to the main pipeline;  
  - save additional evaluation artifacts in `results/reports/`.  
- **Testing and robustness:**
  - add simple unit tests for `data_preprocessing.py` and target construction;  
  - harden the pipeline against unexpected values in `Subscription Status` and other data anomalies.  

All these improvements can be introduced gradually without breaking the existing logic and the current way of running the project through `main.py`.

