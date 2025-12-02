# Shopping Behavior: Segmentation + Purchase Factors

**Topic:** Анализ поведения покупателей в онлайн-магазине  
**Dataset:** Shopping Behavior Dataset  
**Goal:** сегментировать покупателей по паттернам поведения и выявить ключевые факторы покупки.

## Project tasks
- EDA, очистка, нормализация данных
- Кластеризация: KMeans, DBSCAN + оценка Silhouette Score
- Классификация: RandomForest, SVM (прогноз целевой метрики)
- Визуализация профилей сегментов и покупательской активности

## Structure
- `data/raw/` — исходные данные (не коммитим)
- `data/processed/` — подготовленные данные
- `notebooks/` — EDA и эксперименты
- `src/` — код пайплайна
- `results/figures/` — графики
- `models/` — сохранённые модели

## Quick start
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py

## Run pipeline
1) Put dataset here: `data/raw/shopping_behavior.csv` (not committed)
2) Install deps:
```bash
pip install -r requirements.txt

## Results (saved artifacts)
- `results/figures/kmeans_silhouette_by_k.png` — выбор k по Silhouette Score
- `results/figures/kmeans_pca.png` — визуализация сегментов KMeans (PCA 2D)
- `results/figures/dbscan_numeric_pca.png` — DBSCAN по числовым признакам (PCA 2D)
- `results/figures/best_roc_curve_main.png` — ROC-кривая лучшей модели
- `results/reports/metrics_main.json` — метрики пайплайна (silhouette, AUC и т.д.)
- `models/best_model.joblib` — сохранённая лучшая модель
