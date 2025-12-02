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
