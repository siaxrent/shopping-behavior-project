# Shopping Behavior: Segmentation + Purchase Factors

**Topic:** Анализ поведения покупателей в онлайн-магазине  
**Dataset:** Shopping Behavior Dataset  
**Goal:** сегментировать покупателей по паттернам поведения и выявить ключевые факторы покупки.

## Project tasks
- EDA, очистка, нормализация данных
- Кластеризация: KMeans, DBSCAN + оценка Silhouette Score
- Классификация: RandomForest, SVM (прогноз целевой метрики)
- Визуализация профилей сегментов покупателей и их покупательской активности

## Structure
- `data/raw/` — исходные данные (не коммитим)
- `data/processed/` — подготовленные данные
- `notebooks/` — EDA и эксперименты
- `src/` — код пайплайна
- `results/figures/` — сохранённые графики
- `results/reports/` — метрики и отчёты
- `models/` — сохранённая лучшая модель

## Run pipeline
1) Помести датасет сюда: `data/raw/shopping_behavior.csv` *(файл не коммитится)*  
2) Установи зависимости:
```bash
pip install -r requirements.txt
