# Практическое занятие №2: настройка и улучшение ML-модели (Telco Customer Churn)
# ВАЖНО: сначала запустите lab1.py — он сохраняет модель в файл best_model_lab1.joblib

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

CLASS_NAMES = ["Остался (0)", "Ушёл (1)"]

# =====================================================================
# 1. ДИАГНОСТИКА ПРОБЛЕМЫ
# =====================================================================
print("=" * 70)
print("1. ДИАГНОСТИКА СТАРОЙ МОДЕЛИ")
print("=" * 70)

if not os.path.exists("best_model_lab1.joblib"):
    raise FileNotFoundError("Нет файла best_model_lab1.joblib. Сначала запустите lab1.py!")

saved = joblib.load("best_model_lab1.joblib")
old_model = saved["model"]
X_train, X_test = saved["X_train"], saved["X_test"]
y_train, y_test = saved["y_train"], saved["y_test"]
print(f"\nЗагружена модель из практики №1: {saved['model_name']}")

y_pred_old = old_model.predict(X_test)
y_proba_old = old_model.predict_proba(X_test)[:, 1]  # вероятность класса «ушёл»

print("\nClassification report (старая модель):")
print(classification_report(y_test, y_pred_old, target_names=CLASS_NAMES, digits=3))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_old, display_labels=CLASS_NAMES, cmap="Blues"
)
plt.title(f"Старая модель: {saved['model_name']}")
plt.xlabel("Предсказание модели")
plt.ylabel("Реальность")
plt.tight_layout()
plt.show()

# Анализ дисбаланса
print("Распределение классов в обучающей выборке:")
print(y_train.value_counts().rename({0: "Остался", 1: "Ушёл"}))
print(f"Доля ушедших: {y_train.mean():.1%}  ->  классы НЕсбалансированы (~3:1)")

print("""
Какая ошибка критичнее для бизнеса?
  • False Positive — модель сказала «уйдёт», а клиент остался.
    Цена: лишняя скидка/звонок лояльному клиенту — недорого.
  • False Negative — модель сказала «останется», а клиент ушёл.
    Цена: потерянный клиент и весь его будущий доход — дорого.
  => False Negative критичнее. Нужно ловить как можно больше уходящих,
     то есть повышать Recall для класса «Ушёл».

ЦЕЛЬ ОПТИМИЗАЦИИ: повысить Recall для класса «Клиент ушёл» минимум до 0.7,
сохранив при этом приемлемый F1-score.
""")

# =====================================================================
# 2. СОЗДАНИЕ НАДЁЖНОГО КОНВЕЙЕРА
# =====================================================================
print("=" * 70)
print("2. СОЗДАНИЕ КОНВЕЙЕРА")
print("=" * 70)

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [col for col in X_train.columns if col not in numeric_features]

# ColumnTransformer: числовые и категориальные столбцы обрабатываются раздельно
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# Pipeline: предобработка + модель в одном объекте.
# class_weight='balanced' — ошибки на редком классе «ушёл» штрафуются сильнее,
# так модель перестаёт «лениться» и чаще предсказывает уход.
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=42)),
])
print(pipeline)

# =====================================================================
# 3. СИСТЕМНЫЙ ПОИСК ЛУЧШИХ ПАРАМЕТРОВ
# =====================================================================
print("\n" + "=" * 70)
print("3. ПОИСК ГИПЕРПАРАМЕТРОВ (GridSearchCV)")
print("=" * 70)

# Сетка гиперпараметров логистической регрессии:
#   C       — сила регуляризации (меньше C = проще модель, меньше переобучение)
#   l1_ratio — тип регуляризации: 0 = L2 (уменьшает все веса),
#              1 = L1 (может обнулять ненужные признаки), 0.5 = смесь (ElasticNet)
# Приставка "model__" означает: параметр относится к шагу "model" в Pipeline.
param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__l1_ratio": [0, 0.5, 1],
    "model__solver": ["saga"],  # saga поддерживает все варианты l1_ratio
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5 блоков

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",  # F1 для класса «ушёл» — баланс между Recall и Precision
    n_jobs=-1,
)
grid.fit(X_train, y_train)

print(f"\nЛучшие параметры (best_params_): {grid.best_params_}")
print(f"Средний F1 на кросс-валидации (best_score_): {grid.best_score_:.4f}")

cv_results = pd.DataFrame(grid.cv_results_)[
    ["param_model__C", "param_model__l1_ratio", "mean_test_score", "std_test_score"]
].sort_values("mean_test_score", ascending=False)
print("\nВсе комбинации (по убыванию F1):")
print(cv_results.to_string(index=False))

# =====================================================================
# 4. ФИНАЛЬНАЯ ОЦЕНКА И ИНТЕРПРЕТАЦИЯ
# =====================================================================
print("\n" + "=" * 70)
print("4. ФИНАЛЬНАЯ ОЦЕНКА")
print("=" * 70)

# Финальная модель с лучшими параметрами, обучаем на всех тренировочных данных
final_model = grid.best_estimator_
final_model.fit(X_train, y_train)

y_pred_new = final_model.predict(X_test)
y_proba_new = final_model.predict_proba(X_test)[:, 1]

print("\nClassification report (новая модель):")
print(classification_report(y_test, y_pred_new, target_names=CLASS_NAMES, digits=3))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_new, display_labels=CLASS_NAMES, cmap="Greens"
)
plt.title("Новая модель (после оптимизации)")
plt.xlabel("Предсказание модели")
plt.ylabel("Реальность")
plt.tight_layout()
plt.show()

# ROC-кривая и AUC для старой и новой модели
auc_old = roc_auc_score(y_test, y_proba_old)
auc_new = roc_auc_score(y_test, y_proba_new)
print(f"AUC-ROC старой модели: {auc_old:.4f}")
print(f"AUC-ROC новой модели:  {auc_new:.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_predictions(y_test, y_proba_old, name="Старая модель", ax=ax)
RocCurveDisplay.from_predictions(y_test, y_proba_new, name="Новая модель", ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Случайное угадывание (AUC = 0.5)")
ax.set_title("ROC-кривая")
ax.legend()
plt.tight_layout()
plt.show()

# Важность признаков: у логистической регрессии это коэффициенты.
# Признаки отмасштабированы/закодированы, поэтому коэффициенты можно сравнивать.
# > 0 — признак повышает вероятность ухода, < 0 — понижает.
feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
feature_names = [n.split("__", 1)[1] for n in feature_names]  # убираем приставки num__/cat__
coefs = final_model.named_steps["model"].coef_[0]

importance = (
    pd.DataFrame({"Признак": feature_names, "Коэффициент": coefs})
    .assign(Важность=lambda d: d["Коэффициент"].abs())
    .sort_values("Важность", ascending=False)
)
print("\nТоп-15 самых важных признаков:")
print(importance.head(15).to_string(index=False))

top = importance.head(15).iloc[::-1]
colors = ["tab:red" if c > 0 else "tab:blue" for c in top["Коэффициент"]]
plt.figure(figsize=(9, 6))
plt.barh(top["Признак"], top["Коэффициент"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Топ-15 признаков (красный — повышает отток, синий — снижает)")
plt.xlabel("Коэффициент логистической регрессии")
plt.tight_layout()
plt.show()

# Итоговая таблица сравнения старой и новой модели
def get_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (ушёл)": precision_score(y_true, y_pred),
        "Recall (ушёл)": recall_score(y_true, y_pred),
        "F1 (ушёл)": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_proba),
    }

comparison = pd.DataFrame({
    "Старая модель": get_metrics(y_test, y_pred_old, y_proba_old),
    "Новая модель": get_metrics(y_test, y_pred_new, y_proba_new),
})
comparison["Изменение"] = comparison["Новая модель"] - comparison["Старая модель"]

print("\nИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ:")
print(comparison.round(4))

new_recall = comparison.loc["Recall (ушёл)", "Новая модель"]
status = "ДОСТИГНУТА" if new_recall >= 0.7 else "НЕ достигнута"
print(f"\nЦель «Recall для класса 'Ушёл' >= 0.7»: {status} (Recall = {new_recall:.3f})")

comparison[["Старая модель", "Новая модель"]].plot(kind="bar", figsize=(9, 5))
plt.title("Сравнение старой и новой модели")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
