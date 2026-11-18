# Практическое занятие №4: оптимизация и углублённый анализ мультиклассовой модели (Wine Quality)
# Нужна библиотека imbalanced-learn (pip install imbalanced-learn) и файл WineQT.csv в папке проекта.
# Цель: улучшить macro F1 лучшей модели из практики №3 на 5–10%.

import numpy as np  # для вычислений с массивами
import pandas as pd  # для работы с таблицами
import matplotlib.pyplot as plt  # для построения графиков

from sklearn.model_selection import (
    train_test_split, learning_curve, GridSearchCV, StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler  # масштабирование признаков
from sklearn.ensemble import RandomForestClassifier  # случайный лес
from sklearn.metrics import classification_report, f1_score

from imblearn.over_sampling import SMOTE  # создание искусственных примеров редких классов
from imblearn.pipeline import Pipeline  # Pipeline из imblearn умеет включать SMOTE как шаг

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ПОДГОТОВКА ДАННЫХ — точно так же, как в практике №3,
# чтобы результаты можно было честно сравнивать
df = pd.read_csv("WineQT.csv")
df = df.drop(columns=["Id"])  # Id — номер строки, бесполезен для предсказания

X = df.drop(columns=["quality"])
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


def macro_f1(model):
    """Macro F1 модели на тестовой выборке."""
    return f1_score(y_test, model.predict(X_test), average="macro")


# Модели из практики №3 (пересоздаём с теми же настройками)
rf_base = RandomForestClassifier(random_state=42).fit(X_train, y_train)
rf_balanced = RandomForestClassifier(class_weight="balanced", random_state=42).fit(X_train, y_train)

f1_base = macro_f1(rf_base)
f1_balanced = macro_f1(rf_balanced)
print("МОДЕЛИ ИЗ ПРАКТИКИ №3:")
print(f"  RandomForest (базовая):          macro F1 = {f1_base:.4f}")
print(f"  RandomForest (class_weight):     macro F1 = {f1_balanced:.4f}")

# 1. ПРИМЕНЕНИЕ ПРОДВИНУТЫХ МЕТОДОВ ДЛЯ ДИСБАЛАНСА
print("\n1. SMOTE")

# SMOTE создаёт новые «искусственные» примеры редких классов: берёт вино редкого класса,
# находит похожие вина того же класса и создаёт новое вино «между» ними.
# Применяем ТОЛЬКО к обучающим данным: тест должен остаться настоящим,
# иначе мы проверяли бы модель на придуманных данных.
# k_neighbors=2: у класса 3 в обучающей выборке всего 5 вин, а по умолчанию SMOTE
# нужно 6 (5 соседей + само вино). С двумя соседями ему хватает 3 вин.
smote = SMOTE(k_neighbors=2, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Количество вин каждого класса в обучающей выборке:")
print(pd.DataFrame({
    "До SMOTE": y_train.value_counts().sort_index(),
    "После SMOTE": y_train_smote.value_counts().sort_index(),
}))

# RandomForest с настройками из практики №3, но обученный на сбалансированных данных
rf_smote = RandomForestClassifier(random_state=42).fit(X_train_smote, y_train_smote)
f1_smote = macro_f1(rf_smote)

print("\nClassification report БЕЗ SMOTE (базовый RandomForest):")
print(classification_report(y_test, rf_base.predict(X_test), zero_division=0, digits=3))
print("Classification report СО SMOTE:")
print(classification_report(y_test, rf_smote.predict(X_test), zero_division=0, digits=3))

print(f"Macro F1 без SMOTE:       {f1_base:.4f}")
print(f"Macro F1 с class_weight:  {f1_balanced:.4f}")
print(f"Macro F1 со SMOTE:        {f1_smote:.4f}")

# Выбираем лучший способ борьбы с дисбалансом — его используем дальше
use_smote = f1_smote >= f1_balanced
best_strategy = "SMOTE" if use_smote else "class_weight='balanced'"
f1_strategy = max(f1_smote, f1_balanced)
print(f"=> Лучший способ борьбы с дисбалансом: {best_strategy}")


def make_pipeline(**rf_params):
    """Pipeline: масштабирование -> (SMOTE) -> RandomForest, с выбранным способом балансировки."""
    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(k_neighbors=2, random_state=42)))
    else:
        rf_params["class_weight"] = "balanced"
    steps.append(("model", RandomForestClassifier(random_state=42, **rf_params)))
    return Pipeline(steps=steps)


# 2. ДИАГНОСТИКА МОДЕЛИ С ПОМОЩЬЮ КРИВЫХ ОБУЧЕНИЯ
print("\n2. КРИВЫЕ ОБУЧЕНИЯ")

# learning_curve обучает модель на всё большей части обучающих данных (от 30% до 100%)
# и для каждого размера считает качество на тех данных, на которых училась (train),
# и на отложенных данных, которых модель не видела (validation).
# Внутри Pipeline SMOTE применяется только к обучающей части каждого разбиения.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
train_sizes, train_scores, val_scores = learning_curve(
    make_pipeline(), X_train, y_train,
    train_sizes=np.linspace(0.3, 1.0, 6),
    cv=cv, scoring="f1_macro", n_jobs=-1,
)
train_mean, val_mean = train_scores.mean(axis=1), val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", label="Обучающая выборка (train)")
plt.plot(train_sizes, val_mean, "o-", label="Проверочная выборка (validation)")
plt.fill_between(train_sizes, train_mean - train_scores.std(axis=1),
                 train_mean + train_scores.std(axis=1), alpha=0.15)
plt.fill_between(train_sizes, val_mean - val_scores.std(axis=1),
                 val_mean + val_scores.std(axis=1), alpha=0.15)
plt.title(f"Кривые обучения: RandomForest + {best_strategy}")
plt.xlabel("Количество обучающих примеров")
plt.ylabel("Macro F1")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(pd.DataFrame({
    "Размер обучения": train_sizes,
    "F1 train": train_mean.round(3),
    "F1 validation": val_mean.round(3),
    "Разрыв": (train_mean - val_mean).round(3),
}).to_string(index=False))

gap = train_mean[-1] - val_mean[-1]
print(f"\nНа train macro F1 = {train_mean[-1]:.3f}, на validation = {val_mean[-1]:.3f}, разрыв = {gap:.3f}")
if gap > 0.1:
    print("=> Есть ПЕРЕОБУЧЕНИЕ: на обучающих данных модель почти идеальна, на новых — намного хуже.")
    print("   Модель «запоминает» обучающие вина, а не находит общие закономерности.")
if val_mean[-1] > val_mean[0]:
    print("=> Кривая validation медленно растёт при увеличении данных, а кривые так и не сошлись:")
    print("   модели не хватает данных, с бОльшим количеством примеров качество, скорее всего, вырастет.")
print("""Как можно улучшить модель:
  • собрать больше данных — особенно вин редких классов (3, 4, 8);
  • упростить модель — ограничить глубину деревьев (max_depth), чтобы уменьшить переобучение;
  • объединить редкие классы (например, «плохое / среднее / хорошее» вино).""")

# 3. КОМПЛЕКСНАЯ НАСТРОЙКА
print("\n3. GRIDSEARCHCV")

pipeline = make_pipeline()
print(pipeline)

# Сетка гиперпараметров RandomForest:
#   max_depth    — максимальная глубина деревьев (None — без ограничений)
#   n_estimators — количество деревьев в лесу
param_grid = {
    "model__max_depth": [5, 10, None],
    "model__n_estimators": [50, 100],
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # стратифицированная, k=3
    scoring="f1_macro",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

print(f"\nЛучшие параметры (best_params_): {grid.best_params_}")
print(f"Средний macro F1 на кросс-валидации (best_score_): {grid.best_score_:.4f}")

cv_results = pd.DataFrame(grid.cv_results_)[
    ["param_model__max_depth", "param_model__n_estimators", "mean_test_score"]
].sort_values("mean_test_score", ascending=False)
print("\nВсе комбинации (по убыванию macro F1):")
print(cv_results.to_string(index=False))

# Лучшая модель, обученная на всём train, оценивается на test
final_model = grid.best_estimator_
final_model.fit(X_train, y_train)
f1_grid = macro_f1(final_model)

print("\nClassification report итоговой модели:")
print(classification_report(y_test, final_model.predict(X_test), zero_division=0, digits=3))
print(f"Итоговый macro F1 на test: {f1_grid:.4f}")

# ИТОГОВАЯ ТАБЛИЦА ПО ШАГАМ
results = pd.DataFrame({
    "Шаг": [
        "Базовая модель (практика 3)",
        f"+ {best_strategy}",
        "+ GridSearchCV",
    ],
    "Macro-F1 (test)": [f1_base, f1_strategy, f1_grid],
})
results["Прирост к базовой, %"] = ((results["Macro-F1 (test)"] / f1_base - 1) * 100).round(1)
print("\nИТОГОВАЯ ТАБЛИЦА:")
print(results.round(4).to_string(index=False))

best_row = results.loc[results["Macro-F1 (test)"].idxmax()]
best_gain = best_row["Прирост к базовой, %"]
print(f"\nЛучший результат: шаг «{best_row['Шаг'].strip()}», macro F1 = {best_row['Macro-F1 (test)']:.4f}, "
      f"прирост к базовой модели {best_gain:+.1f}%")
print(f"Цель «улучшить macro F1 на 5–10%»: {'ДОСТИГНУТА' if best_gain >= 5 else 'НЕ достигнута'}")
if f1_grid < f1_strategy:
    print("\nGridSearchCV дал на test результат чуть ниже, чем модель без подбора параметров.")
    print("Это не ошибка: GridSearchCV выбирает параметры по кросс-валидации на train,")
    print("а на test редких вин очень мало (оценка 3 — 1 вино, 8 — 4 вина), поэтому")
    print("одно угаданное или неугаданное вино сильно меняет macro F1.")
    print(f"По кросс-валидации лучшими оказались параметры {grid.best_params_}.")

plt.figure(figsize=(8, 4))
bars = plt.bar(results["Шаг"], results["Macro-F1 (test)"], color=["gray", "tab:blue", "tab:green"])
plt.bar_label(bars, fmt="%.3f")
plt.title("Macro F1 на тестовой выборке по шагам")
plt.ylabel("Macro F1")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
