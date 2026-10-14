# Практическое занятие №1: прогнозирование оттока клиентов (Telco Customer Churn)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # [ДОБАВЛЕНО для практики №2]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# =====================================================================
# 1. РАЗБОР ДАННЫХ
# =====================================================================
print("=" * 70)
print("1. РАЗБОР ДАННЫХ")
print("=" * 70)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\nПервые 5 строк:")
print(df.head())

print(f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")

print("\nТипы столбцов:")
print(df.dtypes)

# Особенность датасета: TotalCharges хранится как текст, потому что у 11 новых
# клиентов (tenure = 0) там стоит пробел вместо числа. Переводим в число,
# пробелы превратятся в пропуски (NaN).
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nПропуски по столбцам (после перевода TotalCharges в число):")
print(df.isnull().sum())

# Пропусков всего 11 из 7043 — просто удаляем эти строки
df = df.dropna()
print(f"\nПосле удаления пропусков: {df.shape[0]} строк")

# customerID — просто идентификатор, для предсказания он бесполезен
df = df.drop(columns=["customerID"])

# Целевая переменная
print("\nРаспределение целевой переменной Churn:")
churn_counts = df["Churn"].value_counts()
print(churn_counts)
print("\nВ процентах:")
print((df["Churn"].value_counts(normalize=True) * 100).round(1))

plt.figure(figsize=(5, 4))
sns.countplot(x="Churn", data=df)
plt.title("Распределение клиентов: ушли (Yes) / остались (No)")
plt.ylabel("Количество клиентов")
plt.tight_layout()
plt.show()

# =====================================================================
# 2. ПОДГОТОВКА ДАННЫХ К ОБУЧЕНИЮ
# =====================================================================
print("\n" + "=" * 70)
print("2. ПОДГОТОВКА ДАННЫХ")
print("=" * 70)

# Отделяем целевую переменную: y — что предсказываем, X — по чему предсказываем
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})  # 1 — ушёл, 0 — остался

# Числовые и категориальные признаки.
# SeniorCitizen записан числами 0/1, но по смыслу это категория (да/нет).
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [col for col in X.columns if col not in numeric_features]

print(f"\nЧисловые признаки ({len(numeric_features)}): {numeric_features}")
print(f"Категориальные признаки ({len(categorical_features)}): {categorical_features}")

# Делим на обучающую (75%) и тестовую (25%) выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"\nОбучающая выборка: {X_train.shape[0]} строк")
print(f"Тестовая выборка:  {X_test.shape[0]} строк")

# Преобразователь: числовые -> StandardScaler, категориальные -> OneHotEncoder
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# =====================================================================
# 3. ОБУЧЕНИЕ И ПРОВЕРКА МОДЕЛЕЙ
# =====================================================================
print("\n" + "=" * 70)
print("3. ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 70)

models = {
    "Логистическая регрессия": LogisticRegression(max_iter=1000, random_state=42),
    "Дерево решений": DecisionTreeClassifier(random_state=42),
    "k-ближайших соседей": KNeighborsClassifier(),
}

results = {}
predictions = {}
pipelines = {}  # [ДОБАВЛЕНО для практики №2]

for name, model in models.items():
    # Pipeline: сначала предобработка, потом модель.
    # Scaler и encoder обучаются только на train, к test лишь применяются.
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    results[name] = acc
    predictions[name] = y_pred
    pipelines[name] = pipe  # [ДОБАВЛЕНО для практики №2]
    print(f"{name}: Accuracy = {acc:.4f}")

# =====================================================================
# 4. АНАЛИЗ РЕЗУЛЬТАТОВ
# =====================================================================
print("\n" + "=" * 70)
print("4. АНАЛИЗ РЕЗУЛЬТАТОВ")
print("=" * 70)

# Таблица сравнения
results_df = (
    pd.DataFrame({"Модель": list(results.keys()), "Accuracy": list(results.values())})
    .sort_values("Accuracy", ascending=False)
    .reset_index(drop=True)
)
print("\nСравнение моделей:")
print(results_df)

# График сравнения
plt.figure(figsize=(7, 4))
ax = sns.barplot(x="Модель", y="Accuracy", data=results_df)
for i, v in enumerate(results_df["Accuracy"]):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.ylim(0, 1)
plt.title("Сравнение Accuracy моделей на тестовой выборке")
plt.tight_layout()
plt.show()

# Лучшая модель
best_name = results_df.loc[0, "Модель"]
print(f"\nЛучшая модель: {best_name} (Accuracy = {results[best_name]:.4f})")

# Матрица ошибок для лучшей модели
cm = confusion_matrix(y_test, predictions[best_name])
tn, fp, fn, tp = cm.ravel()

print("\nМатрица ошибок:")
print(cm)
print(f"\nВерно предсказаны как лояльные (TN):          {tn}")
print(f"Верно предсказаны как ушедшие (TP):           {tp}")
print(f"ОШИБОЧНО предсказаны как ушедшие (FP):        {fp}  (на самом деле остались)")
print(f"ОШИБОЧНО предсказаны как лояльные (FN):       {fn}  (на самом деле ушли)")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Остался (0)", "Ушёл (1)"])
disp.plot(cmap="Blues")
plt.title(f"Матрица ошибок: {best_name}")
plt.xlabel("Предсказание модели")
plt.ylabel("Реальность")
plt.tight_layout()
plt.show()

# =====================================================================
# [ДОБАВЛЕНО для практики №2]
# Сохраняем лучшую модель и выборки в файл, чтобы загрузить их в lab2.py.
# Для сдачи практики №1 этот блок не нужен.
# =====================================================================
joblib.dump(
    {
        "model_name": best_name,
        "model": pipelines[best_name],
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    },
    "best_model_lab1.joblib",
)
print("\nЛучшая модель сохранена в файл best_model_lab1.joblib")
