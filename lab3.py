# Практическое занятие №3: многоклассовая классификация — определение качества вина (Wine Quality)
# Датасет: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
# Скачанный файл WineQT.csv нужно положить в папку проекта (рядом с lab1.py)

import pandas as pd  # для работы с таблицами
import matplotlib.pyplot as plt  # для построения графиков
import seaborn as sns  # для удобных графиков

from sklearn.model_selection import train_test_split  # разделение на обучающую и тестовую выборки
from sklearn.preprocessing import StandardScaler  # приведение признаков к одному масштабу
from sklearn.pipeline import Pipeline  # объединяет подготовку данных и модель
from sklearn.ensemble import RandomForestClassifier  # случайный лес — первая модель
from sklearn.linear_model import LogisticRegression  # логистическая регрессия — вторая модель
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

pd.set_option("display.max_columns", None)  # показывать все столбцы
pd.set_option("display.width", 200)  # больше ширины для вывода таблиц

# 1. ИССЛЕДОВАНИЕ ДАННЫХ И ПОДГОТОВКА
print("1. ИССЛЕДОВАНИЕ ДАННЫХ")

df = pd.read_csv("WineQT.csv")  # загрузка файла с данными

print("\nПервые 5 строк:")
print(df.head())

# Id — просто номер строки, для предсказания бесполезен
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

print(f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("\nПропуски по столбцам:")
print(df.isnull().sum())

# Целевая переменная quality — оценка качества вина (целое число).
# Каждое значение оценки — это отдельный класс, поэтому задача многоклассовая.
X = df.drop(columns=["quality"])  # признаки — химические показатели вина
y = df["quality"]  # правильные ответы — оценка качества

classes = sorted(int(c) for c in y.unique())
print(f"\nКоличество признаков: {X.shape[1]}")
print(f"Признаки: {list(X.columns)}")
print(f"\nКоличество классов: {len(classes)}")
print(f"Классы (оценки качества): {classes}")

# Базовый EDA: распределение классов
print("\nРаспределение классов:")
class_counts = y.value_counts().sort_index()
class_share = (y.value_counts(normalize=True).sort_index() * 100).round(1)
print(pd.DataFrame({"Количество": class_counts, "Доля, %": class_share}))

ratio = class_counts.max() / class_counts.min()
print(f"\nСамый частый класс встречается в {ratio:.0f} раз чаще самого редкого")
print("=> Есть сильный дисбаланс классов: большинство вин — средние (5 и 6),")
print("   очень плохих и очень хороших вин мало.")

plt.figure(figsize=(7, 4))
ax = sns.countplot(x=y)
for container in ax.containers:
    ax.bar_label(container)  # подписи количества над столбцами
plt.title("Распределение вин по оценке качества")
plt.xlabel("Оценка качества (quality)")
plt.ylabel("Количество вин")
plt.tight_layout()
plt.show()

# Разделение на train/test.
# stratify=y — в обеих выборках доли классов будут такими же, как во всём датасете.
# Без этого редкие классы могли бы целиком попасть только в одну из выборок.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nОбучающая выборка: {X_train.shape[0]} строк")
print(f"Тестовая выборка:  {X_test.shape[0]} строк")
print("\nКоличество объектов каждого класса в тестовой выборке:")
print(y_test.value_counts().sort_index())

# 2. ОБУЧЕНИЕ И БАЗОВАЯ ОЦЕНКА
print("\n2. ОБУЧЕНИЕ И БАЗОВАЯ ОЦЕНКА")


def evaluate(name, model):
    """Обучает модель, выводит accuracy, отчёт, матрицу ошибок и возвращает отчёт в виде словаря."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    # zero_division=0: если модель ни разу не предсказала класс, precision = 0 (без предупреждений)
    print(classification_report(y_test, y_pred, zero_division=0, digits=3))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=classes, cmap="Blues")
    plt.title(f"Матрица ошибок: {name}")
    plt.xlabel("Предсказанная оценка")
    plt.ylabel("Настоящая оценка")
    plt.tight_layout()
    plt.show()

    return classification_report(y_test, y_pred, zero_division=0, output_dict=True)


# Случайный лес: много деревьев решений голосуют вместе. Масштабирование ему не нужно.
rf = RandomForestClassifier(random_state=42)

# Логистической регрессии нужно масштабирование, поэтому объединяем со StandardScaler
lr = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000, random_state=42)),
])

reports = {
    "RandomForest": evaluate("RandomForest", rf),
    "LogisticRegression": evaluate("LogisticRegression", lr),
}

# Анализ: какой класс предсказывается хуже всего
print("\nRecall по классам (доля вин класса, которые модель угадала):")
recall_table = pd.DataFrame({
    name: [rep[str(c)]["recall"] for c in classes] for name, rep in reports.items()
}, index=classes)
recall_table["Объектов в тесте"] = [int(reports["RandomForest"][str(c)]["support"]) for c in classes]
print(recall_table.round(3))

# Худший класс — с наименьшим средним recall у двух моделей
# (при равенстве берём тот, у которого меньше объектов)
avg_recall = recall_table[["RandomForest", "LogisticRegression"]].mean(axis=1)
worst_class = (
    pd.DataFrame({"recall": avg_recall, "support": recall_table["Объектов в тесте"]})
    .sort_values(["recall", "support"])
    .index[0]
)
print(f"\nХуже всего предсказывается класс {worst_class}: "
      f"recall RF = {recall_table.loc[worst_class, 'RandomForest']:.3f}, "
      f"recall LR = {recall_table.loc[worst_class, 'LogisticRegression']:.3f}, "
      f"объектов в тесте: {recall_table.loc[worst_class, 'Объектов в тесте']}, "
      f"в обучающей выборке: {(y_train == worst_class).sum()}.")
print("Причина: примеров этого класса очень мало, модель почти не видела их при обучении.")
print("Ей «выгоднее» относить такие вина к частым классам 5 и 6 — так она реже ошибается в целом.")
print("В матрице ошибок видно, что вина этого класса попадают в столбцы соседних оценок.")

# 3. СРАВНЕНИЕ И ВЫБОР СТРАТЕГИИ
print("\n3. СРАВНЕНИЕ И ВЫБОР СТРАТЕГИИ")

# Macro avg F1 — среднее F1 по всем классам, где каждый класс весит одинаково,
# независимо от количества объектов. Поэтому эта метрика показывает,
# насколько хорошо модель работает и с редкими классами тоже.
for name, rep in reports.items():
    print(f"Macro F1 {name}: {rep['macro avg']['f1-score']:.4f}")

better = max(reports, key=lambda n: reports[n]["macro avg"]["f1-score"])
print(f"=> С балансом между классами лучше справляется {better}")

# RandomForest с class_weight='balanced':
# ошибки на редких классах штрафуются сильнее, чем на частых
rf_balanced = RandomForestClassifier(class_weight="balanced", random_state=42)
reports["RandomForest (balanced)"] = evaluate("RandomForest (balanced)", rf_balanced)

# Итоговая таблица сравнения
summary = pd.DataFrame({
    name: {
        "Accuracy": rep["accuracy"],
        "Macro F1": rep["macro avg"]["f1-score"],
        f"Recall класса {worst_class}": rep[str(worst_class)]["recall"],
    }
    for name, rep in reports.items()
}).T
print("\nИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ:")
print(summary.round(4))

rf_rep, rfb_rep = reports["RandomForest"], reports["RandomForest (balanced)"]
print(f"\nВлияние class_weight='balanced' на RandomForest:")
print(f"  Recall класса {worst_class}: {rf_rep[str(worst_class)]['recall']:.3f} -> "
      f"{rfb_rep[str(worst_class)]['recall']:.3f}")
print(f"  Macro F1:        {rf_rep['macro avg']['f1-score']:.3f} -> "
      f"{rfb_rep['macro avg']['f1-score']:.3f}")

# Recall по всем классам для трёх моделей — видно, какие классы стали находиться лучше
print("\nRecall по классам для всех моделей:")
recall_all = pd.DataFrame({
    name: [rep[str(c)]["recall"] for c in classes] for name, rep in reports.items()
}, index=classes)
print(recall_all.round(3))

summary.plot(kind="bar", figsize=(9, 5))
plt.title("Сравнение моделей")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Вывод: выбираем модель, которая лучше всего находит редкий класс,
# а при равенстве — с более высоким macro F1
best = summary.sort_values([f"Recall класса {worst_class}", "Macro F1"], ascending=False).index[0]
print("\nВЫВОД:")
print(f"Если важно не пропустить представителей редкого класса, выбираем модель {best}:")
print(f"  macro F1 = {summary.loc[best, 'Macro F1']:.3f} — лучший результат среди моделей.")
if summary[f"Recall класса {worst_class}"].max() == 0:
    print(f"  Класс {worst_class} не нашла ни одна модель: в обучающей выборке всего "
          f"{(y_train == worst_class).sum()} таких вин, этого слишком мало для обучения.")
    print("  Для него нужно больше данных или специальные методы (например, SMOTE).")
print("Accuracy здесь не главная метрика: она высокая, даже если модель вообще не находит")
print("редкие классы, ведь их мало. Важнее recall редкого класса и macro F1.")
