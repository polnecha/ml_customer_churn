# Практическое занятие №5: решение задачи регрессии — оценка стоимости домов (California Housing)
# Датасет встроен в scikit-learn: при первом запуске он скачивается автоматически (нужен интернет),
# потом берётся с компьютера.

import pandas as pd  # для работы с таблицами
import matplotlib.pyplot as plt  # для построения графиков
import seaborn as sns  # для удобных графиков

from sklearn.datasets import fetch_california_housing  # загрузка датасета
from sklearn.model_selection import train_test_split  # разделение на обучающую и тестовую выборки
from sklearn.linear_model import LinearRegression  # линейная регрессия
from sklearn.tree import DecisionTreeRegressor  # дерево решений для регрессии
from sklearn.ensemble import GradientBoostingRegressor  # градиентный бустинг
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# 1. ИССЛЕДОВАНИЕ ДАННЫХ И ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
print("1. ИССЛЕДОВАНИЕ ДАННЫХ")

# Каждая строка — не один дом, а район Калифорнии (данные переписи 1990 года)
df = fetch_california_housing(as_frame=True).frame

# Целевая переменная MedHouseVal — медианная стоимость дома в районе в сотнях тысяч долларов.
# Переводим в тысячи долларов, чтобы было понятнее: 4.526 -> 452.6 тыс. $
df["MedHouseVal"] = df["MedHouseVal"] * 100

print("\nПервые 5 строк:")
print(df.head())
print(f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"Пропусков всего: {df.isnull().sum().sum()}")

print("""
Признаки:
  MedInc     — медианный доход жителей района (в десятках тысяч $: 3.5 = 35 тыс. $ в год)
  HouseAge   — медианный возраст домов в районе (лет)
  AveRooms   — среднее число комнат на дом
  AveBedrms  — среднее число спален на дом
  Population — население района
  AveOccup   — среднее число жильцов на дом
  Latitude, Longitude — широта и долгота района
Целевая переменная:
  MedHouseVal — медианная стоимость дома (тыс. $)""")

print("\nСтатистика целевой переменной MedHouseVal (тыс. $):")
print(df["MedHouseVal"].describe().round(1))

plt.figure(figsize=(8, 4))
sns.histplot(df["MedHouseVal"], bins=50)
plt.title("Распределение стоимости домов")
plt.xlabel("Медианная стоимость дома, тыс. $")
plt.ylabel("Количество районов")
plt.tight_layout()
plt.show()
print("На гистограмме виден высокий столбик справа: в данных стоимость ограничена 500 тыс. $,")
print("все более дорогие дома записаны как 500 тыс. $.")

# Выбираем 3 понятных признака, которые, скорее всего, влияют на цену:
#   MedInc   — чем богаче жители района, тем дороже дома
#   AveRooms — чем больше комнат, тем больше дом и выше цена
#   HouseAge — возраст домов: новые обычно дороже, но старые районы могут быть престижными
features = ["MedInc", "AveRooms", "HouseAge"]

print("\nКорреляция выбранных признаков с ценой (от -1 до 1, чем дальше от 0 — тем сильнее связь):")
print(df[features + ["MedHouseVal"]].corr()["MedHouseVal"].drop("MedHouseVal").round(3))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, feature in zip(axes, features):
    ax.scatter(df[feature], df["MedHouseVal"], s=3, alpha=0.2)
    ax.set_xlabel(feature)
    ax.set_ylabel("Стоимость, тыс. $")
    ax.set_title(f"{feature} и стоимость дома")
# У AveRooms есть редкие выбросы (до 140 комнат) — обрезаем ось, чтобы было видно основную массу
axes[1].set_xlim(0, df["AveRooms"].quantile(0.99))
plt.tight_layout()
plt.show()

# 2. ОБУЧЕНИЕ И ОЦЕНКА ЛИНЕЙНОЙ МОДЕЛИ
print("\n2. ЛИНЕЙНАЯ РЕГРЕССИЯ")

X = df[features]  # признаки — только выбранные 3
y = df["MedHouseVal"]  # правильные ответы — стоимость дома

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Обучающая выборка: {X_train.shape[0]} строк, тестовая: {X_test.shape[0]} строк")


def regression_metrics(y_true, y_pred):
    """Три основные метрики регрессии."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
    }


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
metrics_lr = regression_metrics(y_test, y_pred_lr)

print(f"\nMAE = {metrics_lr['MAE']:.1f} тыс. $ — в среднем модель ошибается на столько")
print(f"MSE = {metrics_lr['MSE']:.1f} — средний квадрат ошибки (сильнее штрафует большие ошибки)")
print(f"R²  = {metrics_lr['R²']:.3f} — модель объясняет {metrics_lr['R²']:.0%} разброса цен "
      f"(1 — идеально, 0 — не лучше, чем всегда называть среднюю цену)")

# Интерпретация коэффициентов
print(f"\nmodel.intercept_ = {lr.intercept_:.2f}")
print("model.coef_:")
for feature, coef in zip(features, lr.coef_):
    print(f"  {feature:9s} {coef:+.2f}")

coef = dict(zip(features, lr.coef_))
print(f"""
Расшифровка для заказчика (при прочих равных условиях):
  • MedInc:   рост медианного дохода жителей района на 10 тыс. $ в год
              меняет прогнозируемую стоимость дома в среднем на {coef['MedInc']:+.1f} тыс. $.
  • AveRooms: увеличение среднего числа комнат на 1
              меняет прогнозируемую стоимость в среднем на {coef['AveRooms']:+.1f} тыс. $.
  • HouseAge: дома в районе старше на 1 год —
              прогнозируемая стоимость меняется в среднем на {coef['HouseAge']:+.1f} тыс. $.
  • intercept_ = {lr.intercept_:.1f} — «стартовая» цена, от которой модель отсчитывает
              (формально — цена при всех признаках, равных 0; сама по себе смысла не имеет).""")
if coef["AveRooms"] < 0:
    print("""  Почему у AveRooms минус, хотя больше комнат — дороже? Число комнат сильно связано с доходом:
  в богатых районах дома больше. Модель уже учла доход через MedInc, поэтому при ОДИНАКОВОМ доходе
  лишние комнаты часто означают дешёвые дома в пригородах и сельской местности.""")

# 3. СРАВНЕНИЕ С НЕЛИНЕЙНЫМИ МОДЕЛЯМИ
print("\n3. СРАВНЕНИЕ С НЕЛИНЕЙНЫМИ МОДЕЛЯМИ")

tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
metrics_tree = regression_metrics(y_test, y_pred_tree)
print(f"DecisionTree:     MAE = {metrics_tree['MAE']:.1f} тыс. $, R² = {metrics_tree['R²']:.3f}")

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
metrics_gb = regression_metrics(y_test, y_pred_gb)
print(f"GradientBoosting: MAE = {metrics_gb['MAE']:.1f} тыс. $, R² = {metrics_gb['R²']:.3f}")

# Таблица сравнения трёх моделей
comparison = pd.DataFrame({
    "LinearRegression": metrics_lr,
    "DecisionTree": metrics_tree,
    "GradientBoosting": metrics_gb,
}).T
print("\nСРАВНЕНИЕ МОДЕЛЕЙ (MAE и MSE — чем меньше, тем лучше; R² — чем больше, тем лучше):")
print(comparison.round(3))

best_name = comparison["R²"].idxmax()
best_pred = {"LinearRegression": y_pred_lr, "DecisionTree": y_pred_tree,
             "GradientBoosting": y_pred_gb}[best_name]
print(f"\nЛучшая модель: {best_name} (R² = {comparison.loc[best_name, 'R²']:.3f}, "
      f"MAE = {comparison.loc[best_name, 'MAE']:.1f} тыс. $)")

# Scatter plot: реальные цены и предсказания лучшей модели
plt.figure(figsize=(7, 7))
plt.scatter(y_test, best_pred, s=4, alpha=0.3)
plt.plot([0, 500], [0, 500], "r--", label="Идеальный прогноз (y = x)")
plt.title(f"Реальная и предсказанная стоимость: {best_name}")
plt.xlabel("Реальная стоимость, тыс. $")
plt.ylabel("Предсказанная стоимость, тыс. $")
plt.xlim(0, 520)
plt.ylim(0, 520)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("""
Как читать график: каждая точка — район из тестовой выборки.
  • Точки на красной линии — модель угадала цену точно.
  • Точки выше линии — модель завысила цену, ниже — занизила.
Чем плотнее точки вокруг линии, тем лучше модель.""")
