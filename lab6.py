# Практическое занятие №6: борьба с переобучением и улучшение регрессионной модели (California Housing)
# Искусственно ухудшаем данные шумовыми признаками, вызываем переобучение и исправляем его Ridge-регрессией.

import numpy as np  # для создания случайных признаков-шумов
import pandas as pd  # для работы с таблицами
import matplotlib.pyplot as plt  # для построения графиков

from sklearn.datasets import fetch_california_housing  # загрузка датасета
from sklearn.model_selection import train_test_split  # разделение на обучающую и тестовую выборки
from sklearn.preprocessing import StandardScaler  # масштабирование признаков
from sklearn.linear_model import LinearRegression, Ridge  # обычная и Ridge-регрессия
from sklearn.metrics import r2_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

N_NOISE = 100  # сколько случайных признаков-шумов добавляем (по заданию 50–100)
ALPHAS = [0.1, 1, 10, 100, 1000]  # значения alpha для Ridge

# 1. ПОДГОТОВКА ДАННЫХ — как в практике №5
print("1. ПОДГОТОВКА ДАННЫХ")

df = fetch_california_housing(as_frame=True).frame
df["MedHouseVal"] = df["MedHouseVal"] * 100  # стоимость в тысячах долларов

features = ["MedInc", "AveRooms", "HouseAge"]  # те же 3 признака, что в практике №5
X = df[features]  # признаки
y = df["MedHouseVal"]  # целевая переменная — стоимость дома

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Обучающая выборка: {X_train.shape[0]} строк, тестовая: {X_test.shape[0]} строк")


def add_noise(X_part, rng):
    """Добавляет к данным N_NOISE столбцов случайного шума — признаков, никак не связанных с ценой."""
    noise = pd.DataFrame(
        rng.normal(size=(len(X_part), N_NOISE)),
        columns=[f"noise_{i}" for i in range(N_NOISE)],
        index=X_part.index,
    )
    return pd.concat([X_part, noise], axis=1)


def run_experiment(X_train, X_test, y_train, y_test):
    """Проводит весь эксперимент: база -> поломка шумом -> спасение Ridge. Возвращает результаты."""
    # Базовая модель: масштабирование + линейная регрессия на 3 признаках.
    # Scaler обучаем только на train, к test лишь применяем.
    scaler = StandardScaler().fit(X_train)
    base = LinearRegression().fit(scaler.transform(X_train), y_train)
    r2_base = r2_score(y_test, base.predict(scaler.transform(X_test)))
    print(f"\nБазовая LinearRegression: R² test = {r2_base:.4f}")

    # СОЗДАЁМ ПРОБЛЕМУ: добавляем шумовые признаки к train и test
    rng = np.random.default_rng(42)
    X_train_noisy = add_noise(X_train, rng)
    X_test_noisy = add_noise(X_test, rng)
    print(f"Добавлено шумовых признаков: {N_NOISE}, всего признаков теперь: {X_train_noisy.shape[1]}")

    # Повторно масштабируем — уже с учётом новых столбцов
    scaler_noisy = StandardScaler().fit(X_train_noisy)
    X_train_noisy_s = scaler_noisy.transform(X_train_noisy)
    X_test_noisy_s = scaler_noisy.transform(X_test_noisy)

    # ДЕМОНСТРАЦИЯ ПОЛОМКИ: обычная линейная регрессия на зашумлённых данных
    broken = LinearRegression().fit(X_train_noisy_s, y_train)
    r2_broken_train = r2_score(y_train, broken.predict(X_train_noisy_s))
    r2_broken = r2_score(y_test, broken.predict(X_test_noisy_s))
    print(f"\n«Сломанная» LinearRegression (с шумом):")
    print(f"  R² train = {r2_broken_train:.4f}")
    print(f"  R² test  = {r2_broken:.4f}  (база была {r2_base:.4f}, изменение {r2_broken - r2_base:+.4f})")

    # ПОИСК И СПАСЕНИЕ: Ridge с разными alpha.
    # Ridge штрафует модель за большие коэффициенты. Чем больше alpha, тем сильнее штраф,
    # тем меньше модель «верит» отдельным признакам — и тем меньше подстраивается под шум.
    ridge_r2 = {}
    for alpha in ALPHAS:
        ridge = Ridge(alpha=alpha).fit(X_train_noisy_s, y_train)
        ridge_r2[alpha] = r2_score(y_test, ridge.predict(X_test_noisy_s))
    print("\nRidge на зашумлённых данных:")
    for alpha, r2 in ridge_r2.items():
        print(f"  alpha = {alpha:<6} R² test = {r2:.4f}")

    best_alpha = max(ridge_r2, key=ridge_r2.get)
    print(f"Лучшее alpha = {best_alpha}, R² test = {ridge_r2[best_alpha]:.4f}")

    return {
        "r2_base": r2_base, "r2_broken": r2_broken, "r2_broken_train": r2_broken_train,
        "ridge_r2": ridge_r2, "best_alpha": best_alpha,
    }


def print_results(res):
    """Итоговая таблица и ответы на вопросы задания."""
    r2_fixed = res["ridge_r2"][res["best_alpha"]]
    drop = res["r2_base"] - res["r2_broken"]
    # Подписи «Катастрофа» / «Спасение» ставим, только если качество действительно заметно изменилось
    broken_label = "Катастрофа!" if drop > 0.05 else "Почти не изменилось"
    fixed_label = "Спасение!" if r2_fixed - res["r2_broken"] > 0.05 else "Почти не изменилось"
    table = pd.DataFrame({
        "Модель/условия": [
            "Базовая (предыдущая практика)",
            "Сломанная (LinearRegression + шум)",
            f"Исправленная (Ridge + шум, alpha = {res['best_alpha']})",
        ],
        "R² на test": [res["r2_base"], res["r2_broken"], r2_fixed],
        "Вывод": ["Наш ориентир", broken_label, fixed_label],
    })
    print("\nИТОГОВАЯ ТАБЛИЦА:")
    print(table.round(4).to_string(index=False))

    print(f"\nНасколько упало качество после добавления шума: на {drop:.4f} "
          f"({drop / res['r2_base']:.1%} от базового R²)")
    print(f"При каком alpha Ridge показал лучший результат: alpha = {res['best_alpha']}")
    print(f"Удалось ли вернуть качество: Ridge дал R² = {r2_fixed:.4f}, база — {res['r2_base']:.4f}, "
          f"разница {r2_fixed - res['r2_base']:+.4f}; "
          f"восстановлено {(r2_fixed - res['r2_broken']) / drop:.0%} потерянного качества" if drop > 0 else "")
    return table


def plot_alpha(res, title, ax):
    """График R² test от alpha, с линиями базовой и сломанной моделей."""
    ax.plot(list(res["ridge_r2"].keys()), list(res["ridge_r2"].values()), "o-", label="Ridge + шум")
    ax.axhline(res["r2_base"], color="green", linestyle="--", label="Базовая модель")
    ax.axhline(res["r2_broken"], color="red", linestyle="--", label="Сломанная модель")
    ax.set_xscale("log")  # alpha меняется в 10 раз — используем логарифмическую шкалу
    ax.set_xlabel("alpha")
    ax.set_ylabel("R² на test")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


# 2–4. ОСНОВНОЙ ЭКСПЕРИМЕНТ — строго по заданию, на всех обучающих данных
print("\n" + "=" * 70)
print("ЭКСПЕРИМЕНТ 1: все обучающие данные")
print("=" * 70)
res_full = run_experiment(X_train, X_test, y_train, y_test)
print_results(res_full)
print(f"""
ВЫВОД ПО ЭКСПЕРИМЕНТУ 1:
Качество почти не упало. Причина — данных очень много: {len(X_train)} обучающих примеров
при {len(features) + N_NOISE} признаках. На таком объёме модель сама «видит», что шумовые признаки
никак не связаны с ценой, и даёт им коэффициенты около нуля. Переобучения почти нет,
поэтому и Ridge почти ничего не меняет: спасать нечего.
Переобучение возникает, когда признаков много, а примеров мало. Проверим это.""")

# ДОПОЛНИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ: та же схема, но обучаемся только на 200 домах.
# Тестовая выборка остаётся той же — сравнение честное.
N_SMALL = 200
print("\n" + "=" * 70)
print(f"ЭКСПЕРИМЕНТ 2: только {N_SMALL} обучающих примеров при {len(features) + N_NOISE} признаках")
print("=" * 70)
X_train_small, y_train_small = X_train.iloc[:N_SMALL], y_train.iloc[:N_SMALL]
res_small = run_experiment(X_train_small, X_test, y_train_small, y_test)
print_results(res_small)
print(f"""
ВЫВОД ПО ЭКСПЕРИМЕНТУ 2:
Теперь видно настоящее переобучение: на обучающих данных R² = {res_small['r2_broken_train']:.3f},
а на тестовых всего {res_small['r2_broken']:.3f}. Модель «выучила» случайный шум, который
на новых данных ничего не значит. Ridge с alpha = {res_small['best_alpha']} не даёт модели
сильно опираться на шумовые признаки и заметно возвращает качество.
Слишком большое alpha (1000) снова ухудшает результат: модель «зажата» так сильно,
что перестаёт использовать и полезные признаки.""")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_alpha(res_full, f"Эксперимент 1: {len(X_train)} обучающих примеров", axes[0])
plot_alpha(res_small, f"Эксперимент 2: {N_SMALL} обучающих примеров", axes[1])
plt.suptitle("R² на test в зависимости от alpha в Ridge")
plt.tight_layout()
plt.show()
