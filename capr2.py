import numpy as np
from scipy.optimize import linprog

# =========================
# 1. Ввод данных
# =========================
# Целевая функция: максимизация прибыли
c = np.array([500.0, 300.0, 200.0])   # прибыль на единицу, руб./шт.

# Ограничения вида A_ub @ x <= b_ub
# 1) бюджет
# 2) складская площадь
A_ub = np.array([
    [2000.0, 1500.0, 800.0],          # руб./шт.
    [0.020,  0.010,  0.015]           # м3/шт.
])
b_ub = np.array([600000.0, 5.0])

# Границы переменных: (нижняя, верхняя)
bounds = [
    (0, 250),   # x1: видеокарты
    (0, 350),   # x2: CPU
    (0, 450)    # x3: материнские платы
]

# =========================
# 2. Решение задачи
# =========================
# linprog минимизирует, поэтому берём -c
res = linprog(
    c=-c,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=bounds,
    method="highs"
)

print("Статус решателя:", res.status, res.message)
if res.success:
    x_opt = res.x
    Z_opt = -res.fun
    print("\nОптимальный план:")
    print(f"x1 (видеокарты) = {x_opt[0]:.2f} шт.")
    print(f"x2 (CPU)        = {x_opt[1]:.2f} шт.")
    print(f"x3 (материнские платы) = {x_opt[2]:.2f} шт.")
    print(f"Максимальная прибыль Z = {Z_opt:.2f} руб.")

    # =========================
    # 3. Баланс ресурсов
    # =========================
    print("\nБаланс ресурсов:")
    resource_names = ["Бюджет, руб.", "Склад, м3"]
    for i, name in enumerate(resource_names):
        used = A_ub[i] @ x_opt
        remaining = b_ub[i] - used
        active = abs(remaining) <= 1e-6 * max(1.0, abs(b_ub[i]))
        print(f"{name}: доступно={b_ub[i]:.2f}, "
              f"использовано={used:.2f}, "
              f"остаток={remaining:.2f}, "
              f"активно={active}")

    # =========================
    # 4. Проверка ограничений по спросу
    # =========================
    print("\nПроверка верхних границ:")
    for j, (name, ub) in enumerate(zip(
        ["x1", "x2", "x3"],
        [250, 350, 450]
    )):
        print(f"{name}: значение={x_opt[j]:.2f}, верхняя граница={ub}, "
              f"активна={abs(x_opt[j] - ub) <= 1e-6}")

    # =========================
    # 5. Теневые цены (двойственные оценки)
    # =========================
    # Для scipy >= 1.7: res.ineqlin.marginals
    try:
        print("\nТеневые цены ограничений (marginals):")
        for i, name in enumerate(resource_names):
            print(f"{name}: {res.ineqlin.marginals[i]:.6f}")
    except Exception as e:
        print("Теневые цены недоступны:", e)
else:
    print("Решение не найдено.")
