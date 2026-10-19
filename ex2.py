import numpy as np

# Для красивого вывода чисел
np.set_printoptions(precision=4, suppress=True)

def solve_least_squares(K, f, task_name):
    print("=" * 60)
    print(f"Задание {task_name}")
    print("=" * 60)
    
    # 1. Вычисляем точное нормальное решение
    # Формула: x = (K^T * K)^(-1) * K^T * f
    K_T = K.T  # Транспонированная матрица K
    K_T_K = np.dot(K_T, K)  # K^T * K
    K_T_K_inv = np.linalg.inv(K_T_K)  # Обратная матрица (K^T * K)^(-1)
    
    x_exact = np.dot(K_T_K_inv, np.dot(K_T, f))
    
    print("\n[1] ТОЧНОЕ РЕШЕНИЕ")
    print(f"Вектор f (точный): {f}")
    print(f"Решение x: \n{x_exact}")
    
    # 2. Добавляем шум (погрешность) в правую часть f
    # Создаем случайный шум, например, от -0.5 до 0.5
    # Зафиксируем seed, чтобы результаты при перезапуске были одинаковыми
    np.random.seed(42) 
    noise = np.random.uniform(-0.5, 0.5, size=f.shape)
    
    # Округлим шум для красоты
    noise = np.round(noise, 2)
    f_noisy = f + noise
    
    print("\n[2] ЗАШУМЛЕННОЕ РЕШЕНИЕ")
    print(f"Добавленный шум: {noise}")
    print(f"Вектор f (с шумом): {f_noisy}")
    
    # Вычисляем решение для зашумленной системы
    x_noisy = np.dot(K_T_K_inv, np.dot(K_T, f_noisy))
    print(f"Решение x (с шумом): \n{x_noisy}")
    
    # 3. Сравнение решений
    print("\n[3] СРАВНЕНИЕ")
    diff = np.abs(x_exact - x_noisy)
    print(f"Разница (по модулю) между точным и зашумленным x:\n{diff}")
    
    # Проверка невязки (насколько сильно Kx отличается от f)
    residual_exact = np.linalg.norm(np.dot(K, x_exact) - f)
    residual_noisy = np.linalg.norm(np.dot(K, x_noisy) - f_noisy)
    print(f"\nНевязка для точного решения: {residual_exact:.4f}")
    print(f"Невязка для зашумленного решения: {residual_noisy:.4f}")
    print("\n")

# --- Матрицы из задания ---

# Задача 1
K1 = np.array([
    [1, 1],
    [1, -1],
    [2, 1]
], dtype=float)

f1 = np.array([2, 0, 2], dtype=float)

# Задача 2
K2 = np.array([
    [3, 2, 1],
    [4, 3, 1],
    [2, 1, 2],
    [2, 1, 1]
], dtype=float)

f2 = np.array([4, 7, -4, 1], dtype=float)

# Запускаем решение
solve_least_squares(K1, f1, "1 (матрица 3x2)")
solve_least_squares(K2, f2, "2 (матрица 4x3)")
