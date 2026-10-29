import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

alternatives = ['A1', 'A2', 'A3', 'A4', 'A5']
criteria = ['K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8']

# Нормированная матрица R из ЛР №4: чем больше, тем лучше
R = np.array([
    [1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
    [0.5000, 0.4286, 0.6667, 0.5000, 0.5714, 0.4286, 0.4286, 0.7500],
    [0.0000, 0.8571, 0.3333, 0.8750, 1.0000, 0.8571, 0.1429, 0.2500],
    [0.6667, 0.5714, 0.8333, 0.6250, 0.7143, 0.5714, 0.5714, 0.7500],
    [0.5000, 1.0000, 0.0000, 1.0000, 0.1429, 1.0000, 1.0000, 0.0000],
])

# Матрица парных сравнений критериев
high_idx = {0, 2, 5, 7}  # K1, K3, K6, K8
n = len(criteria)
C = np.ones((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            C[i, j] = 1
        elif i in high_idx and j not in high_idx:
            C[i, j] = 2
        elif i not in high_idx and j in high_idx:
            C[i, j] = 0.5
        else:
            C[i, j] = 1

RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
      6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

def ahp_priority(matrix):
    A = np.array(matrix, dtype=float)
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real)
    lam = vals[k].real
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    n = A.shape[0]
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    cr = ci / RI[n] if RI.get(n, 0) > 0 else 0.0
    return w, lam, ci, cr

w_crit, lam_c, ci_c, cr_c = ahp_priority(C)

print('Веса критериев:')
print(pd.DataFrame({'Критерий': criteria, 'Вес': w_crit}).round(4))
print(f'lambda_max = {lam_c:.4f}, CI = {ci_c:.4f}, CR = {cr_c:.4f}')

# Локальные приоритеты альтернатив по каждому критерию
alt_priority = np.zeros((len(alternatives), len(criteria)))
cons_rows = []

for j, crit in enumerate(criteria):
    s = 1 + 8 * R[:, j]          # интенсивность 1..9
    A_alt = s[:, None] / s[None, :]
    w_alt, lam_alt, ci_alt, cr_alt = ahp_priority(A_alt)
    alt_priority[:, j] = w_alt
    cons_rows.append([crit, lam_alt, ci_alt, cr_alt])

print('\nЛокальные приоритеты альтернатив:')
print(pd.DataFrame(alt_priority, index=alternatives, columns=criteria).round(4))

print('\nСогласованность матриц альтернатив:')
print(pd.DataFrame(cons_rows, columns=['Критерий', 'lambda_max', 'CI', 'CR']).round(4))

# Глобальные приоритеты
glob = alt_priority @ w_crit

result = pd.DataFrame({
    'Альтернатива': alternatives,
    'AHP': glob
}).sort_values('AHP', ascending=False)
result['Ранг'] = range(1, len(result) + 1)
print('\nГлобальные приоритеты AHP:')
print(result.round(4))

# Сравнение с ЛР №4: взвешенная свёртка и TOPSIS
w_lab4 = np.array([0.20, 0.10, 0.15, 0.10, 0.10, 0.15, 0.10, 0.15])
w_lab4 = w_lab4 / w_lab4.sum()   # важно: в ЛР4 сумма 1.05, нормируем

S = R @ w_lab4
D_plus = np.sqrt(((R - 1) ** 2 * w_lab4).sum(axis=1))
D_minus = np.sqrt((R ** 2 * w_lab4).sum(axis=1))
C_topsis = D_minus / (D_plus + D_minus)

compare = pd.DataFrame({
    'Альтернатива': alternatives,
    'Взвешенная_свёртка': S,
    'TOPSIS': C_topsis,
    'AHP': glob
})
compare['Ранг_WS'] = compare['Взвешенная_свёртка'].rank(ascending=False).astype(int)
compare['Ранг_TOPSIS'] = compare['TOPSIS'].rank(ascending=False).astype(int)
compare['Ранг_AHP'] = compare['AHP'].rank(ascending=False).astype(int)
print('\nСравнение методов:')
print(compare.round(4))

# Анализ чувствительности для K1 и K3
def global_with_weights(w):
    return alt_priority @ w

def scenario_weights(base, idx, delta):
    u = base.copy()
    u[idx] *= (1 + delta)
    return u / u.sum()

scen_rows = []
for name, idx in [('K1', 0), ('K3', 2)]:
    for delta in [0.10, 0.20, -0.10, -0.20]:
        w_new = scenario_weights(w_crit, idx, delta)
        g_new = global_with_weights(w_new)
        best_i = np.argmax(g_new)
        scen_rows.append([
            f'{name} {delta:+.0%}',
            alternatives[best_i],
            g_new[best_i],
            'Нет' if alternatives[best_i] == 'A4' else 'Да'
        ])

scen_df = pd.DataFrame(scen_rows, columns=[
    'Сценарий', 'Лучшая альтернатива', 'Приоритет лидера', 'Изменился ли лидер'
])
print('\nАнализ чувствительности:')
print(scen_df.round(4))
