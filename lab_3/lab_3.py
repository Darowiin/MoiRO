import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ==========================================
# ИСХОДНЫЕ ДАННЫЕ (Вариант 13)
# ==========================================
M1 = np.array([-1, 1])
M2 = np.array([0, -1])

# Равные ковариационные матрицы
B_eq = np.array([[0.7, 0.0], [0.0, 0.7]])

# Неравные ковариационные матрицы
B1_neq = np.array([[0.5, 0.0], [0.0, 0.2]])
B2_neq = np.array([[0.5, 0.25], [0.25, 0.4]])

N = 200  # Объём выборки

# Загрузка данных из ЛР 1
script_dir = os.path.dirname(os.path.abspath(__file__))
lab1_dir = os.path.join(os.path.dirname(script_dir), 'lab_1')

# Равные матрицы
X1_eq = np.load(os.path.join(lab1_dir, 'distrib_1_[-1,1].npy'))
X2_eq = np.load(os.path.join(lab1_dir, 'distrib_2_[0,-1].npy'))

# Неравные матрицы (классы omega0 и omega1)
X1_neq = np.load(os.path.join(lab1_dir, 'distrib_uneq_1_[-1,1].npy'))
X2_neq = np.load(os.path.join(lab1_dir, 'distrib_uneq_2_[0,-1].npy'))


# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def test_classifier(X1, X2, W, w0):
    """Вычисление эмпирических ошибок линейного классификатора g(x) = W^T x + w0.
    Если g(x) >= 0, то x отнесен к omega0 (класс 1).
    Если g(x) <  0, то x отнесен к omega1 (класс 2)."""
    err1 = np.sum((X1 @ W + w0) < 0) / len(X1)   # Ошибка при классификации omega0
    err2 = np.sum((X2 @ W + w0) >= 0) / len(X2)   # Ошибка при классификации omega1
    return err1, err2, (err1 + err2) / 2


def plot_boundary(X1, X2, W, w0, label, color, linestyle='-', linewidth=2, ax=None):
    """Рисование линейной разделяющей границы W^T x + w0 = 0."""
    if ax is None:
        ax = plt.gca()
    x_min = min(X1[:, 0].min(), X2[:, 0].min()) - 0.5
    x_max = max(X1[:, 0].max(), X2[:, 0].max()) + 0.5
    x_vals = np.array([x_min, x_max])
    if abs(W[1]) > 1e-10:
        y_vals = (-W[0] * x_vals - w0) / W[1]
        ax.plot(x_vals, y_vals, color=color, linestyle=linestyle,
                linewidth=linewidth, label=label)
    else:
        # Вертикальная граница
        x_boundary = -w0 / W[0]
        ax.axvline(x=x_boundary, color=color, linestyle=linestyle,
                   linewidth=linewidth, label=label)


def compute_bayes_params(M1, M2, B):
    """Вычисление параметров байесовского классификатора для равных матриц."""
    invB = np.linalg.inv(B)
    W = invB @ (M1 - M2)
    w0 = -0.5 * M1.T @ invB @ M1 + 0.5 * M2.T @ invB @ M2
    d2 = (M1 - M2).T @ invB @ (M1 - M2)
    d = np.sqrt(d2)
    P_err = norm.cdf(-d / 2)
    return W, w0, P_err


def compute_bayes_params_neq(M1, M2, B1, B2):
    """Вычисление параметров байесовского (квадратичного) классификатора
    для неравных ковариационных матриц. Для сравнения используем
    линейное приближение и эмпирическую оценку ошибки."""
    invB1 = np.linalg.inv(B1)
    invB2 = np.linalg.inv(B2)
    # Линейная часть квадратичного дискриминанта
    W = invB1 @ M1 - invB2 @ M2
    w0 = (-0.5 * M1.T @ invB1 @ M1 + 0.5 * M2.T @ invB2 @ M2
           + 0.5 * np.log(np.linalg.det(B2) / np.linalg.det(B1)))
    return W, w0


# ==========================================
# БАЙЕСОВСКИЙ КЛАССИФИКАТОР (для сравнения)
# ==========================================
# Равные матрицы
W_bayes_eq, w0_bayes_eq, P_err_bayes_eq_th = compute_bayes_params(M1, M2, B_eq)
err1_b_eq, err2_b_eq, err_total_b_eq = test_classifier(X1_eq, X2_eq, W_bayes_eq, w0_bayes_eq)

# Неравные матрицы — квадратичный дискриминант (эмпирическая оценка)
def quad_classify(X, M1, M2, B1, B2):
    """Квадратичная дискриминантная классификация (попарно, два класса)."""
    invB1 = np.linalg.inv(B1)
    invB2 = np.linalg.inv(B2)
    g1 = -0.5 * np.sum((X - M1) @ invB1 * (X - M1), axis=1) - 0.5 * np.log(np.linalg.det(B1))
    g2 = -0.5 * np.sum((X - M2) @ invB2 * (X - M2), axis=1) - 0.5 * np.log(np.linalg.det(B2))
    return g1 >= g2  # True = omega0

bayes_pred_1_neq = quad_classify(X1_neq, M1, M2, B1_neq, B2_neq)
bayes_pred_2_neq = quad_classify(X2_neq, M1, M2, B1_neq, B2_neq)
err1_b_neq = np.sum(~bayes_pred_1_neq) / len(X1_neq)
err2_b_neq = np.sum(bayes_pred_2_neq) / len(X2_neq)
err_total_b_neq = (err1_b_neq + err2_b_neq) / 2


# ==========================================
# 1. КЛАССИФИКАТОР ФИШЕРА
# ==========================================
print("=" * 60)
print("ПУНКТ 1: ЛИНЕЙНЫЙ КЛАССИФИКАТОР ФИШЕРА")
print("=" * 60)


def fisher_classifier(X1, X2):
    """Построение классификатора Фишера.
    Максимизирует критерий Фишера: J(w) = (w^T (m1 - m2))^2 / (w^T S_w w),
    где S_w = S1 + S2 — внутриклассовая матрица рассеяния.
    Решение: w = S_w^{-1} (m1 - m2).
    Порог: w0 = -w^T (m1 + m2) / 2."""
    m1 = np.mean(X1, axis=0)
    m2 = np.mean(X2, axis=0)

    # Внутриклассовые матрицы рассеяния
    S1 = (X1 - m1).T @ (X1 - m1)
    S2 = (X2 - m2).T @ (X2 - m2)
    Sw = S1 + S2

    # Вектор весов Фишера
    W = np.linalg.inv(Sw) @ (m1 - m2)

    # Порог — проекция средней точки
    w0 = -W @ (m1 + m2) / 2

    return W, w0


# --- Равные матрицы ---
print("\n--- Равные корреляционные матрицы ---")
W_fisher_eq, w0_fisher_eq = fisher_classifier(X1_eq, X2_eq)
print(f"  W = [{W_fisher_eq[0]:.6f}, {W_fisher_eq[1]:.6f}]")
print(f"  w0 = {w0_fisher_eq:.6f}")

err1_f_eq, err2_f_eq, err_total_f_eq = test_classifier(X1_eq, X2_eq, W_fisher_eq, w0_fisher_eq)
print(f"  Ошибка omega0: {err1_f_eq:.4f}")
print(f"  Ошибка omega1: {err2_f_eq:.4f}")
print(f"  Суммарная ошибка: {err_total_f_eq:.4f}")
print(f"  Байесовская ошибка (теор.): {P_err_bayes_eq_th:.4f}")
print(f"  Байесовская ошибка (эмпир.): {err_total_b_eq:.4f}")

# --- Неравные матрицы ---
print("\n--- Неравные корреляционные матрицы ---")
W_fisher_neq, w0_fisher_neq = fisher_classifier(X1_neq, X2_neq)
print(f"  W = [{W_fisher_neq[0]:.6f}, {W_fisher_neq[1]:.6f}]")
print(f"  w0 = {w0_fisher_neq:.6f}")

err1_f_neq, err2_f_neq, err_total_f_neq = test_classifier(X1_neq, X2_neq, W_fisher_neq, w0_fisher_neq)
print(f"  Ошибка omega0: {err1_f_neq:.4f}")
print(f"  Ошибка omega1: {err2_f_neq:.4f}")
print(f"  Суммарная ошибка: {err_total_f_neq:.4f}")
print(f"  Байесовская ошибка (эмпир.): {err_total_b_neq:.4f}")


# ==========================================
# 2. КЛАССИФИКАТОР МНК (среднеквадратичная ошибка)
# ==========================================
print("\n" + "=" * 60)
print("ПУНКТ 2: ЛИНЕЙНЫЙ КЛАССИФИКАТОР МНК")
print("=" * 60)


def mse_classifier(X1, X2):
    """Построение классификатора, минимизирующего среднеквадратичную ошибку.
    Формируем расширенный вектор [x, 1] и решаем задачу:
       min || X_aug @ a - y ||^2
    где y = +1 для omega0, y = -1 для omega1.
    Решение: a = (X_aug^T X_aug)^{-1} X_aug^T y (псевдообратная матрица)."""
    N1, N2 = len(X1), len(X2)

    # Расширенная матрица: добавляем столбец единиц
    X_aug = np.vstack([
        np.hstack([X1, np.ones((N1, 1))]),
        np.hstack([X2, np.ones((N2, 1))])
    ])

    # Целевые значения
    y = np.hstack([np.ones(N1), -np.ones(N2)])

    # МНК-решение: a = (X^T X)^{-1} X^T y
    a = np.linalg.lstsq(X_aug, y, rcond=None)[0]

    W = a[:2]   # Вектор весов
    w0 = a[2]   # Порог

    return W, w0


# --- Равные матрицы ---
print("\n--- Равные корреляционные матрицы ---")
W_mse_eq, w0_mse_eq = mse_classifier(X1_eq, X2_eq)
print(f"  W = [{W_mse_eq[0]:.6f}, {W_mse_eq[1]:.6f}]")
print(f"  w0 = {w0_mse_eq:.6f}")

err1_m_eq, err2_m_eq, err_total_m_eq = test_classifier(X1_eq, X2_eq, W_mse_eq, w0_mse_eq)
print(f"  Ошибка omega0: {err1_m_eq:.4f}")
print(f"  Ошибка omega1: {err2_m_eq:.4f}")
print(f"  Суммарная ошибка: {err_total_m_eq:.4f}")
print(f"  Байесовская ошибка (теор.): {P_err_bayes_eq_th:.4f}")
print(f"  Байесовская ошибка (эмпир.): {err_total_b_eq:.4f}")
print(f"  Ошибка Фишера: {err_total_f_eq:.4f}")

# --- Неравные матрицы ---
print("\n--- Неравные корреляционные матрицы ---")
W_mse_neq, w0_mse_neq = mse_classifier(X1_neq, X2_neq)
print(f"  W = [{W_mse_neq[0]:.6f}, {W_mse_neq[1]:.6f}]")
print(f"  w0 = {w0_mse_neq:.6f}")

err1_m_neq, err2_m_neq, err_total_m_neq = test_classifier(X1_neq, X2_neq, W_mse_neq, w0_mse_neq)
print(f"  Ошибка omega0: {err1_m_neq:.4f}")
print(f"  Ошибка omega1: {err2_m_neq:.4f}")
print(f"  Суммарная ошибка: {err_total_m_neq:.4f}")
print(f"  Байесовская ошибка (эмпир.): {err_total_b_neq:.4f}")
print(f"  Ошибка Фишера: {err_total_f_neq:.4f}")


# ==========================================
# 3. ПРОЦЕДУРА РОББИНСА-МОНРО
# ==========================================
print("\n" + "=" * 60)
print("ПУНКТ 3: ПРОЦЕДУРА РОББИНСА-МОНРО")
print("=" * 60)


def robbins_monro(X1, X2, a_init, gamma_func, n_iter=500):
    """Итерационная процедура Роббинса-Монро для построения линейного классификатора.

    Обучение: a(k+1) = a(k) + gamma(k) * (y_k - sign(a^T x_k)) * x_k

    Где:
        a = [w1, w2, w0] — расширенный вектор параметров,
        x_k = [x1, x2, 1] — расширенный вектор признаков,
        y_k = +1 для omega0, -1 для omega1,
        gamma(k) — последовательность корректирующих коэффициентов.

    Возвращает: историю параметров и ошибок."""
    N1, N2 = len(X1), len(X2)

    # Расширенные данные
    X1_aug = np.hstack([X1, np.ones((N1, 1))])
    X2_aug = np.hstack([X2, np.ones((N2, 1))])

    # Объединённая выборка
    X_all = np.vstack([X1_aug, X2_aug])
    y_all = np.hstack([np.ones(N1), -np.ones(N2)])

    a = a_init.copy().astype(float)
    history_a = [a.copy()]
    history_err = []

    for k in range(1, n_iter + 1):
        # Случайный выбор элемента обучающей выборки
        idx = np.random.randint(0, N1 + N2)
        x_k = X_all[idx]
        y_k = y_all[idx]

        gamma_k = gamma_func(k)

        # Градиентный шаг: корректируем в сторону уменьшения ошибки
        prediction = np.sign(a @ x_k)
        if prediction == 0:
            prediction = -1  # Считаем неопределённость ошибкой

        a = a + gamma_k * (y_k - prediction) * x_k

        history_a.append(a.copy())

        # Вычисление текущей ошибки
        W_curr = a[:2]
        w0_curr = a[2]
        _, _, err_curr = test_classifier(X1, X2, W_curr, w0_curr)
        history_err.append(err_curr)

    return a, np.array(history_a), np.array(history_err)


# Различные последовательности gamma(k)
gamma_functions = {
    r'$\gamma_k = 1/k$': lambda k: 1.0 / k,
    r'$\gamma_k = 1/\sqrt{k}$': lambda k: 1.0 / np.sqrt(k),
    r'$\gamma_k = 10/(10+k)$': lambda k: 10.0 / (10 + k),
}

# Различные начальные условия
initial_conditions = {
    'a0 = [0, 0, 0]': np.array([0.0, 0.0, 0.0]),
    'a0 = [1, 1, 0]': np.array([1.0, 1.0, 0.0]),
    'a0 = [-0.5, 0.5, 0.1]': np.array([-0.5, 0.5, 0.1]),
}

n_iter = 1000
np.random.seed(42)

# --- Равные матрицы: исследование gamma ---
print("\n--- Равные матрицы: исследование gamma(k) ---")
fig_rm_eq_gamma, axes_rm_eq_gamma = plt.subplots(1, 3, figsize=(18, 5))
fig_rm_eq_gamma.suptitle('Роббинс-Монро: сходимость при разных gamma(k) (равные матрицы)',
                          fontsize=14)

a0_default = np.array([0.0, 0.0, 0.0])
rm_results_eq = {}

for i, (gamma_name, gamma_func) in enumerate(gamma_functions.items()):
    a_final, hist_a, hist_err = robbins_monro(X1_eq, X2_eq, a0_default, gamma_func, n_iter)
    rm_results_eq[gamma_name] = (a_final, hist_a, hist_err)

    W_rm = a_final[:2]
    w0_rm = a_final[2]
    err1, err2, err_total = test_classifier(X1_eq, X2_eq, W_rm, w0_rm)
    print(f"  {gamma_name}: W=[{W_rm[0]:.4f}, {W_rm[1]:.4f}], w0={w0_rm:.4f}, "
          f"ошибка={err_total:.4f}")

    axes_rm_eq_gamma[i].plot(hist_err, linewidth=0.8)
    axes_rm_eq_gamma[i].axhline(y=err_total_b_eq, color='r', linestyle='--',
                                 label=f'Байес ({err_total_b_eq:.4f})')
    axes_rm_eq_gamma[i].set_title(gamma_name)
    axes_rm_eq_gamma[i].set_xlabel('Итерация')
    axes_rm_eq_gamma[i].set_ylabel('Ошибка классификации')
    axes_rm_eq_gamma[i].legend()
    axes_rm_eq_gamma[i].grid(True, alpha=0.3)
    axes_rm_eq_gamma[i].set_ylim([0, 0.6])

plt.tight_layout()

# --- Равные матрицы: исследование начальных условий ---
print("\n--- Равные матрицы: исследование начальных условий ---")
fig_rm_eq_init, axes_rm_eq_init = plt.subplots(1, 3, figsize=(18, 5))
fig_rm_eq_init.suptitle('Роббинс-Монро: сходимость при разных начальных условиях (равные матрицы)',
                         fontsize=14)

gamma_default = lambda k: 1.0 / k
gamma_default_name = r'$\gamma_k = 1/k$'

for i, (init_name, a0) in enumerate(initial_conditions.items()):
    a_final, hist_a, hist_err = robbins_monro(X1_eq, X2_eq, a0, gamma_default, n_iter)

    W_rm = a_final[:2]
    w0_rm = a_final[2]
    err1, err2, err_total = test_classifier(X1_eq, X2_eq, W_rm, w0_rm)
    print(f"  {init_name}: W=[{W_rm[0]:.4f}, {W_rm[1]:.4f}], w0={w0_rm:.4f}, "
          f"ошибка={err_total:.4f}")

    axes_rm_eq_init[i].plot(hist_err, linewidth=0.8)
    axes_rm_eq_init[i].axhline(y=err_total_b_eq, color='r', linestyle='--',
                                label=f'Байес ({err_total_b_eq:.4f})')
    axes_rm_eq_init[i].set_title(f'{init_name}')
    axes_rm_eq_init[i].set_xlabel('Итерация')
    axes_rm_eq_init[i].set_ylabel('Ошибка классификации')
    axes_rm_eq_init[i].legend()
    axes_rm_eq_init[i].grid(True, alpha=0.3)
    axes_rm_eq_init[i].set_ylim([0, 0.6])

plt.tight_layout()

# --- Неравные матрицы: исследование gamma ---
print("\n--- Неравные матрицы: исследование gamma(k) ---")
fig_rm_neq_gamma, axes_rm_neq_gamma = plt.subplots(1, 3, figsize=(18, 5))
fig_rm_neq_gamma.suptitle('Роббинс-Монро: сходимость при разных gamma(k) (неравные матрицы)',
                           fontsize=14)

rm_results_neq = {}

for i, (gamma_name, gamma_func) in enumerate(gamma_functions.items()):
    a_final, hist_a, hist_err = robbins_monro(X1_neq, X2_neq, a0_default, gamma_func, n_iter)
    rm_results_neq[gamma_name] = (a_final, hist_a, hist_err)

    W_rm = a_final[:2]
    w0_rm = a_final[2]
    err1, err2, err_total = test_classifier(X1_neq, X2_neq, W_rm, w0_rm)
    print(f"  {gamma_name}: W=[{W_rm[0]:.4f}, {W_rm[1]:.4f}], w0={w0_rm:.4f}, "
          f"ошибка={err_total:.4f}")

    axes_rm_neq_gamma[i].plot(hist_err, linewidth=0.8)
    axes_rm_neq_gamma[i].axhline(y=err_total_b_neq, color='r', linestyle='--',
                                  label=f'Байес ({err_total_b_neq:.4f})')
    axes_rm_neq_gamma[i].set_title(gamma_name)
    axes_rm_neq_gamma[i].set_xlabel('Итерация')
    axes_rm_neq_gamma[i].set_ylabel('Ошибка классификации')
    axes_rm_neq_gamma[i].legend()
    axes_rm_neq_gamma[i].grid(True, alpha=0.3)
    axes_rm_neq_gamma[i].set_ylim([0, 0.6])

plt.tight_layout()

# --- Неравные матрицы: исследование начальных условий ---
print("\n--- Неравные матрицы: исследование начальных условий ---")
fig_rm_neq_init, axes_rm_neq_init = plt.subplots(1, 3, figsize=(18, 5))
fig_rm_neq_init.suptitle('Роббинс-Монро: сходимость при разных начальных условиях (неравные матрицы)',
                          fontsize=14)

for i, (init_name, a0) in enumerate(initial_conditions.items()):
    a_final, hist_a, hist_err = robbins_monro(X1_neq, X2_neq, a0, gamma_default, n_iter)

    W_rm = a_final[:2]
    w0_rm = a_final[2]
    err1, err2, err_total = test_classifier(X1_neq, X2_neq, W_rm, w0_rm)
    print(f"  {init_name}: W=[{W_rm[0]:.4f}, {W_rm[1]:.4f}], w0={w0_rm:.4f}, "
          f"ошибка={err_total:.4f}")

    axes_rm_neq_init[i].plot(hist_err, linewidth=0.8)
    axes_rm_neq_init[i].axhline(y=err_total_b_neq, color='r', linestyle='--',
                                 label=f'Байес ({err_total_b_neq:.4f})')
    axes_rm_neq_init[i].set_title(f'{init_name}')
    axes_rm_neq_init[i].set_xlabel('Итерация')
    axes_rm_neq_init[i].set_ylabel('Ошибка классификации')
    axes_rm_neq_init[i].legend()
    axes_rm_neq_init[i].grid(True, alpha=0.3)
    axes_rm_neq_init[i].set_ylim([0, 0.6])

plt.tight_layout()


# ==========================================
# ГРАФИКИ: Разделяющие границы
# ==========================================

# Выбираем лучший результат Роббинса-Монро для отображения на итоговых графиках
# (gamma = 1/k, a0 = [0,0,0])
best_gamma_name = r'$\gamma_k = 1/k$'

a_rm_eq = rm_results_eq[best_gamma_name][0]
W_rm_eq = a_rm_eq[:2]
w0_rm_eq_val = a_rm_eq[2]

a_rm_neq = rm_results_neq[best_gamma_name][0]
W_rm_neq = a_rm_neq[:2]
w0_rm_neq_val = a_rm_neq[2]

# --- Равные матрицы: все границы ---
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.scatter(X1_eq[:, 0], X1_eq[:, 1], c='blue', alpha=0.4, s=20, label=r'$\omega_0$ (равные)')
ax1.scatter(X2_eq[:, 0], X2_eq[:, 1], c='red', alpha=0.4, s=20, label=r'$\omega_1$ (равные)')

plot_boundary(X1_eq, X2_eq, W_bayes_eq, w0_bayes_eq,
              f'Байес (ошибка={err_total_b_eq:.4f})', 'black', '-', 2.5, ax1)
plot_boundary(X1_eq, X2_eq, W_fisher_eq, w0_fisher_eq,
              f'Фишер (ошибка={err_total_f_eq:.4f})', 'green', '--', 2, ax1)
plot_boundary(X1_eq, X2_eq, W_mse_eq, w0_mse_eq,
              f'МНК (ошибка={err_total_m_eq:.4f})', 'orange', '-.', 2, ax1)

err_rm_eq = test_classifier(X1_eq, X2_eq, W_rm_eq, w0_rm_eq_val)[2]
plot_boundary(X1_eq, X2_eq, W_rm_eq, w0_rm_eq_val,
              f'Роббинс-Монро (ошибка={err_rm_eq:.4f})', 'purple', ':', 2, ax1)

ax1.set_title('Разделяющие границы (Равные корреляционные матрицы)', fontsize=14)
ax1.set_xlabel('$x_1$', fontsize=12)
ax1.set_ylabel('$x_2$', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Неравные матрицы: все границы ---
fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.scatter(X1_neq[:, 0], X1_neq[:, 1], c='blue', alpha=0.4, s=20, label=r'$\omega_0$ (неравные)')
ax2.scatter(X2_neq[:, 0], X2_neq[:, 1], c='red', alpha=0.4, s=20, label=r'$\omega_1$ (неравные)')

# Байесовская квадратичная граница (contour)
x_min = min(X1_neq[:, 0].min(), X2_neq[:, 0].min()) - 1
x_max = max(X1_neq[:, 0].max(), X2_neq[:, 0].max()) + 1
y_min = min(X1_neq[:, 1].min(), X2_neq[:, 1].min()) - 1
y_max = max(X1_neq[:, 1].max(), X2_neq[:, 1].max()) + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

invB1 = np.linalg.inv(B1_neq)
invB2 = np.linalg.inv(B2_neq)
g1_grid = -0.5 * np.sum((grid - M1) @ invB1 * (grid - M1), axis=1) - 0.5 * np.log(np.linalg.det(B1_neq))
g2_grid = -0.5 * np.sum((grid - M2) @ invB2 * (grid - M2), axis=1) - 0.5 * np.log(np.linalg.det(B2_neq))
Z = (g1_grid - g2_grid).reshape(xx.shape)
ax2.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2.5)
ax2.plot([], [], 'k-', linewidth=2.5, label=f'Байес квадр. (ошибка={err_total_b_neq:.4f})')

plot_boundary(X1_neq, X2_neq, W_fisher_neq, w0_fisher_neq,
              f'Фишер (ошибка={err_total_f_neq:.4f})', 'green', '--', 2, ax2)
plot_boundary(X1_neq, X2_neq, W_mse_neq, w0_mse_neq,
              f'МНК (ошибка={err_total_m_neq:.4f})', 'orange', '-.', 2, ax2)

err_rm_neq = test_classifier(X1_neq, X2_neq, W_rm_neq, w0_rm_neq_val)[2]
plot_boundary(X1_neq, X2_neq, W_rm_neq, w0_rm_neq_val,
              f'Роббинс-Монро (ошибка={err_rm_neq:.4f})', 'purple', ':', 2, ax2)

ax2.set_title('Разделяющие границы (Неравные корреляционные матрицы)', fontsize=14)
ax2.set_xlabel('$x_1$', fontsize=12)
ax2.set_ylabel('$x_2$', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)


# ==========================================
# ГРАФИК: Эволюция параметров Роббинса-Монро
# ==========================================
fig_params, axes_params = plt.subplots(2, 3, figsize=(18, 10))
fig_params.suptitle('Эволюция параметров Роббинса-Монро (равные матрицы, разные gamma)', fontsize=14)

for i, (gamma_name, (a_final, hist_a, hist_err)) in enumerate(rm_results_eq.items()):
    axes_params[0, i].plot(hist_a[:, 0], label='$w_1$', alpha=0.8)
    axes_params[0, i].plot(hist_a[:, 1], label='$w_2$', alpha=0.8)
    axes_params[0, i].plot(hist_a[:, 2], label='$w_0$', alpha=0.8)
    axes_params[0, i].set_title(f'Параметры: {gamma_name}')
    axes_params[0, i].set_xlabel('Итерация')
    axes_params[0, i].legend()
    axes_params[0, i].grid(True, alpha=0.3)

    axes_params[1, i].plot(hist_err, linewidth=0.8, color='tab:blue')
    axes_params[1, i].axhline(y=err_total_b_eq, color='r', linestyle='--',
                               label=f'Байес ({err_total_b_eq:.4f})', linewidth=1.5)
    axes_params[1, i].set_title(f'Ошибка: {gamma_name}')
    axes_params[1, i].set_xlabel('Итерация')
    axes_params[1, i].set_ylabel('Ошибка')
    axes_params[1, i].legend()
    axes_params[1, i].grid(True, alpha=0.3)
    axes_params[1, i].set_ylim([0, 0.6])

plt.tight_layout()


# ==========================================
# ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА
# ==========================================
print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 60)
print(f"\n{'Классификатор':<25} {'Равные (ошибка)':<20} {'Неравные (ошибка)':<20}")
print("-" * 65)
print(f"{'Байес':<25} {err_total_b_eq:<20.4f} {err_total_b_neq:<20.4f}")
print(f"{'Фишер':<25} {err_total_f_eq:<20.4f} {err_total_f_neq:<20.4f}")
print(f"{'МНК':<25} {err_total_m_eq:<20.4f} {err_total_m_neq:<20.4f}")
print(f"{'Роббинс-Монро':<25} {err_rm_eq:<20.4f} {err_rm_neq:<20.4f}")

plt.show()