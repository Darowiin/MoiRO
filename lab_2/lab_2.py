import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from matplotlib.colors import ListedColormap
import os

# ==========================================
# ИСХОДНЫЕ ДАННЫЕ (Вариант 13)
# ==========================================
M1 = np.array([-1, 1])
M2 = np.array([0, -1])
M3 = np.array([2, -1])

# Равные матрицы (задание 1, 2)
B_eq = np.array([[0.7, 0.0], [0.0, 0.7]])

# Неравные матрицы (задание 3)
B1 = np.array([[0.5, 0.0], [0.0, 0.2]])
B2 = np.array([[0.5, 0.25], [0.25, 0.4]])
B3 = np.array([[0.6, -0.25], [-0.25, 0.3]])

N = 200 # Объем выборки

script_dir = os.path.dirname(os.path.abspath(__file__))
lab1_dir = os.path.join(os.path.dirname(script_dir), 'lab_1')

# Задание 1, 2: данные с равными ковариационными матрицами
X1_eq = np.load(os.path.join(lab1_dir, 'distrib_1_[-1,1].npy'))
X2_eq = np.load(os.path.join(lab1_dir, 'distrib_2_[0,-1].npy'))

# Задание 3: данные с неравными ковариационными матрицами
X1 = np.load(os.path.join(lab1_dir, 'distrib_uneq_1_[-1,1].npy'))
X2 = np.load(os.path.join(lab1_dir, 'distrib_uneq_2_[0,-1].npy'))
X3 = np.load(os.path.join(lab1_dir, 'distrib_uneq_3_[2,-1].npy'))

# ==========================================
# ПУНКТ 1 и 2: Равные матрицы (Байес, Минимакс, Нейман-Пирсон)
# ==========================================
print("--- 1-2. ДВА КЛАССА, РАВНЫЕ МАТРИЦЫ ---")

invB = np.linalg.inv(B_eq)
# Вектор весов и порог для Байеса
W = invB @ (M1 - M2)
w0_bayes = -0.5 * M1.T @ invB @ M1 + 0.5 * M2.T @ invB @ M2

# Расстояние Махаланобиса
d2 = (M1 - M2).T @ invB @ (M1 - M2)
d = np.sqrt(d2)

# Теоретическая ошибка Байеса
P_err_bayes_th = norm.cdf(-d/2)
print(f"Байес: Теоретическая ошибка = {P_err_bayes_th:.4f}")

# Нейман-Пирсон (p0 = 0.05)
alpha = 0.05
z_alpha = norm.ppf(1 - alpha)
theta_np = d**2 / 2 - z_alpha * d
w0_np = w0_bayes + (theta_np) # Сдвиг порога

# Эмпирические ошибки
def test_linear_classifier(X1, X2, W, w0):
    err1 = np.sum((X1 @ W + w0) < 0) / len(X1) # Ложно классифицированы как 2
    err2 = np.sum((X2 @ W + w0) >= 0) / len(X2) # Ложно классифицированы как 1
    return err1, err2, (err1 + err2)/2

err1_b, err2_b, err_total_b = test_linear_classifier(X1_eq, X2_eq, W, w0_bayes)
err1_np, err2_np, err_total_np = test_linear_classifier(X1_eq, X2_eq, W, w0_np)

print(f"Байес/Минимакс: Эмпирическая суммарная ошибка = {err_total_b:.4f}")
print(f"Нейман-Пирсон: Эмпирическая ошибка = {err_total_np:.4f} (Ошибка 1 рода: {err1_np:.4f})")

# Отрисовка
plt.figure(figsize=(10, 6))
plt.scatter(X1_eq[:,0], X1_eq[:,1], label='Class 1', alpha=0.5)
plt.scatter(X2_eq[:,0], X2_eq[:,1], label='Class 2', alpha=0.5)

x_vals = np.array(plt.gca().get_xlim())
# y = (-W[0]*x - w0) / W[1]
y_bayes = (-W[0]*x_vals - w0_bayes) / W[1]
y_np = (-W[0]*x_vals - w0_np) / W[1]

plt.plot(x_vals, y_bayes, 'k-', linewidth=2, label='Байес / Минимакс')
plt.plot(x_vals, y_np, 'r--', linewidth=2, label='Нейман-Пирсон (a=0.05)')
plt.title('Разделяющие границы (Равные матрицы)')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# ПУНКТ 3: Неравные матрицы, 3 класса
# ==========================================
print("\n--- 3. ТРИ КЛАССА, НЕРАВНЫЕ МАТРИЦЫ ---")

def quad_discriminant(x, M, B):
    invB = np.linalg.inv(B)
    detB = np.linalg.det(B)
    # Возвращает логарифм правдоподобия
    return -0.5 * np.sum((x - M) @ invB * (x - M), axis=1) - 0.5 * np.log(detB)

# Классификатор
X_all = np.vstack((X1, X2, X3))
y_true = np.array([0]*N + [1]*N + [2]*N)

g1 = quad_discriminant(X_all, M1, B1)
g2 = quad_discriminant(X_all, M2, B2)
g3 = quad_discriminant(X_all, M3, B3)
y_pred = np.argmax(np.vstack((g1, g2, g3)), axis=0)

# Оценка между классами 1 и 2 (как запрошено в п.3)
mask12 = (y_true == 0) | (y_true == 1)
y_true_12 = y_true[mask12]
y_pred_12 = y_pred[mask12]
# Считаем ошибкой всё, что не совпало (даже если улетело в 3 класс)
err_12_emp = np.sum(y_true_12 != y_pred_12) / len(y_true_12)
print(f"Эмпирическая ошибка между Классами 1 и 2: {err_12_emp:.4f}")

# Расчет объема выборки N для погрешности <= 5%
if err_12_emp > 0:
    rel_error = np.sqrt((1 - err_12_emp) / (err_12_emp * len(y_true_12)))
    N_required = (1 - err_12_emp) / (err_12_emp * 0.05**2)
    print(f"Текущая относительная погрешность: {rel_error*100:.1f}%")
    print(f"Для погрешности 5% требуется объем выборки N >= {int(np.ceil(N_required))}")

# Отрисовка границ (contour)
xx, yy = np.meshgrid(np.linspace(-4, 5, 200), np.linspace(-4, 4, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
g1_grid = quad_discriminant(grid_points, M1, B1)
g2_grid = quad_discriminant(grid_points, M2, B2)
g3_grid = quad_discriminant(grid_points, M3, B3)
Z = np.argmax(np.vstack((g1_grid, g2_grid, g3_grid)), axis=0).reshape(xx.shape)

plt.figure(figsize=(10, 6))
cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
plt.scatter(X1[:,0], X1[:,1], color='red', label='Class 1', edgecolor='k', marker='o', alpha=0.7)
plt.scatter(X2[:,0], X2[:,1], color='green', label='Class 2', edgecolor='k', marker='o', alpha=0.7)
plt.scatter(X3[:,0], X3[:,1], color='blue', label='Class 3', edgecolor='k', marker='o', alpha=0.7)
plt.title('Байесовские границы (Неравные матрицы, квадратичный дискриминант)')
plt.legend()
plt.show()