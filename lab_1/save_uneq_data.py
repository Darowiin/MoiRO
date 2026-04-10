"""
Генерация и сохранение данных с неравными ковариационными матрицами
(3 класса) для использования в ЛР №2.

Данные генерируются тем же методом, что и в ЛР №1.
"""

import numpy as np
import os


def generate_normal_2d(M, B, N):
    """Генерация двумерного нормального распределения (метод из ЛР №1)."""
    M = np.asarray(M)
    B = np.asarray(B)

    b00, b01 = B[0, 0], B[0, 1]
    b11 = B[1, 1]

    a00 = np.sqrt(b00)
    a10 = b01 / np.sqrt(b00)
    a11 = np.sqrt(b11 - (b01**2) / b00)

    X = np.zeros((N, 2))

    for i in range(N):
        xi1 = np.sum(np.random.rand(12)) - 6
        xi2 = np.sum(np.random.rand(12)) - 6

        x1 = M[0] + a00 * xi1
        x2 = M[1] + a10 * xi1 + a11 * xi2

        X[i, 0] = x1
        X[i, 1] = x2

    return X


if __name__ == '__main__':
    np.random.seed(42)  # Для воспроизводимости

    N = 200

    # Параметры из ЛР №1 (Вариант 13) — неравные ковариационные матрицы
    M1 = [-1.0, 1.0]
    B1 = [[0.5, 0],
        [0, 0.2]]

    M2 = [0.0, -1.0]
    B2 = [[0.5, 0.25],
        [0.25, 0.4]]


    M3 = [2.0, -1.0]
    B3 = [[0.6, -0.25],
        [-0.25, 0.3]]

    X1 = generate_normal_2d(M1, B1, N)
    X2 = generate_normal_2d(M2, B2, N)
    X3 = generate_normal_2d(M3, B3, N)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(script_dir, 'distrib_uneq_1_[-1,1].npy'), X1)
    np.save(os.path.join(script_dir, 'distrib_uneq_2_[0,-1].npy'), X2)
    np.save(os.path.join(script_dir, 'distrib_uneq_3_[2,-1].npy'), X3)

    print("Данные с неравными ковариационными матрицами сохранены:")
    print(f"  X1: shape={X1.shape}, mean={X1.mean(axis=0)}")
    print(f"  X2: shape={X2.shape}, mean={X2.mean(axis=0)}")
    print(f"  X3: shape={X3.shape}, mean={X3.mean(axis=0)}")
