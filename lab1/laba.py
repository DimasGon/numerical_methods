import numpy as np
import matplotlib.pyplot as plt
import math

APPROXIMATION_TYPE = {
    'type-1': 'двухточечная аппроксимация с первым порядком',
    'type-2': 'трехточечная аппроксимация со вторым порядком',
    'type-3': 'двухточечная аппроксимация со вторым порядком'
}

def analytical_solution(x, t, a):
    """ Точное решение """
    return np.exp(-a * t) * np.sin(x)

def initial_cond(x):
    """Начальное условие"""
    return np.sin(x)

def left_cond(a, t):
    """Граничное условие"""
    return np.exp(-a * t)

def right_cond(a, t):
    """Граничное условие"""
    return -np.exp(-a * t)

def count_time_interval(h, a, T):
    """Подсчет количества временных интервалов"""
    return math.ceil(T * 2 * a / h**2)

def get_error(x_vals, time_result, t, a, K):
    """
    Ошибка расчитывается как норма
    разности аналитического и численного решения

    """
    error = []
    i = 0
    for x in x_vals:
        error.append(time_result[i] - analytical_solution(x, t, a))
        i += 1
    return np.linalg.norm(error)

def explicit_method(a, N, K, L, T, x_vals, approx_type):
    """
    Явный метод
    a - коэффицент теплопроводности
    N - количество интервалов по Х
    K - количетсво интревалов по T
    L - длина стержня
    T - временной промеждуток
    h - шаг по X
    tao - шаг по T

    """
    u = []
    h = L / N
    tao = T / K
    sigma = a * tao / (h ** 2)
    k = 0
    err = []
    for t in np.linspace(0, T, K + 1):
        if t == 0:
            time_result = []
            for x in x_vals:
                time_result.append(initial_cond(x))
            u.append(time_result)
        else:
            time_result = []
            for i in np.arange(1, N):
                val = sigma * u[k - 1][i + 1] + u[k - 1][i] * (1 - 2 * sigma) + sigma * u[k - 1][i - 1]
                time_result.append(val)
            if approx_type == 'type-1':
                time_result.insert(0, time_result[0] - h * left_cond(a, t))
                time_result.append(h * right_cond(a, t) + time_result[N - 1])
            elif approx_type == 'type-2':
                time_result.insert(0, (4 * time_result[0] - 2 * h * left_cond(a, t) - time_result[1]) / 3)
                time_result.append((2 * h * right_cond(a, t) - time_result[N - 2] + 4 * time_result[N - 1]) / 3)
            elif approx_type == 'type-3':
                time_result.insert(0, (2 * a * tao * time_result[0] + (h ** 2) * u[k - 1][0] - 2 * a * h * tao * left_cond(a, t)) / (2 * a * tao + (h ** 2)))
                time_result.append((2 * a * h * tao * right_cond(a, t) + 2 * a * tao * time_result[N - 1] + (h ** 2) * u[k - 1][N]) / (2 * a * tao + (h ** 2)))
            u.append(time_result)
        k += 1
    return u

def combined_method(a, N, K, L, T, x_vals, approx_type, teta):
    u = []
    h = L / N
    tao = T / K
    err = []
    for k, t in enumerate(np.linspace(0, T, K + 1)):
        if t == 0:
            time_result = []
            for x in x_vals:
                time_result.append(initial_cond(x))
            u.append(time_result)
        else:
            A = np.zeros((len(x_vals), len(x_vals)), dtype=float)
            B = np.zeros((len(x_vals)), dtype=float)
            for i, x in enumerate(x_vals):
                if i == 0:
                    if approx_type == 'type-1':
                        A[i][0] = -1 / h
                        A[i][1] = 1 / h
                        B[i] = left_cond(a, t)
                    elif approx_type == 'type-2':
                        A[i][0] = -3 / (2 * h)
                        A[i][1] = 2 / h
                        A[i][2] = -1 / (2 * h)
                        B[i] = left_cond(a, t)
                    elif approx_type == 'type-3':
                        A[i][0] = 2 * a / h + h / tao
                        A[i][1] = -2 * a / h
                        B[i] = h * u[k - 1][0] / tao - left_cond(a, t) * 2 * a
                elif i == N:
                    if approx_type == 'type-1':
                        A[i][i - 1] = -1 / h
                        A[i][i] = 1 / h
                        B[i] = right_cond(a, t)
                    elif approx_type == 'type-2':
                        A[i][i - 2] = 1 / (2 * h)
                        A[i][i - 1] = -2 / h
                        A[i][i] = 3 / (2 * h)
                        B[i] = right_cond(a, t)
                    elif approx_type == 'type-3':
                        A[i][i - 1] = -2 * a / h
                        A[i][i] = 2 * a / h + h / tao
                        B[i] = h * u[k - 1][N] / tao + right_cond(a, t) * 2 * a
                else:
                    A[i][i - 1] = teta * a / (h ** 2)
                    A[i][i] = -2 * teta * a / (h ** 2) - (1 / tao)
                    A[i][i + 1] = teta * a / (h ** 2)
                    B[i] = u[k - 1][i] * (2 * a * (1 - teta) / (h ** 2) - (1 / tao)) - (1 - teta) * a * \
                           u[k - 1][i + 1] / (h ** 2) - a * (1 - teta) * u[k - 1][i - 1] / (h ** 2)
            time_result = list(np.linalg.solve(A, B))
            u.append(time_result)
    return u

def solve(a, n, T, approx_type, scheme):
    exact_solution = []
    L = np.pi
    h = L / n
    K = count_time_interval(h, a, T)
    x_vals = [x for x in np.linspace(0, L, n + 1)]
    # time_vals = [t for t in np.linspace(0, T, K)]
    if scheme == 'Явный':
        result = explicit_method(a, n, K, L, T, x_vals, approx_type)
    elif scheme == 'Неявный':
        teta = 1
        result = combined_method(a, n, K, L, T, x_vals, approx_type, teta)
    else:
        teta = 0.5
        result = combined_method(a, n, K, L, T, x_vals, approx_type, teta)
    for x in x_vals:
        exact_solution.append(analytical_solution(x, T, a))
    
    plt.plot(x_vals, exact_solution, color='green')
    plt.plot(x_vals, result[K], color='red')
    plt.grid(True)
    plt.title('График')
    plt.savefig('graph.png', format='png', dpi=300)
    plt.clf()
    
    error = np.array(exact_solution) - np.array(result[K])
    plt.plot(x_vals, error, color='blue')
    plt.grid(True)
    plt.title('Погрешность')
    plt.savefig('error.png', format='png', dpi=300)
    plt.clf()