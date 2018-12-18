from numpy import array, exp, linalg, sin, linspace, zeros, around
import matplotlib.pyplot as plt
from math import pi

def f0(x):
    """Вычисление значение на нулевом слое"""
    return around(sin(x), decimals=10)

def f1_a(a):
    def f1(tk):
        return round(exp(-a * tk), 10)
    return f1

def f2_a(a):
    def f2(tk):
        return round(-exp(-a * tk), 10)
    return f2

def U_a(a):
    def U(xi, tk):
        return round(exp(-a * tk) * sin(xi), 10)
    return U

def select_method(method):
    if method == 'Явный':
        teta = 0
    elif method == 'Неявный':
        teta = 1
    else:
        teta = 0.5
    def calculate_points(u, h, t, a, approximation):
        N = u.shape[1] - 1
        last_layer = u.shape[0] - 1
        h_2 = round(h ** 2, 10); t_2 = round(t ** 2, 10)
        f1 = f1_a(a)
        f2 = f2_a(a)
        if teta == 0:
            for k in range(1, last_layer + 1):
                for i in range(N + 1):
                    if i == 0:
                        continue
                    elif i == N:
                        u = approximation(u, k, h, t, a, f1, f2)
                    else:
                        u[k, i] = round(u[k-1, i] + t * (u[k-1, i+1] - 2*u[k-1, i] + u[k-1, i-1]) / h_2, 10)
        else:
            pass # неявный метод
        return u
    return calculate_points

def twopoint_approximation__first_order(u, k, h, t, a, f1, f2, **kwargs):
    """Двухточечная аппроксимация 1-ого порядка\n\nu - заполняемая матрица\n\nk - слой заполнения\n\nh - шаг по иксу"""
    N = u.shape[1] - 1
    if 'i' in kwargs:
        if kwargs['i'] == 0:
            return None
        elif kwargs['i'] == N:
            return None
    else:
        u[k, 0] = round(u[k, 1] - h * f1(t * k), 10)
        u[k, N] = round(u[k, N-1] + h * f2(t * k), 10)
    return u

def twopoint_approximation__second_order(u, k, h, t, a, f1, f2, **kwargs):
    """Двухточечная аппроксимация 2-ого порядка\n\nu - заполняемая матрица\n\nk - слой заполнения\n\nh - шаг по иксу"""
    N = u.shape[1] - 1
    if 'i' in kwargs:
        if kwargs['i'] == 0:
            return None
        elif kwargs['i'] == N:
            return None
    else:
        u[k, 0] = round((2*a*t) / (2*a*t + h**2) * (u[k, 1] + h**2 / (2*a*t) * u[k-1, 0] - h * f1(t*k)), 10)
        u[k, N] = round((2*a*t) / (2*a*t + h**2) * (u[k, N-1] + h**2 / (2*a*t) * u[k-1, N] + h * f2(t*k)), 10)
    return u

def threepoint_approximation__second_order(u, k, h, t, a, f1, f2, **kwargs):
    """Трёхточечная аппроксимация 2-ого порядка\n\nu - заполняемая матрица\n\nk - слой заполнения\n\nh - шаг по иксу\n\nt - шаг по времени"""
    N = u.shape[1] - 1
    if 'i' in kwargs:
        if kwargs['i'] == 0:
            return None
        elif kwargs['i'] == N:
            return None
    else:
        u[k, 0] = round(1/3 * (4 * u[k, 1] - u[k, 2] - 2 * h * f1(t*k)), 10)
        u[k, N] = round(1/3 * (4 * u[k, N-1] - u[k, N-2] + 2 * h * f2(t*k)), 10)
    return u

# def explicit_method(u, h, t, approximation):
#     """Явный метод\n\nu - заполняемая матрица\n\nh - шаг по иксу\n\nt - шаг по времени\n\napproximation - метод аппрокисмации"""
#     N = u.shape[1] - 1
#     last_layer = u.shape[0] - 1
#     h = h**2; t = t**2 # Возведение h и t в квадрат для экономии времени при расчете слоя
#     for k in range(2, last_layer+1):
#         for i, _ in enumerate(u[k]):
#             if i == 0:
#                 continue
#             elif i == N:
#                 u = approximation(u, k, sqrt(h), sqrt(t))
#                 break
#             u[k, i] = t/h * u[k-1, i-1] + (2*h - 5*h*t - 2*t)/h * u[k-1, i] + t/h * u[k-1, i+1] - u[k-2, i]
#             u[k, i] = u[k, i]
#     return u

# def implicit_method(u, h, t, approximation):
#     """Неявный метод\n\nu - заполняемая матрица\n\nh - шаг по иксу\n\nt - шаг по времени\n\napproximation - метод аппрокисмации"""
#     N = u.shape[1] - 1
#     last_layer = u.shape[0] - 1
#     for k in range(2, last_layer+1):
#         A = zeros((N+1, N+1))
#         B = zeros((N+1))
#         for i, _ in enumerate(u[k]):
#             if i == 0:
#                 A[i, 0], A[i, 1], A[i, 2], B[i] = approximation(u, k, h, t, i=0)
#             elif i == N:
#                 A[N, N], A[N, N-1], A[N, N-2], B[N] = approximation(u, k, h, t, i=N)
#                 break
#             else:
#                 A[i][i - 1] = 1 / h**2
#                 A[i][i] = -1/t**2 - 2/h**2 - 5
#                 A[i][i + 1] = 1 / h**2
#                 B[i] = (-2 * u[k-1, i] + u[k-2, i]) / t**2
#         u[k] = linalg.solve(A, B)
#     return u

def get_error(split_x, split_t, u, N, last_layer, U):
    error = zeros((last_layer))
    for k, t in enumerate(split_t):
        true_res = zeros((N))
        for i, x in enumerate(split_x):
            true_res[i] = round(U(x, t), 10)
        error[k] = round(abs(sum(true_res - u[k]) / N), 10)
    return error
    
def solve(method, approximation, num_split, t_end, sigma, a):
    """Решатель начально-краевой задачи для дифференциального уравнения гиперболического типа\n\nmethod - схема решения\n\napproximation - аппроксимация производной по x\n\nsecond_initial_condition - аппроксимация второго начального условия\n\nt_end - время окончания\n\nnum_split - количество разбиений икса"""

    SELECT_APPROXIMATION = {
        'Двухточечная 1-ого порядка': twopoint_approximation__first_order,
        'Двухточечная 2-ого порядка': twopoint_approximation__second_order,
        'Трехточечная 2-ого порядка': threepoint_approximation__second_order,
    }

    method = select_method(method)
    approximation = SELECT_APPROXIMATION[approximation]
    U = U_a(a)
    x0 = 0; xN = pi
    t_start = 0

    split_x = linspace(x0, xN, num_split)
    h = round(split_x[1] - split_x[0], 10)
    num_split = int(t_end / (h**2 * sigma / a)) + 1
    split_t = linspace(t_start, t_end, num_split)
    t = split_t[1] - split_t[0]

    N = len(split_x)
    last_layer = len(split_t)
    u = zeros((last_layer, N)) # В u загоняем решение

    tmp = []
    for xi in split_x:
        tmp.append(U(xi, t_end))
    true_points = array(tmp)

    tmp = []
    for xi in split_x:
        tmp.append(f0(xi))
    u[0] = array(tmp) # 0-ой слой

    u = method(u, h, t, a, approximation)

    plt.plot(split_x, true_points, color='green')
    plt.plot(split_x, u[last_layer-1], color='red', linewidth=1)
    plt.grid(True)
    plt.title('График')
    plt.savefig('graph.png', format='png', dpi=300)
    plt.clf()

    error = get_error(split_x, split_t, u, N, last_layer, U)
    plt.plot(split_t, error, color='blue')
    plt.grid(True)
    plt.title('Погрешность')
    plt.savefig('error.png', format='png', dpi=300)
    plt.clf()

# if __name__ == '__main__':
#     solve('Явный', 'Двухточечная 2-ого порядка', '1-ого порядка', 2 * pi, 50, 0.1)
