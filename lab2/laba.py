from numpy import exp, cos, array, zeros, sqrt, linalg, around, array_repr, linspace
import matplotlib.pyplot as plt
from math import ceil, pi, sqrt

def f1(x):
    """Вычисление значение на нулевом слое\n\nx - i-ое значение икса из разбиения числовой прямой"""
    return round(exp(2 * x), 10)

def U(x, t):
    """Аналитическое решение\n\nx - i-ое значение икса на разбиении числовой прямой\n\nt - k-ое времени t на разбиении времени"""
    return round(f1(x) * cos(t), 10)

def explicit_method(u, h, t, approximation):
    """Явный метод\n\nu - заполняемая матрица\n\nh - шаг по иксу\n\nt - шаг по времени\n\napproximation - метод аппрокисмации"""
    N = u.shape[1] - 1
    last_layer = u.shape[0] - 1
    h = round(h**2, 10); t = round(t**2, 10) # Возведение h и t в квадрат для экономии времени при расчете слоя
    for k in range(2, last_layer+1):
        for i, _ in enumerate(u[k]):
            if i == 0:
                continue
            elif i == N:
                u = approximation(u, k, sqrt(h), sqrt(t))
                break
            u[k, i] = t/h * u[k-1, i-1] + (2*h - 5*h*t - 2*t)/h * u[k-1, i] + t/h * u[k-1, i+1] - u[k-2, i]
            u[k, i] = round(u[k, i], 10)
    return u

def implicit_method(u, h, t, approximation):
    """Неявный метод\n\nu - заполняемая матрица\n\nh - шаг по иксу\n\nt - шаг по времени\n\napproximation - метод аппрокисмации"""
    N = u.shape[1] - 1
    last_layer = u.shape[0] - 1
    for k in range(2, last_layer+1):
        A = zeros((N+1, N+1))
        B = zeros((N+1))
        for i, _ in enumerate(u[k]):
            if i == 0:
                A[i, 0], A[i, 1], A[i, 2], B[i] = approximation(u, k, h, t, i=0)
            elif i == N:
                A[N, N], A[N, N-1], A[N, N-2], B[N] = approximation(u, k, h, t, i=N)
                break
            else:
                A[i][i - 1] = round(1 / h**2, 10)
                A[i][i] = round(-1/t**2 - 2/h**2 - 5, 10)
                A[i][i + 1] = round(1 / h**2, 10)
                B[i] = round((-2 * u[k-1, i] + u[k-2, i]) / t**2, 10)
        u[k] = around(linalg.solve(A, B), decimals=7)
    return u

def twopoint_approximation__first_order(u, k, h, *args, **kwargs):
    """Двухточечная аппроксимация 1-ого порядка\n\nu - заполняемая матрица\n\nk - слой заполнения\n\nh - шаг по иксу"""
    N = u.shape[1] - 1
    if 'i' in kwargs:
        if kwargs['i'] == 0:
            return 1, round(-1 / (1 + 2*h), 10), 0, 0
        elif kwargs['i'] == N:
            return 1, round(-1 / (1 - 2*h), 10), 0, 0
    else:
        u[k, 0] = round(u[k, 1] / (1 + 2*h), 10)
        u[k, N] = round(u[k, N-1] / (1 - 2*h), 10)
    return u
    
def twopoint_approximation__second_order(u, k, h, t, **kwargs):
    """Двухточечная аппроксимация 2-ого порядка\n\nu - заполняемая матрица\n\nk - слой заполнения\n\nh - шаг по иксу"""
    N = u.shape[1] - 1
    if 'i' in kwargs:
        if kwargs['i'] == 0:
            return round(4*h + 3, 10), -4, 1, 0
        elif kwargs['i'] == N:
            return round(4*h - 3, 10), 4, -1, 0
    else:
        u[k, 0] = (4*u[k, 1] - u[k, 2]) / (4*h + 3)
        u[k, 0] = round(u[k, 0], 10)
        u[k, N] = (u[k, N-2] - 4*u[k, N-1]) / (4*h - 3)
        u[k, N] = round(u[k, N], 10)
    return u

def threepoint_approximation__second_order(u, k, h, t, **kwargs):
    """Трёхточечная аппроксимация 2-ого порядка\n\nu - заполняемая матрица\n\nk - слой заполнения\n\nh - шаг по иксу\n\nt - шаг по времени"""
    N = u.shape[1] - 1
    h_2 = round(h**2, 10); t_2 = round(t**2, 10) # чтобы квадраты не высчитывались каждый раз
    if 'i' in kwargs:
        if kwargs['i'] == 0:
            return list(map(
                lambda j: round(j, 10), [ 2*t_2 + h_2 + 4*h*t_2 + 5*h_2*t_2, -2*t_2, 0, 2*h_2*u[k-1, 0] - h_2*u[k-2, 0] ]
            ))
        elif kwargs['i'] == N:
            return list(map(
                lambda j: round(j, 10), [ 2*t_2 + h_2 - 4*h*t_2 + 5*h_2*t_2, -2*t_2, 0, 2*h_2*u[k-1, N] - h_2*u[k-2, N] ]
            ))
    else:
        u[k, 0] = (-h_2*u[k-2, 0] + 2*h_2*u[k-1, 0] + 2*t_2*u[k, 1]) / (2*t_2 + h_2 + 4*h*t_2 + 5*h_2*t_2)
        u[k, 0] = round(u[k, 0], 10)
        u[k, N] = (-h_2*u[k-2, N] + 2*h_2*u[k-1, N] + 2*t_2*u[k, N-1]) / (2*t_2 + h_2 - 4*h*t_2 + 5*h_2*t_2)
        u[k, N] = round(u[k, N], 10)
    return u

def second_initial_condition__first_order(u, *args):
    """2-ое начальное условие 1-ого порядка\n\nu - заполняемая матрица"""
    u[1] = u[0]
    return u

def second_initial_condition__second_order(u, split_x, t):
    """2-ое начальное условие 2-ого порядка\n\nu - заполняемая матрица\n\nsplit_x - разбиение числовой прямой\n\nt - шаг по времени"""
    u[1] = u[0] * (1 - t**2 / 2)
    return u

def split(start, end, step):
    """Разбиение от start до end с шагом step. Значение end попадает в конец результирующего numpy array"""

    spliting = []
    i = start
    while i < end:
        spliting.append(round(i, 10))
        i = round(i + step, 10)
    spliting.append(end)

    return array(spliting)

def get_error(split_x, split_t, u, N, last_layer):

    error = zeros((last_layer))
    for k, t in enumerate(split_t):
        true_res = zeros((N))
        for i, x in enumerate(split_x):
            true_res[i] = U(x, t)
        error[k] = abs(sum(true_res - u[k]) / N)
    error[last_layer-1] = error[last_layer-2]
    
    return error
    
def solve(method, approximation, second_initial_condition, t_end, num_split_x, sigma):
    """Решатель начально-краевой задачи для дифференциального уравнения гиперболического типа\n\nmethod - схема решения\n\napproximation - аппроксимация производной по x\n\nsecond_initial_condition - аппроксимация второго начального условия\n\nt_end - время окончания\n\nnum_split - количество разбиений икса"""
    SELECT_METHOD = {
        'Явный': explicit_method,
        'Неявный': implicit_method,
    }
    SELECT_APPROXIMATION = {
        'Двухточечная 1-ого порядка': twopoint_approximation__first_order,
        'Двухточечная 2-ого порядка': twopoint_approximation__second_order,
        'Трехточечная 2-ого порядка': threepoint_approximation__second_order,
    }
    SELECT_SECOND_INITIAL_CONDITION = {
        '1-ого порядка': second_initial_condition__first_order,
        '2-ого порядка': second_initial_condition__second_order,
    }
    method = SELECT_METHOD[method]
    approximation = SELECT_APPROXIMATION[approximation]
    second_initial_condition = SELECT_SECOND_INITIAL_CONDITION[second_initial_condition]

    x0 = 0; xN = 1
    t_start = 0
    
    split_x = linspace(x0, xN, num_split_x)
    h = round(split_x[1] - split_x[0], 10)
    t = round(h * sigma, 10)
    split_t = split(t_start, t_end, t)

    N = len(split_x)
    last_layer = len(split_t)
    u = zeros((last_layer, N)) # В u загоняем решение

    tmp = []
    for xi in split_x:
        tmp.append(U(xi, t_end))
    true_points = array(tmp)

    tmp = []
    for xi in split_x:
        tmp.append(f1(xi))
    u[0] = array(tmp) # 0-ой слой

    u = second_initial_condition(u, split_x, t) # 1-ый слой
    u = method(u, h, t, approximation)

    plt.plot(split_x, true_points, color='green')
    plt.plot(split_x, u[last_layer-1], color='red', linewidth=1)
    plt.grid(True)
    plt.title('График')
    plt.savefig('graph.png', format='png', dpi=300)
    plt.clf()

    error = get_error(split_x, split_t, u, N, last_layer)
    plt.plot(split_t, error, color='blue')
    plt.grid(True)
    plt.title('Погрешность')
    plt.savefig('error.png', format='png', dpi=300)
    plt.clf()

if __name__ == '__main__':
    solve('Явный', 'Двухточечная 2-ого порядка', '1-ого порядка', 2 * pi, 50, 0.1)