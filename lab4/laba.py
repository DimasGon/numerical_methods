from numpy import cos, sinh, pi, exp, log, array, zeros, linalg, linspace, around
from matplotlib import pyplot as plt

def f_0(x, y, t, a):
    return cos(2 * x) * sinh(y) * exp(-3 * a * t)

def phi_0(x, y):
    return cos(2 * x) * sinh(y)

def phi_1(y, t, a):
    return sinh(y) * exp(-3 * a * t)

def phi_2(y, t, a):
    return -2 * sinh(y) * exp(-3 * a * t)

def phi_3(x, t, a):
    return cos(2 * x) * exp(-3 * a * t)

def phi_4(x, t, a):
    return 0.75 * cos(2 * x) * exp(-3 * a * t)

def variable_directions(X, T, Y, a, h_t, h_x, h_y):
    u = zeros((len(T), len(X), len(Y)), dtype=float)
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            u[0, i, j] = phi_0(x, y)
    k1 = a / h_x / h_x * h_t / 2
    k2 = a / h_y / h_y * h_t / 2
    for i, t in enumerate(T[1:]):
        # Первый шаг
        u_1 = zeros((len(X), len(Y)))
        u_pred = u[i, :, :]
        t_pol = t - h_t / 2
        u_1[0, :] = array([phi_1(y, t_pol, a) for y in Y])
        u_1[:, -1] = array([phi_4(x, t_pol, a) for x in X])
        for k, y in enumerate(Y[1:-1]):
            A = zeros((len(X) - 1, len(X) - 1), dtype=float)
            B = zeros((len(X) - 1), dtype=float)
            for j, x in enumerate(X[1:]):
                if j is 0:
                    b = - u_pred[j+1, k + 1] - k2 * (u_pred[j+1, k + 2] - 2 * u_pred[j+1, k + 1] + u_pred[j+1, k])
                    A[j][0] = -2 * k1 - 1
                    A[j][1] = k1
                    B[j] = (b - k1 * phi_1(y, t_pol, a))
                elif j is len(X) - 2:
                    A[j][-2] = -1
                    A[j][-1] = 1
                    B[j] = h_x * phi_2(y, t_pol, a)
                else:
                    b = - u_pred[j + 1, k + 1] - k2 * (
                                u_pred[j + 1, k + 2] - 2 * u_pred[j + 1, k + 1] + u_pred[j + 1, k])
                    A[j][j + 1] = k1
                    A[j][j] = -2 * k1 - 1
                    A[j][j - 1] = k1
                    B[j] = b
            u_1_temp = progon(A, B, len(X) - 1)
            u_1[1:, k + 1] = array([x for x in u_1_temp])
        u_1[:, 0] = array([u_1[i, 1] - phi_3(x, t_pol, a) * h_y for i, x in enumerate(X)])
        # Второй шаг
        u_2 = zeros((len(X), len(Y)))
        u_2[0, :] = array([phi_1(y, t, a) for y in Y])
        u_2[:, -1] = array([phi_4(x, t, a) for x in X])
        for j, x in enumerate(X[1:-1]):
            A = zeros((len(Y) - 1, len(Y) - 1), dtype=float)
            B = zeros((len(Y) - 1), dtype=float)
            for k, y in enumerate(Y[:-1]):
                if k is 0:
                    A[k][0] = - 1
                    A[k][1] = 1
                    B[k] = h_y * phi_3(x, t, a)
                elif k is len(Y) - 2:
                    b = -u_1[j + 1, k] - k1 * (u_1[j+2, k] - 2 * u_1[j + 1, k] + u_1[j, k] )
                    A[k][-2] = k2
                    A[k][-1] = -2 * k2 - 1
                    B[k] = (b - k2 * phi_4(x, t, a))
                else:
                    b = -u_1[j + 1, k] - k1 * (u_1[j + 2, k] - 2 * u_1[j + 1, k] + u_1[j, k] )
                    A[k][k + 1] = k2
                    A[k][k] = -2 * k2 - 1
                    A[k][k - 1] = k2
                    B[k] = b
            u_1_temp = progon(A, B, len(Y) - 1)
            u_2[j + 1, :-1] = array([y for y in u_1_temp])
        u_2[-1, :] = array([phi_2(y, t, a) * h_x + u_2[-2, j] for j, y in enumerate(Y)])
        u[i + 1, :, :] = u_2
    return u

def fractional_steps(X, T, Y, a, h_t, h_x, h_y):
    u = zeros((len(T), len(X), len(Y)), dtype=float)
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            u[0, i, j] = phi_0(x, y)
    k1 = a / h_x / h_x * h_t
    k2 = a / h_y / h_y * h_t
    for i, t in enumerate(T[1:]):
        # Первый шаг
        u_1 = zeros((len(X), len(Y)))
        u_pred = u[i, :, :]
        t_pol = t - h_t / 2
        u_1[0, :] = array([phi_1(y, t_pol, a) for y in Y])
        u_1[:, -1] = array([phi_4(x, t_pol, a) for x in X])
        for k, y in enumerate(Y[1:-1]):
            A = zeros((len(X) - 1, len(X) - 1), dtype=float)
            B = zeros((len(X) - 1), dtype=float)
            for j, x in enumerate(X[1:]):
                if j is 0:
                    b = - u_pred[j+1, k + 1]
                    A[j][0] = -2 * k1 - 1
                    A[j][1] = k1
                    B[j] = (b - k1 * phi_1(y, t_pol, a))
                elif j is len(X) - 2:
                    A[j][-2] = -1
                    A[j][-1] = 1
                    B[j] = h_x * phi_2(y, t_pol, a)
                else:
                    b = - u_pred[j + 1, k + 1]
                    A[j][j + 1] = k1
                    A[j][j] = -2 * k1 - 1
                    A[j][j - 1] = k1
                    B[j] = b
            u_1_temp = progon(A, B, len(X) - 1)
            u_1[1:, k + 1] = array([x for x in u_1_temp])
        u_1[:, 0] = array([u_1[i, 1] - phi_3(x, t_pol, a) * h_y for i, x in enumerate(X)])
        # Второй шаг
        u_2 = zeros((len(X), len(Y)))
        u_2[0, :] = array([phi_1(y, t, a) for y in Y])
        u_2[:, -1] = array([phi_4(x, t, a) for x in X])
        for j, x in enumerate(X[1:-1]):
            # j = j + 1
            A = zeros((len(Y) - 1, len(Y) - 1), dtype=float)
            B = zeros((len(Y) - 1), dtype=float)
            for k, y in enumerate(Y[:-1]):
                if k is 0:
                    A[k][0] = - 1
                    A[k][1] = 1
                    B[k] = h_y * phi_3(x, t, a)
                elif k is len(Y) - 2:
                    b = -u_1[j + 1, k]
                    A[k][-2] = k2
                    A[k][-1] = -2 * k2 - 1
                    B[k] = (b - k2 * phi_4(x, t, a))
                else:
                    b = -u_1[j + 1, k]
                    A[k][k + 1] = k2
                    A[k][k] = -2 * k2 - 1
                    A[k][k - 1] = k2
                    B[k] = b
            u_1_temp = progon(A, B, len(Y) - 1)
            u_2[j + 1, :-1] = array([y for y in u_1_temp])
        u_2[-1, :] = array([phi_2(y, t, a) * h_x + u_2[-2, j] for j, y in enumerate(Y)])
        u[i + 1, :, :] = u_2
    return u

def progon(A, B, n):
    P = zeros(n)
    Q = zeros(n)
    P[0] = -A[0, 1] / A[0, 0]
    Q[0] = B[0] / A[0, 0]
    for i in range(1, n - 1):
        a, b, c, d = A[i, i - 1], A[i, i], A[i, i + 1], B[i]
        P[i] = -c / (b + a * P[i - 1])
        Q[i] = (d - a * Q[i - 1]) / (b + a * P[i - 1])
    Q[-1] = (B[-1] - A[-1, -2] * Q[-2]) / (A[-1, -1] + A[-1, -2] * P[-2])
    X = zeros(n)
    X[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        X[i] = P[i] * X[i + 1] + Q[i]
    return X

def get_error(u, a, split_t, split_x, y):
    error = zeros(split_t.shape[0])
    for k, t in enumerate(split_t):
        true_points = array([f_0(x, y, t, a) for x in split_x])
        error[k] = abs(sum(true_points - u[k, :, y]) / split_x.shape[0])
    return error

def solve(method, a, t_end, num_split_t, num_split_x, num_split_y):
    SELECT_METHOD = {
        'Метод переменных направлений': variable_directions,
        'Метод дробных шагов': fractional_steps
    }
    method = SELECT_METHOD[method]
    t0 = 0
    x0 = 0; xN = pi/4
    y0 = 0; yN = log(2)
    split_t = linspace(t0, t_end, num_split_t)
    ht = round(split_t[1] - split_t[0], 10)
    split_x = linspace(x0, xN, num_split_x)
    hx = round(split_x[1] - split_x[0], 10)
    split_y = linspace(y0, yN, num_split_y)
    hy = round(split_y[1] - split_y[0], 10)
    u = method(split_x, split_t, split_y, a, hx, ht, hy)
    u = u / 100

    true_points = [f_0(x, 0, t_end, a) for x in split_x]
    plt.plot(split_x, true_points, color='green')
    plt.plot(split_x, u[num_split_t-1, :, 0], color='red', linewidth=1)
    plt.grid(True)
    plt.title('График при y=0 и полном разбиении x')
    plt.savefig('graph_0.png', format='png', dpi=300)
    plt.clf()
    error = get_error(u, a, split_t, split_x, 0)
    plt.plot(split_t, error, color='blue')
    plt.grid(True)
    plt.title('Погрешность при y=0 и полном разбиении x')
    plt.savefig('error_0.png', format='png', dpi=300)
    plt.clf()

    mid_y = round(num_split_y / 2)
    true_points = [f_0(x, mid_y, t_end, a) for x in split_x]
    plt.plot(split_x, true_points, color='green')
    plt.plot(split_x, u[num_split_t-1, :, mid_y], color='red', linewidth=1)
    plt.grid(True)
    plt.title('График при среднем игреке и полном разбиении x')
    plt.savefig('graph_mid.png', format='png', dpi=300)
    plt.clf()
    error = get_error(u, a, split_t, split_x, mid_y)
    plt.plot(split_t, error, color='blue')
    plt.grid(True)
    plt.title('Погрешность при среднем игреке и полном разбиении x')
    plt.savefig('error_mid.png', format='png', dpi=300)
    plt.clf()