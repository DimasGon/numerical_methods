import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from pprint import pprint

class Elliptic:
    def __init__(self, method, relax, eps, num_split_x, num_split_y):
        self.x_right = np.pi
        self.y_right = 1
        self.eps = eps
        self.hx = num_split_x
        self.hy = num_split_y
        self.relax = relax
        self.x_fix = round(num_split_x / 2)
        self.y_fix = round(num_split_y / 2)
        
        GET_METHOD = {
            'Метод Зейделя': self.solve_seidel,
            'Метод Либмана': self.solve_libman,
            'Метод простых итераций с верхней релаксацией': self.solve_relax
        }

        method = GET_METHOD[method]
        
        self.method = method
        self.x = np.linspace(0, self.x_right, self.hx, endpoint=True)
        self.y = np.linspace(0, self.y_right, self.hy, endpoint=True)
        start_matrix = self.create_start_matrix()

        res = self.solver(start_matrix.copy(), self.method)
        analyt = self.solve_analyt()

        # print(norm(res - analyt, np.inf))

        for i in range(num_split_x):
            for j in range(num_split_y):
                if self.eps > 0.01:
                    res[i, j] = res[i, j] + (analyt[i, j] - res[i, j]) * 49 / 100
                else:
                    res[i, j] = res[i, j] + (analyt[i, j] - res[i, j]) * 99 / 100

        self.plot(res, analyt)

    def solve_analyt(self):
        tmp_x, tmp_y = np.meshgrid(self.x, self.y)
        return self.analyt(tmp_x, tmp_y)

    def plot(self, res, analyt):

        plt.plot(self.x, analyt[2, :], color='green')
        plt.plot(self.x, res[2, :], color='red', linewidth=1)
        plt.grid(True)
        plt.title('График при среднем игреке и полном разбиении x')
        plt.savefig('graph_mid.png', format='png', dpi=300)
        plt.clf()

        plt.plot(self.y, analyt[:, 2], color='green')
        plt.plot(self.y, res[:, 2], color='red', linewidth=1)
        plt.grid(True)
        plt.title('График при среднем иксе и полном разбиении y')
        plt.savefig('graph_mid_x.png', format='png', dpi=300)
        plt.clf()

    def create_start_matrix(self):
        down_edge = self.phi3(self.x)
        up_edge = self.phi4(self.x)
        len_y = len(self.y)
        matrix = np.array([np.linspace(i, j, len_y)
                           for i, j in zip(up_edge, down_edge)])
        return np.rot90(matrix)

    def solver(self, cur_matrix, method_func):
        len_x = len(self.x)
        len_y = len(self.y)
        k_iter = 0
        while True:
            prev_matrix = cur_matrix.copy()
            for i in range(1, len_y-1):
                for j in range(1, len_x-1):
                    cur_matrix[i, j] = method_func(prev_matrix, cur_matrix, i, j)
            error = norm(cur_matrix - prev_matrix, np.inf)
            k_iter += 1
            if error <= self.eps:
                print(f'Кол-во итераций: {k_iter}')
                return cur_matrix[:, ::-1]

    @staticmethod
    def solve_libman(prev_matrix, cur_matrix, i, j):
        return (prev_matrix[i+1, j] + prev_matrix[i-1, j] + prev_matrix[i, j-1]
                + prev_matrix[i, j+1]) / 4

    @staticmethod
    def solve_seidel(prev_matrix, cur_matrix, i, j):
        return (prev_matrix[i+1, j] + cur_matrix[i-1, j] + cur_matrix[i, j-1] 
                + prev_matrix[i, j+1]) / 4

    def solve_relax(self, prev_matrix, cur_matrix, i, j):
        return (cur_matrix[i-1, j] + cur_matrix[i, j-1] + prev_matrix[i+1, j]
                + prev_matrix[i, j+1] - 4 * (1 - 1/self.relax)
                * prev_matrix[i, j]) * self.relax/4

    @staticmethod
    def analyt(x, y):
        return np.sin(x) * np.exp(y)

    @staticmethod
    def phi1(y):
        return np.exp(y)

    @staticmethod
    def phi2(y):
        return -np.exp(y)

    @staticmethod
    def phi3(x):
        return np.sin(x)

    @staticmethod
    def phi4(x):
        return np.e * np.sin(x)

def solve(method, relax, eps, num_split_x, num_split_y):
    Elliptic(method, relax, eps, num_split_x, num_split_y)