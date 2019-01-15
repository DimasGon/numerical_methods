import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from pprint import pprint

class Elliptic:
    def __init__(self):
        # set params
        self.x_right = np.pi #1
        self.y_right = 1 #np.pi / 2
        self.eps = 0.00001
        self.hx = 40
        self.hy = 20
        self.relax = 1.2
        self.x_fix = 2
        self.y_fix = 2
        self.method = 2

        method_funcs = [self.solve_libman,
                        self.solve_seidel,
                        self.solve_relax]

        self.x = np.linspace(0, self.x_right,
                             self.hx, endpoint=True)
        self.y = np.linspace(0, self.y_right,
                             self.hy, endpoint=True)
        start_matrix = self.create_start_matrix()

        res = self.solver(start_matrix.copy(), method_funcs[self.method])
        analyt = self.solve_analyt()

        print(norm(res - analyt, np.inf))

        # plot
        self.plot_projection(res, analyt)
        self.plot_contour(res, analyt)

        # debug
        # pprint(res.round(5))
        # pprint(m.round(5))

    def solve_analyt(self):
        tmp_x, tmp_y = np.meshgrid(self.x, self.y)
        return self.analyt(tmp_x, tmp_y)

    def plot_contour(self, res, analyt):
        tmp_x, tmp_y = np.meshgrid(self.x, self.y)
        _, ax = plt.subplots(1, 2)
        cntr = ax[0].contour(tmp_x, tmp_y, res, levels=5)
        ax[0].clabel(cntr, inline=1, fontsize=10)
        ax[0].set_title('Numeric')
        cntr = ax[1].contour(tmp_x, tmp_y, analyt, levels=5)
        ax[1].clabel(cntr, inline=1, fontsize=10)
        ax[1].set_title('Analyt')
        plt.show()

    def plot_projection(self, res, analyt):
        _, ax = plt.subplots(1, 2)
        ax[0].plot(self.x, analyt[self.y_fix, :], label='Analyt')
        ax[0].plot(self.x, res[self.y_fix, :], label='Numeric')
        ax[0].set_title(f'y_fix = {self.y[self.y_fix].round(4)}')
        ax[1].plot(self.y, analyt[:, self.x_fix], label='Analyt')
        ax[1].plot(self.y, res[:, self.x_fix], label='Numeric')
        ax[1].set_title(f'x_fix = {self.x[self.x_fix].round(4)}')
        ax[0].legend()
        ax[1].legend()
        plt.show()

    def create_start_matrix(self):
        # var 3
        # l_edge = self.phi2(self.y) # np.e * np.cos(self.y)
        # r_edge = self.phi1(self.y) # np.cos(self.y)
        # len_x = len(self.x)
        # matrix = np.array([np.linspace(i, j, len_x)
        #                    for i, j in zip(l_edge, r_edge)])
        # var 4
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
            # cur_matrix[0, :] = cur_matrix[1, :] - self.phi3(self.x) * self.hy
            # cur_matrix[-1, :] = cur_matrix[-2, :] + self.phi4(self.x) * self.hy
            error = norm(cur_matrix - prev_matrix, np.inf)
            k_iter += 1
            if not k_iter % 100:
                print('100 steps')
            if error <= self.eps:
                print('Error:', error, 'Iter:', k_iter)
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
        # return np.exp(x) * np.cos(y)
        return np.sin(x) * np.exp(y)

    @staticmethod
    def phi1(y):
        # return np.cos(y)
        return np.exp(y)

    @staticmethod
    def phi2(y):
        # return np.e * np.cos(y)
        return -np.exp(y)

    @staticmethod
    def phi3(x):
        # return 0
        return np.sin(x)

    @staticmethod
    def phi4(x):
        # return -np.exp(x)
        return np.e * np.sin(x)

def main():
    Elliptic()


if __name__ == "__main__":
    main()
