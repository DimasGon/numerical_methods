from math import cos, pi
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, pi/2]

for i in lst:
    print(round(i, 10), round(i%pi, 10) < round(pi/2, 10))