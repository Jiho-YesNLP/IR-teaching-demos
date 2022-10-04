import code
import numpy as np

L = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
a = np.array([1, 1, 1, 1, 1])
h = np.array([1, 1, 1, 1, 1])
epsilon = 1e-6

def update(L, a, h):
    a = L.T.dot(h)
    h = L.dot(a)
    a = a / np.linalg.norm(a, 2)
    h = h / np.linalg.norm(h, 2)
    return a, h


diff_a = 999
diff_h = 999
iter_cnt = 0
while diff_a > epsilon or diff_h > epsilon:
    a_old = a
    h_old = h
    iter_cnt += 1
    a, h = update(L, a_old, h_old)
    diff_a = np.linalg.norm(a - a_old, 1)
    diff_h = np.linalg.norm(h - h_old, 1)
    print("iter #{}: diff_a {:.6f}, diff_h {:.6f}".format(iter_cnt, diff_a, diff_h))

np.set_printoptions(precision=3)
print(f'a: {a}')
print(f'h: {h}')


