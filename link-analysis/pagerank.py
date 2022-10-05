import numpy as np

np.set_printoptions(precision=3)
N = 11  # A to K (purples from the left)

# todo, initialize r_old and r_new
r_old = np.ones(N) / N
r_new = np.ones(N) * 999

epsilon = 1e-6
beta = 0.85

# todo, set the sparse transition matrix M acoording to the given graph
M = np.zeros((N,N))
M[1,2] = 1    # B -> C
M[2,1] = 1    # C -> B
M[3,0] = 1/2  # D -> A
M[3,1] = 1/2  # D -> B
M[4,1] = 1/3  # E -> B
M[4,3] = 1/3  # E -> D
M[4,5] = 1/3  # E -> F
M[5,1] = 1/2  # F -> B
M[5,4] = 1/2  # F -> E
M[6,1] = 1/2  # G -> B
M[6,4] = 1/2  # G -> E
M[7,1] = 1/2  # H -> B
M[7,4] = 1/2  # H -> E
M[8,1] = 1/2  # I -> B
M[8,4] = 1/2  # I -> E
M[9,4] = 1    # J -> E
M[10,4] = 1   # J -> E

M = M.T  # Edges: column to row

iter_cnt = 0
while True:
    r_new = np.inner(beta*M, r_old)
    r_new += (1-r_new.sum())/N
    if np.linalg.norm(r_new - r_old) < epsilon:
        break
    else:
        r_old = r_new

    iter_cnt += 1
    if iter_cnt >= 100:
        break

print(f'Iteration: {iter_cnt}')
for r, i in zip(r_new, 'ABCDEFGHIJ'):
    print(f'{i}: {r*100:.1f}')
