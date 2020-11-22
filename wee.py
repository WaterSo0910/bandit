import numpy as np

a = np.eye(3)
b = np.matrix([1, 2, 3])
print(a)
print(b)
print(np.matmul(a,b.T))
for i in range(3):
    print(i)

