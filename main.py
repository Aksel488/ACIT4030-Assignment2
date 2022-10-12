import numpy as np
from matplotlib import pyplot as plt
from utils import read_off


path = 'data/ModelNet10/bathtub/train/bathtub_0001.off'

vertices_bath, faces_bath = read_off(path)
x = [x[0] for x in vertices_bath]
y = [y[1] for y in vertices_bath]
z = [z[2] for z in vertices_bath]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(x, y, z, c='r', marker='o')
# ax.contourf(X=x, Y=y, Z=z)
# plt.show()


