import numpy as np
import pandas as pd
from numpy import pi
import matplotlib.pyplot as plt




X=np.loadtxt("./data/prediction_result_encoder.csv",delimiter=",")



fig = plt.figure()
h = plt.scatter(X[:,3],X[:,4])
plt.title('latent space')
plt.show()


from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2])
plt.title('design space')
plt.show()
