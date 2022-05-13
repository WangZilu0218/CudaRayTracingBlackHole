from RayTraceBlackHole import renderBlackHoleOneFrame
import numpy as np

PI = 3.1415926
R = 30

theta = np.arange(0, PI / 4, 0.0015)
x = np.zeros(theta.shape)
y = np.cos(theta) * R
z = np.sin(theta) * R

cameraPos = np.vstack((x, y, z))
lookAt = np.array([0.0, 0.0, 0.0])
up = np.array([-0.3, 1.0, 0.0])

for i in range(cameraPos.shape[1]):
    renderBlackHoleOneFrame(cameraPos[:, i], lookAt, up, i)
