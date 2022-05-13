# RayTracingBlackHole
This is a black hole renderer implemented by cuda. It will run automatically on multi-gpus in your system and render a very high resolution image in several seconds.


**Dependencies:** 


minimum cuda 9.0 and opencv 4.0+ required. If your cuda is 11.6+, please download cuda samples and use target_include_derectories() to link /usr/local/cuda/samples/common/inc folder to the target.


**Build on Linux():** 

In project directory
1.mkdir build

2.cd build

3.cmake ..

4.make

**Invoke render function by Python**
A .so library RayTraceBlackHole.cpython-36m-x86_64-linux-gnu.so will be generated in the directory
you can use this .so by python

'''
from RayTraceBlackHole import renderBlackHoleOneFrame
import numpy as np
cameraPos = np.array([0.0, 0.3, -20.0])
lookAt = np.array([0.0, 0.0, 0.0])
up = np.array([-0.3, 1.0, 0.0])
renderBlackHoleOneFrame(cameraPos, lookAt, up, 0)
'''

5.or copy the example python script observeBlackHole.py to build
 
6.python observeBlackHole.py
![out2.jpg](https://github.com/WangZilu0218/CudaRayTracingBlackHole/blob/master/out2.jpg)
