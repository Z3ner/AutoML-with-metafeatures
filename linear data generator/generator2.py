import numpy as np
import matplotlib.pyplot as plt

size = 1002
v1 = np.linspace(-1, 1, size/2) * -15
v2 = np.linspace(-1, 1, size/2) * 3
v3 = np.linspace(-1, 1, size/2) * 5.75

class1 = -4 * v1 + -0.2 * v2 + 3 * v3 + np.random.randn(*v1.shape) * 0.33 + 0.4
class2 = -4 * v1 + -0.2 * v2 + 3 * v3 + np.random.randn(*v1.shape) * 0.33 - 0.4

f_fd = open("linear3.data", "w")

f_fd.write(str(size) + "\n")
f_fd.write("V1,V2,V3,V4,Class\n")
f_fd.write("Continuous,Continuous,Continuous,Continuous,Nominal\n")

i=0
for c1 in class1:
	f_fd.write(str(v1[i]) +"," + str(v2[i]) +","+ str(v3[i]) +","+ str(c1) +"," + str(0) +'\n')
	i+=1

i=0
for c2 in class2:
	f_fd.write(str(v1[i]) +"," + str(v2[i]) +","+ str(v3[i]) +","+ str(c2) +"," + str(1) +'\n')
	i+=1

f_fd.close()
