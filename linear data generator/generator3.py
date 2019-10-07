import numpy as np
import matplotlib.pyplot as plt

size = 3400
v1 = np.linspace(-2, 0, size/2) * -20
v2 = np.linspace(2, 5, size/2) * 31
v3 = np.linspace(-10, 3, size/2) * 10
v4 = np.linspace(54, 55, size/2) * 24
v5 = np.linspace(2.2, 15.1, size/2) * -70
v6 = np.linspace(-22, -1, size/2) * -57.5
v7 = np.linspace(-1, 1, size/2) * -90
v8 = np.linspace(-43, -40, size/2) * 40
v9 = np.linspace(13, 42, size/2) * -31.4


class1 = 1.2 * v1[::2] + 0.4 * v2[::2] + -3.9 * v3[::2] + 3.1 * v4[::2] + 2.6 * v5[::2] + -4.5 * v6[::2] + 1.2 * v7[::2] - 0.4 * v8[::2] + -1 * v9[::2] + np.random.randn(*v1[::2].shape) * 0.33 + 0.42
class2 = 1.2 * v1[1::2] + 0.4 * v2[1::2] + -3.9 * v3[1::2] + 3.1 * v4[1::2] + 2.6 * v5[1::2] + -4.5 * v6[1::2] + 1.2 * v7[1::2] - 0.4 * v8[1::2] + -1 * v9[1::2] + np.random.randn(*v1[1::2].shape) * 0.33 - 0.42

f_fd = open("linear4.data", "w")

f_fd.write(str(size) + "\n")
f_fd.write("V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,Class\n")
f_fd.write("Continuous,Continuous,Continuous,Continuous,Continuous,Continuous,Continuous,Continuous,Continuous,Continuous,Nominal\n")

i=0
for c1 in class1:
	f_fd.write(str(v1[i]) +"," + str(v2[i]) +","+ str(v3[i]) +","+ str(v4[i]) +"," + str(v5[i]) +","+ str(v6[i]) + ","+ str(v7[i]) + "," + str(v8[i]) +","+ str(v9[i]) +","+str(c1) +"," + str(0) +'\n')
	i+=2

i=1
for c2 in class2:
	f_fd.write(str(v1[i]) +"," + str(v2[i]) +","+ str(v3[i]) +","+ str(v4[i]) +"," + str(v5[i]) +","+ str(v6[i]) + ","+ str(v7[i]) + "," + str(v8[i]) +","+ str(v9[i]) +","+str(c2) +"," + str(1) +'\n')
	i+=2

f_fd.close()
