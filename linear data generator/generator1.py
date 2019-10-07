import numpy as np
import matplotlib.pyplot as plt

size = 700
x_train = np.linspace(-1, 1, size/2) * 70
y_train = 0.72 * x_train + np.random.randn(*x_train.shape) * 0.4 + 0.4
y_train2 = 0.72 * x_train + np.random.randn(*x_train.shape) * 0.4 - 0.4

print("X:", x_train)

print("Y1:", y_train)

print("Y2:", y_train2)

plt.scatter(x_train, y_train, c ="b")
y_learned = x_train*0.72
plt.scatter(x_train, y_train2, c ="r")
plt.plot(x_train, y_learned, 'g')
plt.show()

f_fd = open("linear2.data", "w")

f_fd.write(str(size) + "\n")
f_fd.write("V1,V2,Class\n")
f_fd.write("Continuous,Continuous,Nominal\n")

i=0
for y1 in y_train:
	f_fd.write(str(x_train[i]) +","+ str(y1) +"," + str(0) +'\n')
	i+=1

i=0
for y2 in y_train2:
	f_fd.write(str(x_train[i]) +","+ str(y2) +"," + str(1) +'\n')
	i+=1

f_fd.close()
