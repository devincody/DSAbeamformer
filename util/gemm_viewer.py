import numpy as np
import matplotlib.pyplot as plt
import sys



e = sys.stdin.readline().split(",")
# print(e)
x = np.array([float(i) for i in e])
r = x[:3]
x = x[3:]

# print (j)
# print(csv.reader(iter(sys.stdin.readline, '')))
# x = np.array([])

plt.plot(range(1,len(x)+1), x/(10**12))
plt.title("SGEMM (%d,n)x(n,%d)"%(r[0], r[1]))
plt.xlabel("Sizeof GEMM")
plt.ylabel("TFLOPS")
plt.show()



