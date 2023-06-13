from class_3d import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

mesh0 = RectangleMesh(15,15,1,1)
def asinh(x):
    return ln(x+sqrt(x**2+1))

def g1(x,ee):
    return [0,x[1], 0]

def g2(x,ee):
    return [asinh(x[0]),x[1], sqrt(x[0]**2+1)-1]

def f(x,ee):
    return [-(6*x[0]**3-9*x[0])/(x[0]**2+1)**(7/2),
            0,
            (12*x[0]**2-3)/(x[0]**2+1)**(7/2)]

def y_init(x):
    return[asinh(x[0]+0.001),x[1], sqrt(x[0]**2+1)-1-0.01]

k=9
index_array_1 = np.arange(1e5,1e6,1e5)
index_array_0 = np.arange(1e8,1e9,1e8)

DG_error_array = np.zeros((k, k))

for i in range(k):
    for j in range(k):
        DG_error_array[i][j] = VectorBiharmonicSolver(family="DG",g = [g1,g2,None,None], phi = [None]*4,f=f,
                                                      mesh0=mesh0, gamma0 =index_array_0[i], gamma1=index_array_1[j]).convergence(y_exact=y_init, index_range = [1])
        
DG_error_df = pd.DataFrame(index = index_array_0, columns=index_array_1, data = np.log(DG_error_array))


plt.figure()
sns.heatmap(DG_error_df, cmap=sns.color_palette("Blues", as_cmap=True))
plt.xlabel(r"$\log(\gamma_1)$",size=16)
plt.ylabel(r"$\log(\gamma_0)$",size=16)
plt.savefig("heatmap DG 2")




