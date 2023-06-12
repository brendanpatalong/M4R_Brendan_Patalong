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
    return[asinh(x[0]),x[1], sqrt(x[0]**2+1)-1]

k=8
index_array = np.arange(1,k+1,1)
gamma = (10*np.ones(k))**index_array
gamma_0, gamma_1 = np.meshgrid(gamma, gamma)
DG_error_array = np.zeros((k, k))
CG_error_array = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        DG_error_array[i][j] = VectorBiharmonicSolver(family="DG",g = [g1,g2,None,None], phi = [None]*4,f=f,
                                                      mesh0=mesh0, gamma0 =gamma[i], gamma1=gamma[j]).convergence(y_exact=y_init, index_range = [1])
        CG_error_array[i][j] = VectorBiharmonicSolver(family="CG",g = [g1,g2,None,None], phi = [None]*4,f=f,
                                                      mesh0=mesh0, gamma0 =gamma[i], gamma1=gamma[j]).convergence(y_exact=y_init, index_range = [1])

DG_error_df = pd.DataFrame(index = index_array, columns=index_array, data = np.log(DG_error_array))
CG_error_df = pd.DataFrame(index = index_array, columns=index_array, data = np.log(CG_error_array))

plt.figure()
sns.heatmap(DG_error_df, cmap=sns.color_palette("Blues", as_cmap=True))
plt.xlabel(r"$\log(\gamma_1)$",size=16)
plt.ylabel(r"$\log(\gamma_0)$",size=16)
plt.savefig("heatmap DG 1")

plt.figure()
sns.heatmap(CG_error_df, cmap=sns.color_palette("Blues", as_cmap=True))
plt.xlabel(r"$\log(\gamma_1)$",size=16)
plt.ylabel(r"$\log(\gamma_0)$",size=16)
plt.savefig("heatmap CG 1")












