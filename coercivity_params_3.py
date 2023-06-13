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
    return[asinh(x[0]),x[1], sqrt(x[0]**2+1)-1-0.001]


gamma0 = 1e4
gamma1 = np.arange(1e4, 1e6,4e4)
CG_error=[]

for i in gamma1:
    CG = VectorBiharmonicSolver(family="CG",g = [g1,g2,None,None], phi = [None]*4,f=f,
                                                      mesh0=mesh0, gamma0 =gamma0, gamma1=i).convergence(y_exact=y_init, index_range = [1])
    CG_error.append(CG)

plt.figure()
plt.title(r"Graph of Log Error against $\gamma_1$")
plt.scatter(gamma1, np.log10(np.array(CG_error)), color="blue")
plt.xlabel(r"$\gamma_1$", size= 12)
plt.ylabel("log error", size = 12)
plt.savefig("gamma_0,scatter")












