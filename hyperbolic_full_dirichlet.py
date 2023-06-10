from class_3d import *
import matplotlib.pyplot as plt
mesh0 = RectangleMesh(10,10,1,1)

def asinh(x):
    return ln(x+sqrt(x**2+1))


def g1(x, ee):
    return [0,x[1],0]

def g2(x, ee):
    return [ee*asinh(1/ee),x[1],ee*(sqrt(ee**(-2)+1)-1)]

    
def g3(x, ee):
    return [ee*asinh(x[0]/ee),0,ee*(sqrt((x[0]/ee)**2+1)-1)]

def g4(x, ee):
    return [asinh(x[0]/ee),1, ee*(sqrt((x[0]/ee)**2+1)-1)]


def y_init(x):
    return [asinh(x[0]), x[1], cosh(asinh(x[0]))-1]

def f(x, ee):
    return [-(6*x[0]**3-9*ee**2*x[0])/(ee**6*((x[0]/ee)**2+1)**(7/2)),
            0,
            (12*x[0]**2-3*ee**2)/(ee**5*((x[0]/ee)**2+1)**(7/2))]


def phi1(x, ee):
    return [[1/sqrt((x[0]/ee)**2+1), 0, 0], [0,1,0], [x[0]/(ee*sqrt((x[0]/ee)**2+1)),0,0]]

def phi2(x, ee):
    return [[1/sqrt(x[0]**2+1), 0, 0], [0,1,0], [x[0]/sqrt(x[0]**2+1),0,0]]

def phi3(x, ee):
    return [[1/sqrt(x[0]**2+1), 0, x[0]/sqrt(x[0]**2+1)],[0,1,0],[0,0,0]]

def phi4(x, ee):
    return [[1/sqrt(x[0]**2+1), 0, x[0]/sqrt(x[0]**2+1)],[0,1,0],[0,0,0]]


errors_DG =[]
errors_CG = []
r = np.array([5.0,10.0,15,20,25,30])
for j in r:
    mesh0 = RectangleMesh(int(j),int(j),1,1)
    c_DG =  VectorBiharmonicSolver(family="DG", g=[g1,g2,g3,g4], phi=[phi1,phi1,phi1,phi1], mesh0=mesh0, f=f, gamma0=1e5, gamma1=3e6).convergence(y_exact=y_init, index_range=[1], file_name = None)
    errors_DG.append(c_DG)
    c_CG =  VectorBiharmonicSolver(family="CG", g=[g1,g2,g3,g4], phi=[phi1,phi1,phi1,phi1], mesh0=mesh0, f=f, gamma0=1e5, gamma1=3e6).convergence(y_exact=y_init, index_range=[1], file_name = None)
    errors_CG.append(c_CG)
print(errors_DG)
print(errors_CG)
slope_DG, intercept_DG = np.polyfit(np.log(1.0/r), np.log(errors_DG), 1)
slope_CG, intercept_CG = np.polyfit(np.log(1.0/r), np.log(errors_CG), 1)
print(slope_DG, intercept_DG)
print(slope_CG, intercept_CG)

plt.figure()
plt.title("Log-log Plot of Errors against Mesh Fineness")
plt.xlabel("Mesh Fineness")
plt.ylabel("Errors")
plt.loglog(1.0/r, np.array(errors_DG), ".", color="red")
plt.loglog(1.0/r, np.array(errors_CG), ".", color="blue")
plt.loglog(1.0/r, 1.0/1000/r, color="black")
plt.legend(["DG", "CG", r"line $\log(y)=-3 +\log(x)$"])
plt.savefig("convergence_dirichlet_1,2,3")
