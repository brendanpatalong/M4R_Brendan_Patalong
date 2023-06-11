from class_3d import *
import matplotlib.pyplot as plt
mesh0 = RectangleMesh(100,5,6,.1)
def asinh(x):
    return ln(x+sqrt(x**2+1))


def g1(x, ee):
    return [ee,x[1],0]

def g2(x, ee):
    return [x[0],cos(ee)*(x[1]-1/2)+1/2,sin(ee)*(x[1]-1/2)+1/2]

def y_init(x):
    return [x[0],x[1],0] 
    

def f(x, ee):
    return [sin(x[0]), cos(x[0]), 0]


def phi1(x, ee):
    return [[1, 0, 0], [0,1,0], [0,0,0]]

def phi2(x, ee):
    return [[1, 0, 0], [0,1,0], [0,0,0]]

#round1 = VectorBiharmonicSolver(family="DG", g=[g1,g2,None,None], phi=[None,None,None,None],
 #                               mesh0=mesh0,f=f,gamma0=1e4,gamma1=3e6,normals="full").continuation_solution(y_init=y_init,
  #                                                                                                          index_range = np.arange(0,0.5,.02), file_name("turn.pvd"))

VectorBiharmonicSolver(family="DG", g=[None,g2,None,None], phi=[phi1,None,None,None],
                       mesh0=mesh0, f=f, gamma0=1e3, gamma1=3e3, normals="+").continuation_solution(y_init=y_init,
                                                                                       index_range=np.arange(0.01,8,0.05),
                                                                                       file_name = "twirl.pvd")



















