from class_3d import *
import matplotlib.pyplot as plt
mesh0 = RectangleMesh(100,6,10,1)
def asinh(x):
    return ln(x+sqrt(x**2+1))


def g1(x, ee):
    return [0,x[1],0]

def g2(x, ee):
    return [10*(1-ee),x[1],1/64*ee**2]

def y_init(x):
    return [x[0], x[1], -sin(pi*x[0])]

def f(x, ee):
    return [0,0, 0]


def phi1(x, ee):
    return [[10, 0, 0], [0,1,0], [0,0,0]]

def phi2(x, ee):
    return [[-10, 0, 0], [0,1,0], [0,0,0]]


VectorBiharmonicSolver(family="DG", g=[g1,g2,None,None], phi=[None,None,None,None],
                       mesh0=mesh0, f=f, gamma0=1e6, gamma1=1e6, normals="full").continuation_solution(y_init=y_init,
                                                                                       index_range=np.arange(0.5,2.05,0.05),
                                                                                       file_name = "slinky.pvd")



















