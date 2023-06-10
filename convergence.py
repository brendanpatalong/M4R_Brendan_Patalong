from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

def asinh(x):
    return ln(x + sqrt(1.0 + x**2))

def convergence(mesh0=RectangleMesh(10, 10, 1, 1), family = "DG", gamma_0= 100,gamma_1=100):
    # create a 2d mesh in 2d
    # create a 2d mesh embedded in 3d
    Vx = VectorFunctionSpace(mesh0, family, 2, dim=3)
    x0, y0 = SpatialCoordinate(mesh0)
    X = Function(Vx).interpolate(as_vector([x0, y0, 0]))
    mesh = Mesh(X)

    # create the vector function space to define p, f, g1 and  g2 (scalar boundaries)
    V = VectorFunctionSpace(mesh, family, 2, dim=3)
    f = Function(V)
    y_exact = Function(V)
    g1 = Function(V)
    g2 = Function(V)
    x = SpatialCoordinate(mesh)

    y_exact = as_vector((asinh(x[0]), x[1], cosh(asinh(x[0]))-1))
    f.interpolate(as_vector((-(6*x[0]**3-9*x[0])/(x[0]**2+1)**(7/2), 0, (12*x[0]**2-3)/(x[0]**2+1)**(7/2))))
    g1.interpolate(as_vector((0,x[1],0)))
    g2.interpolate(as_vector((asinh(1),x[1],sqrt(2)-1)))
    #g1 = y_exact
    #g2 = y_exact
    # build tensor space used to define phi (grad boundaries) 
    U = TensorFunctionSpace(mesh, family, 1, shape = (3,3))
    phi = Function(U)
    phi.interpolate(as_tensor([[1/sqrt(x[0]**2+1), 0, 0], [0, 1,0], [x[0]/sqrt(x[0]**2+1), 0, 0]]))
    #phi = grad(y_exact)
    n = FacetNormal(mesh)
    
    # create the cross space to define y, p (solution)
    Q = TensorFunctionSpace(mesh, "DG", 0, shape=(2,2), symmetry = True)
    W = V*Q
    w = Function(W)
    y, p = w.split()

    y, p = split(w)

    # build the LHS of the equation
    # commented out code in the section below corresponds to when phi neq 0
    gamma1 = Constant(gamma_1)
    gamma0 = Constant(gamma_0)
    h = avg(CellVolume(mesh)) / FacetArea(mesh)

    E = inner(grad(grad(y)), grad(grad(y)))/2*dx
    E -= inner(f, y)*dx

    E -= inner(avg(dot(grad(grad(y)), n)), jump(grad(y)))*dS
    E += inner(avg(dot(grad(div(grad(y))), n)), jump(y))*dS

    E -= inner(dot(grad(grad(y)), n), grad(y) - phi)*ds((1,2))

    E += gamma1*inner(jump(grad(y)),jump(grad(y)))/(h*2)*dS
    E += gamma0*inner(jump(y),jump(y))*h**(-3)/2*dS

    E += gamma1*inner((grad(y)- phi),(grad(y) - phi))/(h*2)*ds((1,2))

    E += gamma0*inner((y-g1),(y-g1))*h**(-3)/2*ds(1)
    E += gamma0*inner((y-g2),(y-g2))*h**(-3)/2*ds(2)
    E += inner(dot(grad(div(grad(y))), n), y-g1)*ds(1)
    E += inner(dot(grad(div(grad(y))), n), y-g2)*ds(2)

    K = dot(grad(y).T, grad(y))
    Khat = as_tensor([[K[0,0], K[0,1]], [K[1,0], K[1,1]]])

    # Isometry condition and equation
    S = E + inner(p, Khat - Identity(2))*dx

    Equation = derivative(S, w)
    parameters = {"ksp_type": "preonly", "pc_type": "lu", "mat_type": "aij", "pc_factor_mat_solver_type": "mumps",
                  "snes_monitor": None}
    parameters2 = {"ksp_type":"gmres", "pc_type": "ilu", "ksp_monitor": None, "snes_monitor": None}
    #solve(Equation == 0, w, solver_parameters = parameters)
    problem = NonlinearVariationalProblem(Equation, w)
    solver =  NonlinearVariationalSolver(problem, solver_parameters = parameters)

    ees = np.arange(0.05, 0.05, 0.01)

    #mesh coordinate warping
    #X_c = mesh.coordinates
    #X_ref = X_c.copy()
    y_error = Function(V, name = "y_error")
    y, p = w.split()
    #X_c.assign(y)
    file_0 = File("isometry2.pvd")
    #X_c.assign(X_ref)
    #y.interpolate(y_exact)
    y.interpolate(as_vector((asinh(x[0]), x[1], 0.9*(cosh(asinh(x[0]))-1))))    
    solver.solve()
    y_error.interpolate(y-y_exact)
    file_0.write(y, p, y_error)
    return h, norm(y_error)

colour_list = (("blue","skyblue"),("green","yellow"),("orange","red"))
plt.figure()
plt.title("Loglog Plot of DG and CG errors")
errors_DG =[]
errors_CG = []
r = np.array([10.0,20,30,50,60,80,100])
for j in r:
    print(j)
    mesh0 = RectangleMesh(int(j),int(j),1,1)
    c_DG = convergence(mesh0=mesh0, family = "DG")
    errors_DG.append(c_DG[1])
    c_CG = convergence(mesh0=mesh0, family = "CG")
    errors_CG.append(c_CG[1])
print(errors_DG)
print(errors_CG)

slope_DG, intercept_DG = np.polyfit(np.log(1.0/r),np.log(errors_DG), 1)
slope_CG, intercept_CG = np.polyfit(np.log(1.0/r),np.log(errors_CG), 1)
print(slope_DG, intercept_DG)
print(slope_CG, intercept_CG)

plt.xlabel("mesh fineness")
plt.ylabel("log errors")
plt.loglog(1.0/r, np.array(errors_DG), ".", color="red")
plt.loglog(1.0/r, np.array(errors_CG), ".", color="blue")
plt.loglog(1.0/r, 1.0/1000/r, color="black")
plt.legend(["DG","CG",r"line $\log(y)=-3 +\log(x)$"])
plt.savefig("convergence_graph_1")
#plt.show()




