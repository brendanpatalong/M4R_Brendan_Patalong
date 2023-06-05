from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

def asinh(x):
    return ln(x + sqrt(1.0 + x**2))

def convergence(mesh0=RectangleMesh(10, 10, 1, 1)):
    # create a 2d mesh in 2d
    # create a 2d mesh embedded in 3d
    family = "DG"
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
    #f.interpolate(as_vector(((6*x[0]**3+9*x[0])/(x[0]**2+1)**(7/2), 0, (3-12*x[0]**2)/(x[0]**2+1)**(7/2))))
    #g1.interpolate(as_vector((asinh(x[0]), 0, cosh(asinh(x[0]))-1)))
    #g2.interpolate(as_vector((asinh(x[0]), 1, cosh(asinh(x[0]))-1)))
    
    f = div(grad(div(grad(y_exact))))
    g1 = y_exact
    g2 = y_exact
    # build tensor space used to define phi (grad boundaries) 
    U = TensorFunctionSpace(mesh, "DG", 1, shape = (3,3))
    phi = Function(U)
    #phi.interpolate(as_tensor([[1/sqrt(x[0]**2+1), 0, 0], [0, 1,0], [x[0]/sqrt(x[0]**2+1), 0, 0]]))
    phi = grad(y_exact)
    n = FacetNormal(mesh)
    
    # create the cross space to define y, p (solution)
    Q = TensorFunctionSpace(mesh, "DG", 0, shape=(2,2), symmetry = True)
    W = V*Q
    w = Function(W)
    y, p = w.split()

    y, p = split(w)

    # build the LHS of the equation
    # commented out code in the section below corresponds to when phi neq 0
    gamma1 = Constant(100)
    gamma0 = Constant(100)
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
    y.interpolate(y_exact)
    #y.interpolate(as_vector((asinh(x[0]), x[1], cosh(asinh(x[0]))-1)))    
    solver.solve()
    y_error.interpolate(y-y_exact)
    file_0.write(y, p, y_error)
    return h, norm(y_error)

errors = []
r = np.array([10.0,20,40, 80])
for i, j in enumerate(r):
    mesh0 = RectangleMesh(int(j),int(j),1,1)
    c = convergence(mesh0=mesh0)
    errors.append(c[1])
print(errors)


plt.loglog(1.0/r, np.array(errors), ".")
plt.loglog(1.0/r, 1.0/1000/r)
plt.savefig("convergence_graph")





