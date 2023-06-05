from firedrake import *
import numpy as np

initial_mesh = RectangleMesh(10,10,1,1)


def g1(x, ee):
    return [0, x[1], 1]

def g2(x, ee):
    return [2 - 10*ee, x[1], 1]

def g3(x, ee):
    return [x[0], 0, 0]

def g4(x, ee):
    return [x[0], 1, 0]

def phi1(x, ee):
    return [[1,0,0], [0,1,0], [0,0,0]]

def phi2(x, ee):
    return [[1,0,0], [0,1,0], [0,0,0]]

def phi3(x, ee):
    return [[1,0,0], [0,1,0], [0,0,0]]

def phi4(x,ee):
    return [[1,0,0], [0,1,0], [0,0,0]] 

def f(x):
    return [0, 0, 0]

def y_init(x):
    return [x[0], x[1], cos(pi*x[0])]

class VectorBiharmonicSolver():
    def __init__(self, family="DG", B1=True, B2=True, B3=False, B4=False, mesh0 = initial_mesh, gamma0=100, gamma1=100):
        """
        class that implements the solution to the vector biharmonic equation
        params:
        ------------
        family = string, describes the type of galerkin method used
        m = tuple, list of numbers that describe the rectangular mesh
        B1,B2,B3,B4 bbolean values corresponding to if that boundary is Dirichlet or not
        gamma1, gamma2 = integers that are specific to the DG method that govern the extent of
        discontinuities across elements.
        -----------
        """
        self.B1, self.B2, self.B3, self.B4 = B1, B2, B3, B4
        self.gamma0, self.gamma1 = Constant(gamma0), Constant(gamma1)
        
        # create a 2d mesh embedded in 3d
        Vx = VectorFunctionSpace(mesh0, family, 2, dim=3)
        x0, y0 = SpatialCoordinate(mesh0)
        X = Function(Vx).interpolate(as_vector([x0, y0, 0]))
        self.mesh = Mesh(X)

        # create the vector function space to define p, f, g1 and  g2 (scalar boundaries)
        self.V = VectorFunctionSpace(self.mesh, family, 2, dim=3)
        V = self.V
        self.f = Function(V)
        self.g1 = Function(V)
        self.g2 = Function(V)
        self.g3 = Function(V)
        self.g4 = Function(V)
        self.x = SpatialCoordinate(self.mesh)
        
        # build tensor space used to define phi (grad boundaries) 
        U = TensorFunctionSpace(self.mesh, "DG", 1, shape = (3,3))
        self.phi1 = Function(U)
        self.phi2 = Function(U)
        self.phi3 = Function(U)
        self.phi4 = Function(U)
        n = FacetNormal(self.mesh)

        # create the cross space to define y, p (solution)
        Q = TensorFunctionSpace(self.mesh, "DG", 0, shape=(2,2), symmetry = True)
        W = V*Q
        self.w = Function(W)
        self.y, self.p = self.w.split()
        self.y, self.p = split(self.w)
                            
        h =  avg(CellVolume(self.mesh)) / FacetArea(self.mesh)

        E = inner(grad(grad(self.y)), grad(grad(self.y)))/2*dx
        E -= inner(self.f, self.y)*dx

        E -= inner(avg(dot(grad(grad(self.y)), n)), jump(grad(self.y)))*dS
        E += inner(avg(dot(grad(div(grad(self.y))), n)), jump(self.y))*dS

        E += self.gamma1*inner(jump(grad(self.y)),jump(grad(self.y)))/(h*2)*dS
        E += self.gamma0*inner(jump(self.y),jump(self.y))*h**(-3)/2*dS

        # adds required boundary terms to the formulation
        for b, v in zip([self.B1, self.B2, self.B3, self.B4],
                        [[1, self.g1, self.phi1], [2, self.g2, self.phi2],
                         [3, self.g3, self.phi3], [4, self.g4, self.phi4]]):
            if b:
                E += self.gamma0*inner((self.y - v[1]),(self.y - v[1]))*h**(-3)/2*ds(v[0])
                E += inner(dot(grad(div(grad(self.y))), n), self.y - v[1])*ds(v[0])
                E -= inner(dot(grad(grad(self.y)), n), grad(self.y) - v[2])*ds(v[0])
                E += self.gamma1*inner((grad(self.y) - v[2]),(grad(self.y) - v[2])/(h*2))*ds(v[0])
        
        K = dot(grad(self.y).T, grad(self.y))
        Khat = as_tensor([[K[0,0], K[0,1]], [K[1,0], K[1,1]]])

        # Isometry condition and equation
        S = E + inner(self.p, Khat - Identity(2))*dx

        Equation = derivative(S, self.w)

        parameters = {"ksp_type": "preonly", "pc_type": "lu", "mat_type": "aij", "pc_factor_mat_solver_type": "mumps",
                      "snes_monitor": None}
        problem = NonlinearVariationalProblem(Equation, self.w)
        self.solver = NonlinearVariationalSolver(problem, solver_parameters = parameters)

    def continuation_solution(self, g1, g2, g3, g4, f, y_init,
                              phi1, phi2, phi3, phi4, index_range):
        """
        -----------
        g1: function,  function that is applied to the ds(1) boundary similar for g2,g3,g4
        phi1: the value of the gradient at the boundary ds(1) similar for phi2,phi3,phi4
        f: function representing LHS of the PDE
        y_init: the initial solution guess
        index range: the range on which the continuation method is applied
        -----------
        """
        # mesh coordinate warping
        X_c = self.mesh.coordinates
        X_ref = Function(self.V).assign(X_c)

        self.y, self.p = self.w.split()
        file_0 = File("isometry.pvd")
        
        self.f.interpolate(as_vector(f(self.x)))
        self.y.interpolate(as_vector(y_init(self.x)))
        file_0.write(self.y, self.p)
        
        for ee in index_range:
            print(str(ee), "ee")
            # update the boundaries for each value of epsilon
            # if one boundary has a dirichlet condition (B_i=True)
            #then interpolate into self.gi and self.phi_i the functions g_i and phi_i
            for b, v in zip((self.B1, self.B2, self.B3, self.B4),
                          [[self.g1, g1, self.phi1, phi1],
                           [self.g2, g2, self.phi2, phi2],
                           [self.g3, g3, self.phi3, phi3],
                           [self.g4, g4, self.phi4, phi4]]):
                if b:
                    v[0].interpolate(as_vector(v[1](self.x, ee)))
                    v[2].interpolate(as_vector(v[3](self.x, ee)))

            self.solver.solve()
            X_c.assign(self.y)
            print(assemble(self.y[2]**2*dx))
            file_0.write(self.y, self.p)
            X_c.assign(X_ref)
    
    def convergence(self, exact):
        raise NotImplementedError

VectorBiharmonicSolver().continuation_solution(g1, g2, g3, g4, f, y_init, phi1, phi2, phi3, phi4, np.arange(0.02, 1,0.02))

