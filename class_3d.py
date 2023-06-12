from firedrake import *
import numpy as np


class VectorBiharmonicSolver():
    def __init__(self, f, g, phi, mesh0, family="CG", gamma0=100000, gamma1=300000, normals = "+"):
        """
        class that implements the solution to the vector biharmonic equation
        params:
        ------------
        g: list of function boundary conditions expressed in UFL
        phi: list of gradient boundary conditions expressed in UFL.
        (note if a boundary if g_i or phi_i is not defined the entry in the above lists should be None)
        family = string, describes the type of galerkin method used
        mesh0 = Fierdrake mesh object
        gamma0, gamma1 = integers that are specific to the DG and CG methods that govern the extent of
        discontinuities across elements.
        -----------
        """
        self.f_bar = f
        self.g = g
        self.phi = phi 
        self.gamma0, self.gamma1 = Constant(gamma0), Constant(gamma1)

        # create a 2d mesh embedded in 3d
        Vx = VectorFunctionSpace(mesh0, family, 2, dim=3)
        x0, y0 = SpatialCoordinate(mesh0)
        X = Function(Vx).interpolate(as_vector([x0, y0, 0]))
        self.mesh = Mesh(X)
        self.x = SpatialCoordinate(self.mesh)
        n = FacetNormal(self.mesh)
        h = avg(CellVolume(self.mesh)) / FacetArea(self.mesh)
        
        # create the vector function space to define f, g1, g2, g3, g4
        self.V = VectorFunctionSpace(self.mesh, family, 2, dim=3)
        V = self.V
        for val in ["f","g1","g2","g3","g4"]:
                exec(f"self.{val} = Function(V)")
        
        
        # build tensor space used to define phis (grad boundaries) 
        U = TensorFunctionSpace(self.mesh, "DG", 1, shape = (3,3))
        for val in ["phi1", "phi2", "phi3", "phi4"]:
            exec(f"self.{val} = Function(U)")

        # create the cross space to define y, p (solution)
        Q = TensorFunctionSpace(self.mesh, "DG", 0, shape=(2,2), symmetry = True)
        W = V*Q
        self.w = Function(W)
        self.y, self.p = self.w.split()
        self.y, self.p = split(self.w)

        # begin to assemble the energy expression
        self.E = inner(grad(grad(self.y)), grad(grad(self.y)))/2*dx
        self.E -= inner(self.f, self.y)*dx

        if normals == "+":
            self.E -= inner(dot(avg(grad(grad(self.y))), n("+")), jump(grad(self.y)))*dS
            self.E += inner(dot(avg(grad(div(grad(self.y)))), n("+")), jump(self.y))*dS
        else:
            self.E -= inner(avg(dot(grad(grad(self.y)), n)), jump(grad(self.y)))*dS
            self.E += inner(avg(dot(grad(div(grad(self.y))), n)), jump(self.y))*dS

        self.E += self.gamma1*inner(jump(grad(self.y)),jump(grad(self.y)))/(h*2)*dS
        self.E += self.gamma0*inner(jump(self.y),jump(self.y))*h**(-3)/2*dS
           
        # adds required boundary terms to the formulation
        for i, b in enumerate(self.g):
            if b:
                exec(f"self.E += self.gamma0*inner((self.y - self.g{i+1}),(self.y - self.g{i+1}))*h**(-3)/2*ds({i+1})")
                exec(f"self.E += inner(dot(grad(div(grad(self.y))), n), self.y - self.g{i+1})*ds({i+1})")
                #exec(f"self.E += inner(dot(grad(self.y), n), self.y - self.g{i+1})*ds({i+1})")

        for i, b in enumerate(self.phi):
            if b:
                exec(f"self.E -= inner(dot(grad(grad(self.y)), n), grad(self.y) - self.phi{i+1})*ds(i+1)")
                exec(f"self.E += self.gamma1*inner((grad(self.y) - self.phi{i+1}),(grad(self.y) - self.phi{i+1})/(h*2))*ds(i+1)")

        K = dot(grad(self.y).T, grad(self.y))
        Khat = as_tensor([[K[0,0], K[0,1]], [K[1,0], K[1,1]]])

        # Isometry condition and equation
        S = self.E + inner(self.p, Khat - Identity(2))*dx

        Equation = derivative(S, self.w)

        parameters = {"ksp_type": "preonly", "pc_type": "lu", "mat_type": "aij", "pc_factor_mat_solver_type": "mumps",
                      "snes_monitor": None}
        problem = NonlinearVariationalProblem(Equation, self.w)
        self.solver = NonlinearVariationalSolver(problem, solver_parameters = parameters)

    def continuation_solution(self, y_init, index_range=np.arange(0.02,0.4,0.05), file_name ="isometry.pvd"):

        # mesh coordinate warping
        X_c = self.mesh.coordinates
        X_ref = Function(self.V).assign(X_c)
        self.y, self.p = self.w.split()
        self.y.interpolate(as_vector(y_init(self.x)))
        if file_name:
            file_0 = File(file_name)
            file_0.write(self.y, self.p)
        for ee in index_range:
            print(str(ee), "ee")
            self.f.interpolate(as_vector(self.f_bar(self.x, ee)))
            for i,b in enumerate(self.g):
                if b:
                    exec(f"self.g{i+1}.interpolate(as_vector(b(self.x, ee)))")
                
            for i,b in enumerate(self.phi):
                if b:
                    exec(f"self.phi{i+1}.interpolate(as_vector(b(self.x, ee)))")

            self.solver.solve()
            X_c.assign(self.y)
            print(assemble(self.y[2]**2*dx))
            if file_name:
                file_0.write(self.y, self.p)
            X_c.assign(X_ref)

    def convergence(self, y_exact, file_name=None, index_range=[1], y_init=None):
        """
        measures the error between exact and simulated solution
        params:
        ---------
        exact: a UFL function representing the analytic solution
        ---------
        """
        y_error = Function(self.V)
        if not y_init:
            y_init = y_exact 
        self.continuation_solution(y_init = y_init, index_range=index_range, file_name=file_name)
        y_error.interpolate(self.y - as_vector(y_exact(self.x)))
        return norm(y_error)
