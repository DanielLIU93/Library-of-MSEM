# -*- coding: utf-8 -*-
"""
A form

@author: Yi Zhang [2017]
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import numpy as np
import functionals
import ultint
from numbering import numbering
from basic_form import BasicForm


# %% THE CLASS BODY
class TraceGauss0form(BasicForm):
    """
    SUMMARY: This is a trace 0-form which leaves in the space of dimension n-1
             We expand the 0-form with gauss polynomials, so we finally call it
             "TraceGauss0form"
    """
    def __init__(self, mesh, p, is_inner=True, numbering_scheme=None, name=None, info=None):
        super().__init__(mesh, p, numbering_scheme, name, info)

        self._k = 0
        self._is_trace_form = True
        self._is_inner = is_inner

        self._what_form = '0-gauss_tr'

        self._nodal_grid = 'gauss'
        self._nodal_nodes = [getattr(functionals, self._nodal_grid+'_quad')(p)[0] for p in self.p]

        self._num_basis_south = self.p[0]
        self._num_basis_north = self.p[0]
        self._num_basis_west = self.p[1]
        self._num_basis_east = self.p[1]
        self._num_basis = 2 * (self.p[0] + self.p[1])

        self._numbering_scheme = numbering_scheme

        self._quad_type, self._quad_order = ('gauss', 'gauss'), (self.p[0], self.p[1])
        self._quad_nodes[0], self._quad_weights[0] = getattr(functionals, self._quad_type[0]+'_quad')(self._quad_order[0])
        self._quad_nodes[1], self._quad_weights[1] = getattr(functionals, self._quad_type[1]+'_quad')(self._quad_order[1])

        self.evaluate_basis()
        self._basis_trace = (('gauss_node', self.p[0]), ('gauss_node', self.p[1]))

        self._cochain_boundary = None
        self._dof_map_boundary = None

        self._basis_nodes = None

        self._func = None
        self._form = "p"

    # %% FUNC related

    # %% PROPERTIES
    @property
    def num_basis_south(self):
        return self._num_basis_south

    @property
    def num_basis_north(self):
        return self._num_basis_north

    @property
    def num_basis_west(self):
        return self._num_basis_west

    @property
    def num_basis_east(self):
        return self._num_basis_east

    @property
    def basis_trace(self):
        return self._basis_trace

    # %% DOF MAP RELATED
    @property
    def dof_map(self):
        if self._dof_map is not None:
            return self._dof_map

        self._dof_map, self._dof_map_boundary = numbering(self, numbering_scheme=self.numbering_scheme)

        return self._dof_map

    # %% DOF MAP BOUNDARY MAKER
    @property
    def dof_map_boundary(self):
        """
        #SUMMARY: This dof_map get the number of dof on boundary
        #OUTPUTS:
            CrazyMesh: a tuple of 4 entries which correspond to the (S, N, W, E) boundaries
        """
        if self._dof_map_boundary is not None:
            return self._dof_map_boundary

        self._dof_map, self._dof_map_boundary = numbering(self)

        return self._dof_map_boundary

    # %% DOF MAP LOCAL TRACE MAKER
    @property
    def dof_map_local_trace(self):
        """
        #SUMMARY: This dof_map maps the dof on boundary to the local numbering
                  For a trace form, it is every easy.
                  For Example:
                      for a trace gauss 0-form @ p = 3
                      dof_map_local_trace =
                      ([  0   1   2 ],
                       [  3   4   5 ],
                       [  6   7   8 ],
                       [  9  10  11 ])
                      The rows correspond to the S, N, W, E boundaries
        """
        px, py = self.p
        dof_map_local_trace_0 = np.array([int(i)for i in range(px)])
        dof_map_local_trace_1 = np.array([int(i+px)for i in range(px)])
        dof_map_local_trace_2 = np.array([int(i+2*px)for i in range(py)])
        dof_map_local_trace_3 = np.array([int(i+2*px+py) for i in range(py)])

        dof_map_local_trace = (dof_map_local_trace_0, dof_map_local_trace_1,
                               dof_map_local_trace_2, dof_map_local_trace_3)

        return dof_map_local_trace

#  COCHAIN BOUNDARY SHOWER
    @property
    def cochain_boundary(self):
        """
        #SUMMARY: Return the cochains on the boundaries
        #OUtPUTS: [0] tuple, number of entries depend on the domain, and how to divide the boundaries
                             For Example, Normally, Crazy_Mesh will have four boundaries
                             For more details, check self.dof_map_boundary?
        """
        BC = self.dof_map_boundary
        if self._cochain_boundary is None:
            self._cochain_boundary = ()
            for boundary_id, boundary_cells in enumerate(BC):
                self._cochain_boundary = self._cochain_boundary + (self.cochain[boundary_cells],)
        return self._cochain_boundary

    # %% BASIS NODES
    @property
    def basis_nodes(self):
        """2D coordinates of degrees of freedom"""
        if self._basis_nodes is None:
            xi = np.concatenate((self._nodal_nodes[0], self._nodal_nodes[0], -np.ones(self.p[1]), np.ones(self.p[1])))
            eta = np.concatenate((-np.ones(self.p[0]), np.ones(self.p[0]), self._nodal_nodes[1], self._nodal_nodes[1]))
            self._basis_nodes = (xi, eta)
        return self._basis_nodes

    # %% EVALUATE BASIS
    def evaluate_basis(self, domain=None):
        """
        #SUMMARY: Update self.xi, eta and basis.
        # UPDATE: xi, eta, basis
                  basis: A list, [0], [1] corresponding to S;N, W;E
        """
        if domain is None:
            # evaluate the lagrange basis at the default quad nodes
            self._xi, self._eta = self._quad_nodes[0], self._quad_nodes[1]
            self._basis = [functionals.lagrange_basis(
                self._nodal_nodes[i], self._quad_nodes[i]) for i in range(2)]
            self._evaluate_basis_domain = (self._quad_nodes[0], self._quad_nodes[1])
            
        else:
            self._xi, self._eta = domain[0], domain[1]
            self._basis = [functionals.lagrange_basis(
                self._nodal_nodes[i], domain[i]) for i in range(2)]
            self._evaluate_basis_domain = domain
            
        self._basis_SN = self._basis[0]
        self._basis_WE = self._basis[1]

    # %% BASIS
    @property
    def basis_SN(self):
        return self._basis_SN
    
    @property
    def basis_WE(self):
        return self._basis_WE
    
    # %% THE WEDGE PRODUCT
    def wedged(self, other):
        """
        #SUMMARY: Compute the "local" Trace Wedge Matrix
                  self.basis  corresponds to the 1st axis
                  other.basis corresponds to the 2nd axis
        #OUTPUTS: [0] the "local" Trace Wedge Matrix: local_W
        """
        assert self.mesh is other.mesh, "Mesh of the forms do not match"
        assert self.k + other.k == self.mesh.dim - 1, 'Must be a boundary integral'
        if self.quad_grid == other.quad_grid:
            quad = self.quad_grid
        else:
            p0 = np.max([self.p[0], other.p[0]])+1
            p1 = np.max([self.p[1], other.p[1]])+1
            quad = (('gauss', 'gauss'), (p0, p1))
        quad = ((quad[0][0], quad[1][0]), (quad[0][1], quad[1][1]))

        Wx = ultint.integral1d_(1, other.basis_trace[0], self.basis_trace[0], quad[0])
        Wy = ultint.integral1d_(1, other.basis_trace[1], self.basis_trace[1], quad[1])

        S = self.dof_map_local_trace
        O = other.dof_map_local_trace
        local_W = np.zeros(shape=(other.num_basis, self.num_basis))

        if other.__class__.__name__ == 'Lobatto1form':
            for i in range(other.p[0]):
                for j in range(self.p[0]):
                    local_W[O[0][i], S[0][j]] = + Wx[i, j]  # South
                    local_W[O[1][i], S[1][j]] = - Wx[i, j]  # North
            for i in range(other.p[1]):
                for j in range(self.p[1]):
                    local_W[O[2][i], S[2][j]] = - Wy[i, j]  # West
                    local_W[O[3][i], S[3][j]] = + Wy[i, j]  # East

        return local_W

    # %% DISCRETIZATION OR REDUCTION
    def discretize(self, func):
        """
        #SUMMARY: Project a function onto a finite dimensional space of 0-forms.
        """
        self.cochain_local = func(*self.mesh.mapping(*self.basis_nodes))
        self.func = func
    
    # %% L^2 error
    def L2_error(self):
        raise Exception(" <FORM> <TRAC> : Trace form have no L2_error")
        
    # %%
    def plot_mesh(self, regions=None, elements=None, plot_density=10):
        self.mesh.plot_mesh(regions=regions, elements=elements, plot_density=plot_density,
                            internal_mesh_type=('gauss', self.p))

# %% THE MAIN TEST PART 
if __name__ == '__main__':
    from mesh_crazy import CrazyMesh
    p = (16, 15)
    n = (2, 3)
    c = 0.15
    domain = ((-1, 1), (-1, 1))
    mesh = CrazyMesh(elements_layout=n, curvature=c, bounds_domain=domain)

    t0 = TraceGauss0form(mesh, p, is_inner = True)
    def p0func(x, y): return np.sin(np.pi * x) * np.sin(np.pi * y)
    t0.discretize(p0func)
