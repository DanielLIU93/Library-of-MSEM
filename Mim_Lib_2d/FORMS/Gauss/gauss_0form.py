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
from assemble import assemble
from basic_form import BasicForm

# %% THE CLASS BODY
class Gauss0form(BasicForm):
    """
    A class: Gauss 0 form

    #SUMMARY:
    """
    def __init__(self, mesh, p, is_inner=True, numbering_scheme=None, name=None, info=None):
        super().__init__(mesh, p, numbering_scheme, name, info)
        self._k = 0
        self._is_form = True
        self._is_inner  = is_inner
        self._what_form = '0-gauss'
        
        self._nodal_grid = 'gauss'
        self._nodal_nodes = [getattr(functionals, self._nodal_grid + '_quad')(p)[0] for p in self.p]
        
        self._quad_type, self._quad_order = ('gauss','gauss'), (self.p[0], self.p[1])
        self._quad_nodes[0], self._quad_weights[0] = getattr(functionals, self._quad_type[0] + '_quad')(self._quad_order[0])
        self._quad_nodes[1], self._quad_weights[1] = getattr(functionals, self._quad_type[1] + '_quad')(self._quad_order[1])

        self._num_basis = self.p[0] * self.p[1]
        
        self._basis_nodes = None
        self._reconstructed = None

        self._func = None
        self._form = "p"
        #<-------------------------------------------------------------------->
        #standard init done, below are special variabels for this extended form
        self._E10 = None
        self.evaluate_basis()
        
    # %% FUNC related

    # %% PROPERTIES
        
    # %% INCIDENCE MATRIX MAKER

    # %% BASIS_Nodes
    @property
    def basis_nodes(self):
        """2D coordinates of degrees of freedom"""
        if self._basis_nodes is None:
            xi, eta = np.meshgrid(self._nodal_nodes[0], self._nodal_nodes[1])
            self._basis_nodes = (xi.ravel('F'), eta.ravel('F'))
        return self._basis_nodes

    #%% EVALUATE BASIS
    def evaluate_basis(self, domain=None):
        """
        #SUMMARY: Update self.xi, eta and basis.
        """
        if domain is None:
            # evaluate the lagrange basis at the default quad nodes
            self._xi, self._eta = np.meshgrid(self._quad_nodes[0], self._quad_nodes[1])
            nodal_basis_1d = [functionals.lagrange_basis(
                self._nodal_nodes[i], self._quad_nodes[i]) for i in range(2)]
            self._evaluate_basis_domain = (self._quad_nodes[0], self._quad_nodes[1])
            
        else:
            self._xi, self._eta = np.meshgrid(*domain)
            nodal_basis_1d = [functionals.lagrange_basis(
                self._nodal_nodes[i], domain[i]          ) for i in range(2)]
            self._evaluate_basis_domain = domain
            
        self._basis = np.kron(nodal_basis_1d[0], nodal_basis_1d[1])

    # %% INNER PRODUCT
    def inner(self, other):
        """
        #SUMMARY: Compute inner product.
        #OUTPUTS: [0] The Inner Matrix
        #             self .basis -> 0 axis
        #             other.basis -> 1 axis
        """
        assert self.mesh is other.mesh, "Mesh of the forms do not match"
        _, _, quad_weights_2d = self._do_same_quad_as(other)

        # det of the jacobian
        g = self.mesh.g(self.xi.ravel('F'), self.eta.ravel('F'))

        # inner product
        inner_matrix = np.dot(self.basis, other.basis[:, :, np.newaxis]
                     * (quad_weights_2d * g)[np.newaxis, :, :])

        return inner_matrix

    # %% HODGE MATRIX
    def H(self, _2form, update_self_cochain = True):
        """
        #SUMMARY: Compute the Hodge Matrix H: self = H * _2form
        #OUTPUTS: [0] H
                  [1] Assembled H
        """
        assert self.is_inner is not _2form.is_inner, " Hodge needs to connect two differently oriented forms"
        if _2form.__class__.__name__ in ('Lobatto2form',):
            W = _2form.wedged(self)
            H = np.tensordot(self.invM, W, axes=((2), (0)))
            H = np.rollaxis(H, 0, 3)
            H_a = assemble(H, self.dof_map, _2form.dof_map)
            if _2form.cochain is not None and update_self_cochain is True:
                self.cochain = H_a.dot(_2form.cochain)
                if _2form.func is not None:
                    self.func = _2form.func

            return H, H_a

    # %% WEDGE PRODUCT
    def wedged(self, other):
        """
        #SUMMARY: Integrate the wedge product of two basis
        #OUTPUTS: [0] The Wedge Matrix: W
        #             other.basis -> 0 axis
        #             self .basis -> 1 axis
        """
        assert self.mesh is other.mesh, "Mesh of the forms do not match"
        assert self.k + other.k == self.mesh.dim, 'k-form wedge l-form, k+l should be equal to n'
        _, quad_weights, _ = self._do_same_quad_as(other)

        quad_weights_2d = np.kron(quad_weights[0], quad_weights[1]
                                  ).reshape(1,np.size(quad_weights[0]) * np.size(quad_weights[1]))

        W = np.tensordot(other.basis, self.basis * quad_weights_2d, axes=((1), (1)))

        return W

    # %% DISCRETIZE RELATED
    def discretize(self, func):
        """
        #SUMMARY: Project a function onto a finite dimensional space of 0-forms.
        """
        self.func = func
        self.cochain_local = func(*self.mesh.mapping(*self.basis_nodes))
        print(' <FORM> <{}> : Red; Er={:.10f}'.format(self.__class__.__name__, self.L2_error()[0]))

    # %% RECONSTRUCT RELATED
    def reconstruct(self, xi=None, eta=None):
        """
        #SUMMARY: Reconstruct the 0-form on the physical domain.
        # UPDATE: xi; eta; basis; reconstructed;
        """

        assert self.cochain is not None, "no cochain to reconstruct"
        xi, eta = self._check_xi_eta(xi, eta)

        self.evaluate_basis(domain=(xi, eta))
        self._x, self._y = self.mesh.mapping(self.xi, self.eta)

        self._reconstructed = np.tensordot(self.basis, self.cochain_local, axes=([0], [0]))

    # %% RECONSTRUCTED
    @property
    def reconstructed(self):
        if self._reconstructed is not None:
            return self._reconstructed
        else:
            raise Exception("reconstructed is None, first reconstruct")

    # %% PLOT SELF.MESH
    def plot_mesh(self, regions=None, elements=None, plot_density=10):
        self.mesh.plot_mesh(regions=regions, elements=elements, plot_density=plot_density,
                            internal_mesh_type=('gauss', self.p))

# %% Main test part
if __name__ == '__main__':
    from mesh_crazy import CrazyMesh
    p = (16, 15)
    n = (2, 3)
    c = 0.15
    domain = ((-1, 1), (-1, 1))
    mesh = CrazyMesh(elements_layout=n, curvature=c, bounds_domain=domain)

    def p0func(x, y): return np.sin(np.pi*x) * np.sin(np.pi*y)

    f0 = Gauss0form(mesh, p, is_inner = False)
        
    f0.discretize(p0func)
    f0.plot_self()

    print('L2_error=',f0.L2_error()[0])

    