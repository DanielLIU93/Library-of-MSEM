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
import matplotlib.pyplot as plt


# %% CLASS BODY
class Lobatto2form(BasicForm):
    """Lobatto 2form"""
    def __init__(self, mesh, p, is_inner=True, separated_dof=None, numbering_scheme=None, name=None, info=None):
        # separated_dof is useless here, just for convinence in forms.py
        super().__init__(mesh, p, numbering_scheme, name, info)

        self._k = 2
        self._is_form = True
        self._is_inner = is_inner
        self._separated_dof = None
        self._what_form = '2-lobatto'

        self._face_grid = 'lobatto', 'lobatto'
        self._face_nodes = [getattr(functionals, self._face_grid[i] + '_quad')(self.p[i])[0] for i in range(2)]

        self._num_basis = self.p[0] * self.p[1]

        self._quad_type, self._quad_order = ('gauss', 'gauss'), (self.p[0], self.p[1])
        self._quad_nodes[0], self._quad_weights[0] = getattr(functionals, self._quad_type[0] + '_quad')(self._quad_order[0])
        self._quad_nodes[1], self._quad_weights[1] = getattr(functionals, self._quad_type[1] + '_quad')(self._quad_order[1])

        self.evaluate_basis()

        self._func = None
        self._form = "fdxdy"

    # %% FUNC related

    # %% PROPERTIES
    @property
    def basis(self):
        return self._basis

    # EVALUATE THE BASIS
    def evaluate_basis(self, domain=None):
        """Evaluate the basis.

        The basis are evaluated in at the position of the dof or quad nodes (if supplied) or
        at the domain specified.
        """
        if domain is None:
            # evaluate the lagrange basis in one 1d for both x and y at quad points
            self._xi, self._eta = np.meshgrid(self._quad_nodes[0], self._quad_nodes[1])
            edge_basis_1d = [functionals.edge_basis(self._face_nodes[i], self._quad_nodes[i]) for i in range(2)]
            self._evaluate_basis_domain = (self._quad_nodes[0], self._quad_nodes[1])
            
        else:
            self._xi, self._eta = np.meshgrid(*domain)
            edge_basis_1d = [functionals.edge_basis(self._face_nodes[i], domain[i]) for i in range(2)]
            self._evaluate_basis_domain = domain
            
        self._basis = np.kron(edge_basis_1d[0], edge_basis_1d[1])
        return self._basis

    # THE INNER PRODUCT
    def inner(self, other):
        """
        #SUMMARY: Compute inner product.
        #OUTPUTS: [0] The Inner Matrix
        #             self .basis -> 0 axis
        #             other.basis -> 1 axis
        """
        assert self.mesh is other.mesh, "Mesh of the forms do not match"
        _, _, quad_weights_2d = self._do_same_quad_as(other)

        g = self.mesh.g(self.xi.ravel('F'), self.eta.ravel('F'))

        M_2 = np.tensordot(
            self.basis, other.basis[:, :, np.newaxis] *
            (np.reciprocal(g) * quad_weights_2d), axes=((1), (1)))
        return M_2

    # %% THE HODGE MATRIX
    def H(self, _0form, update_self_cochain = True):
        """
        #SUMMARY: Compute the Hodge Matrix H: self = H * _2form
        #OUTPUTS: [0] Assembled H
        """
        assert self.is_inner is not _0form.is_inner, " Hodge needs to connect two differently oriented forms"
        if _0form.__class__.__name__ == 'Gauss0form':
            W = _0form.wedged(self)
            H = np.tensordot(self.invM, W, axes=((2), (0)))
            H = np.rollaxis(H, 0, 3)
            H_a = assemble(H, self.dof_map, _0form.dof_map)
            if _0form.cochain is not None and update_self_cochain is True:
                self.cochain = H_a.dot(_0form.cochain)
                if _0form.func is not None:
                    self.func = _0form.func
            return H, H_a

        if _0form.__class__.__name__ == 'ExtGauss0form':
            W = _0form.wedged(self)
            H = np.tensordot(self.invM, W, axes=((2), (0)))
            H = np.rollaxis(H, 0, 3)
            H_a = assemble(H, self.dof_map, _0form.dof_map_internal)
            if _0form.cochain_internal is not None and update_self_cochain is True:
                self.cochain = H_a.dot(_0form.cochain_internal)
                if _0form.func is not None:
                    self.func = _0form.func
            return H, H_a

    # %% THE WEDGE MATRIX
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
                                  ).reshape(1, np.size(quad_weights[0]) * np.size(quad_weights[1]))

        W = np.tensordot(other.basis, self.basis * quad_weights_2d, axes=(1, 1))

        # plt.matshow(self.basis)
        # plt.set_cmap('viridis')
        # plt.colorbar()
        # plt.matshow(quad_weights_2d)
        # plt.colorbar()
        # plt.show()

        return W

    # %% DISCRETIZATION OR REDUCTION
    def discretize(self, func, quad=None ) :
        """
        #SUMMARY: Project a function onto a finite dimensional space of 2-forms.
        """
        self.func = func
        if quad is None:
            the_p = np.max([self.p[0]+5,self.p[1]+5])
            quad = (('gauss', 'gauss'), (the_p,the_p))
        quad_nodes, quad_weights, _ = self._evaluate_quad_grid( quad )
        _, (p, _p) = quad
        assert _[0] == _[1], 'this methid only support the same quad_type on x,y directions now'
        assert p == _p, 'this method can only support the same quad_order on x,y directions now'
        if _ == ('gauss','gauss'):
            p = p-1

        xi_ref, eta_ref = np.meshgrid(quad_nodes[0], quad_nodes[1])
        quad_weights_2d = np.kron(quad_weights[0], quad_weights[1])

        # calculate the dimension of the edges of the cells
        dim_faces = [self._face_nodes[i][1:] - self._face_nodes[i][:-1]
                     for i in range(2)]
        # set up the right amout of x and y dimensions of the edges of the cell
        x_dim = np.repeat(dim_faces[0], self.p[1])
        y_dim = np.tile(dim_faces[1], self.p[0])
        magic_factor = 0.25
        cell_nodes = [(0.5 * (dim_faces[i][np.newaxis, :]) *
                       (quad_nodes[i][:, np.newaxis] + 1) +
                       self._face_nodes[i][:-1]).ravel('F') for i in range(2)]

        cell_area = np.diag(magic_factor * x_dim * y_dim)
        # xi coordinates of the quadrature nodes
        # in the column are stored the coordinates of the quad points for contant xi for all faces
        xi = np.repeat(np.repeat(cell_nodes[0], (p + 1) ).reshape((p + 1)**2, self.p[0], order='F'), self.p[1], axis=1)

        eta = np.tile(np.tile(cell_nodes[1].reshape( p + 1, self.p[1], order='F'), (p + 1, 1)), self.p[0])

        # map onto the physical domain and compute the Jacobian
        x, y = self.mesh.mapping(xi, eta)
        g = self.mesh.g(xi, eta)

        # compute the cochain integrating and then applying inverse pullback
        self.cochain_local = np.sum(np.tensordot(quad_weights_2d, (self.func(x, y) * g),
                                axes=((0), (0))) * cell_area[:, :, np.newaxis], axis=0)
        
        print(' <FORM> <{}> : Red; Er={:.10f}'.format(self.__class__.__name__, self.L2_error()[0]))

    # %% RECONSTRUCTION
    def reconstruct(self, xi=None, eta=None):
        """
        #SUMMARY: Reconstruct the 0-form on the physical domain.
        # UPDATE: xi; eta; basis; reconstructed;
        """
        assert self.cochain is not None, "no cochain to reconstruct"
        xi, eta = self._check_xi_eta(xi, eta)

        self.evaluate_basis(domain=(xi, eta))
        self._x, self._y = self.mesh.mapping(self.xi, self.eta)

        xi, eta = self.xi.ravel('F'), self.eta.ravel('F')
        self._reconstructed = np.tensordot(
                self.basis, self.cochain_local, axes=(0, 0)) / self.mesh.g(xi, eta)

    # %% reconsturcted
    @property
    def reconstructed(self):
        if self._reconstructed is not None:
            return self._reconstructed
        else:
            raise Exception("reconstructed is None, first reconstruct")

    # %% PLOT SELF.MESH
    def plot_mesh(self, regions=None, elements=None, plot_density=10):
        self.mesh.plot_mesh(regions=regions, elements=elements, plot_density=plot_density,
                            internal_mesh_type=('lobatto', self.p))

# %% THE MAIN TEST PART
if __name__ == '__main__':
    from mesh_crazy import CrazyMesh
    p = (16, 15)
    n = (2, 3)
    c = 0.15
    domain = ((-1, 1), (-1, 1))
    mesh = CrazyMesh(elements_layout=n, curvature=c, bounds_domain=domain)

    f2 = Lobatto2form(mesh, p, is_inner=True)
    M = f2.M[0]
    f2.invM

    def f2func(x, y): return -8 * np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)

    f2.discretize(f2func)
    f2.plot_self()
