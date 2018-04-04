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
from numbering import numbering
from numpy.linalg import inv
from assemble import assemble
from lobatto_2form import Lobatto2form
from basic_form import BasicForm
import matplotlib.pyplot as plt
import scipy.io as sio


# %% CALSS BODY
class Lobatto1form(BasicForm):
    """Lobatto 1form"""

    def __init__(self, mesh, p, is_inner=True, separated_dof=True, numbering_scheme=None, name=None, info=None):
        super().__init__(mesh, p, numbering_scheme, name, info)

        self._k = 1
        self._is_form = True
        self._is_inner = is_inner
        self._numbering_scheme = numbering_scheme
        self._separated_dof = separated_dof
        self._what_form = '1-lobatto'

        self._nodal_grid = ('lobatto', 'lobatto')
        self._edge_grid = ('lobatto', 'lobatto')
        self._nodal_nodes = [getattr(functionals, self._nodal_grid[i] + '_quad')(self.p[1 - i])[0] for i in range(2)]
        self._edge_nodes = [getattr(functionals, self._edge_grid[i] + '_quad')(self.p[i])[0] for i in range(2)]

        self._num_basis_xi = self.p[0] * (self.p[1] + 1)
        self._num_basis_eta = self.p[1] * (self.p[0] + 1)
        self._num_basis = self.p[0] * (self.p[1] + 1) + self.p[1] * (self.p[0] + 1)
        self._basis_1edge = None

        if self.mesh.anti_corner_singularity is False:
            self._quad_type, self._quad_order = ('lobatto', 'lobatto'), (self.p[0], self.p[1])
        else:
            self._quad_type, self._quad_order = ('gauss', 'gauss'), (self.p[0] + 1, self.p[1] + 1)

        self._quad_nodes[0], self._quad_weights[0] = getattr(functionals, self._quad_type[0] + '_quad')(self._quad_order[0])

        self._quad_nodes[1], self._quad_weights[1] = getattr(functionals, self._quad_type[1] + '_quad')(self._quad_order[1])

        self._u = None
        self._v = None

        self._func_dx = None  # _dx
        self._func_dy = None  # _dy

        self._func_dx_in_form = None
        self._func_dy_in_form = None

        # <-------------------------------------------------------------------->
        # standard init done, below are special variabels for this extended form
        self.evaluate_basis()

        self._dof_map_boundary = None

        self._corresponding2 = 'TraceGauss0form'
        self._dof_map_local_trace = None

        self._E21 = None
        self._reconstructed_dx = None
        self._reconstructed_dy = None

    # %% func related
    @property
    def u(self):
        return self._u

    @property
    def v(self):
        return self._v

    @property
    def numbering_scheme(self):
        return self._numbering_scheme

    @property
    def func(self):
        return self._func_dx, self._func_dy

    @property
    def func_in_form(self):
        return self._func_dx_in_form, self._func_dy_in_form

    @func.setter
    def func(self, func):
        assert np.shape(func) == (2,), "we need 2 entries in func for 1form, shape(func)={}".format(np.shape(func))
        for i, func_i in enumerate(func):
            assert callable(func_i), "{}th function to be assigned to the form is not callable".format(i)

        if self.is_inner is True:
            self._func_dx, self._func_dy = func
            self._u, self._v = func
            self._func_dx_in_form = self._u
            self._func_dy_in_form = self._v

        elif self.is_inner is False:
            self._func_dy, self._func_dx = func
            self._u, self._v = func
            self._func_dx_in_form = self._v
            self._func_dy_in_form = lambda x, y: -self._u(x, y)

    # %%
    @property
    def form(self):
        if self.is_inner is True:
            return "u dx + v dy"
        elif self.is_inner is False:
            return "v dx - u dy"

    # %% PROPERTIES
    @property
    def num_basis_xi(self):
        return self._num_basis_xi

    @property
    def num_basis_eta(self):
        return self._num_basis_eta

    # %% DOF MAP RELATED
    @property
    def dof_map(self):
        """
        #SUMMARY:
        #OUTPUTS: [0] the dof_map, 2d-array: elements -> 1st axis
                                             local_numbering -> 2nd axis
        """
        if self._dof_map is not None:
            return self._dof_map

        self._dof_map, self._dof_map_boundary = numbering(self, numbering_scheme=self.numbering_scheme)
        return self._dof_map

    # %% DOF MAP INTERFACE PARIS
    @property
    def dof_map_interface_pairs(self):
        """
        #SUMMARY: get the interface dof pairs
        #OUTPUTS: [0] the pair, 2d-array, 1st axis -> the number of pairs
                                          2nd axis -> the two paired up dofs
        """
        assert self.separated_dof is True, "<FORM> <GL> : elements not separated, no interface pairs needed"

        if self.mesh.__class__.__name__ == "CrazyMesh":
            px, py = self.p
            nx, ny = self.mesh.n_x, self.mesh.n_y
            interface_edge_pair = np.zeros(((nx - 1) * ny * py + nx * (ny - 1) * px, 2), dtype=np.int64)
            n = 0
            for i in range(nx - 1):
                for j in range(ny):
                    s1 = j + i * ny
                    s2 = j + (i + 1) * ny
                    for m in range(py):
                        interface_edge_pair[n, 0] = self.dof_map[s1, px * (py + 1) + py * px + m]
                        interface_edge_pair[n, 1] = self.dof_map[s2, px * (py + 1) + m]
                        n += 1
            for i in range(nx):
                for j in range(ny - 1):
                    s1 = j + i * ny
                    s2 = j + 1 + i * ny
                    for m in range(px):
                        interface_edge_pair[n, 0] = self.dof_map[s1, (m + 1) * (py + 1) - 1]
                        interface_edge_pair[n, 1] = self.dof_map[s2, m * (py + 1)]
                        n += 1
            return interface_edge_pair
        else:
            raise Exception("<FORM> <GL> : not coded yet for this mesh class:" + self.mesh.__class__.__name__)

    # %% DOF MAP ON BOUNDARY
    @property
    def dof_map_boundary(self):
        """
        #SUMMARY: This dof_map maps the dof on boundary to the local numbering
        #OUTPUTS:
            CrazyMesh: a tuple of 4 entries which correspond to the (S, N, W, E) boundaries
        """
        if self._dof_map_boundary is not None:
            return self._dof_map_boundary

        self._dof_map, self._dof_map_boundary = numbering(self, numbering_scheme=self.numbering_scheme)

        return self._dof_map_boundary

    # %%
    @property
    def corresponding2(self):
        return self._corresponding2

    @corresponding2.setter
    def corresponding2(self, corresponding2):
        self._corresponding2 = corresponding2
        self._dof_map_local_trace = None

    @property
    def basis_trace(self):
        if self.corresponding2 == 'TraceGauss0form':
            return ('lobatto_edge', self.p[0]), ('lobatto_edge', self.p[1])
        else:
            raise Exception("<FORM> <GL> : basis_trace corresponding to {} not coded yet".format(self.corresponding2))

    # %% DOF MAP LOCAL TRACE
    @property
    def dof_map_local_trace(self):
        """
        #SUMMARY: This dof_map maps the dof on boundary to the local numbering
                  For Example: p = 3 and corresponding2 TraceGauss0form
                      dof_map_local_trace =
                      ([  0   1  2  ],
                       [  9   10  11 ],
                       [ 12  13  14 ],
                       [ 21  22  23 ])
                      The rows correspond to the S, N, W, E boundaries
        """

        if self._dof_map_local_trace is not None:
            return self._dof_map_local_trace

        if self.corresponding2 == 'TraceGauss0form':
            px, py = self.p
            num_cells = px * (py + 1) + py * (px + 1)
            half_num_cells = px * (py + 1)
            dof_map_local_trace_S = np.array([int(i * (py + 1)) for i in range(px)])
            # dof_map_local_trace_S = np.array([int(i) for i in range(px)])
            dof_map_local_trace_N = np.array([int((i + 1) * (py + 1) - 1) for i in range(px)])
            # dof_map_local_trace_N = np.array([int(i) for i in range(self.num_basis_xi-px, self.num_basis_xi)])
            dof_map_local_trace_W = np.array([int(i) for i in range(half_num_cells, half_num_cells + py)])
            dof_map_local_trace_E = np.array([int(i) for i in range(num_cells - py, num_cells)])

            self._dof_map_local_trace = (dof_map_local_trace_S, dof_map_local_trace_N,
                                         dof_map_local_trace_W, dof_map_local_trace_E)
        else:
            raise Exception("<FORM> <GL> : dof_map_local_trace corresponding to {} not coded yet"
                            .format(self.corresponding2))

        return self._dof_map_local_trace

    @property
    def coboundary(self):
        """
        #SUMMARY: Compute the E21
        # INPUTS:
        #   OPTIONAL:
        #OUTPUTS: [0] E21
                  [1] (OPTIONAL) E21_assembled
                  [2] (OPTIONAL) _2form
        """
        if self.is_inner is True:
            E21 = -self.E21
        else:
            E21 = self.E21

        _2form = Lobatto2form(self.mesh, self.p, is_inner=self.is_inner)
        E21_a = assemble(E21, _2form.dof_map, self.dof_map)
        if self._cochain is not None:
            _2form.cochain = E21_a.dot(self.cochain)
        return E21, E21_a, _2form

    # %% COCHAIN
    @property
    def cochain_xi(self):
        """Return the dx component of the cochain."""
        return self._cochain[:self.basis.num_basis_xi]

    @property
    def cochain_eta(self):
        """Return the dy component of the cochain."""
        return self._cochain[-self.basis.num_basis_eta:]

    # %% SPLITE COCHAIN
    def split_cochain(self, cochain):
        """Split the cochain in the dx and dy component."""
        return cochain[:self.num_basis_xi], cochain[-self.num_basis_eta:]

    # %% EVALUATE BASIS
    def evaluate_basis(self, domain=None):
        """
        #SUMMARY: Update self.xi, eta and basis.
        """
        if domain is None:
            self._xi, self._eta = np.meshgrid(self._quad_nodes[0], self._quad_nodes[1])
            nodal_basis_1d = [functionals.lagrange_basis(self._nodal_nodes[i], self._quad_nodes[1 - i]) for i in range(2)]
            edge_basis_1d = [functionals.edge_basis(self._edge_nodes[i], self._quad_nodes[i]) for i in range(2)]
            self._basis = np.zeros((self.num_basis, np.size(self._quad_nodes[0]) * np.size(self._quad_nodes[1])))
            self._evaluate_basis_domain = (self._quad_nodes[0], self._quad_nodes[1])
        else:
            self._xi, self._eta = np.meshgrid(domain[0], domain[1])
            nodal_basis_1d = [functionals.lagrange_basis(self._nodal_nodes[i], domain[1 - i]) for i in range(2)]
            edge_basis_1d = [functionals.edge_basis(self._edge_nodes[i], domain[i]) for i in range(2)]
            self._basis = np.zeros((self.num_basis, np.size(domain[0]) * np.size(domain[1])))
            self._evaluate_basis_domain = domain

        if self.numbering_scheme == 'symmetric1':
            self._basis[:self.num_basis_xi] = np.kron(nodal_basis_1d[0], edge_basis_1d[0])
            self._basis[-self.num_basis_eta:] = np.kron(nodal_basis_1d[1], edge_basis_1d[1])

            # self.basis_1edge = np.kron(edge_basis_1d[0], np.ones(np.shape(edge_basis_1d[0])))
            # self.basis_1node = np.kron(nodal_basis_1d[0], np.ones(np.shape(nodal_basis_1d[0])))
            # self.basis_1node = nodal_basis_1d[0]
        # sio.savemat('edgebasis', mdict={'edgebasis': self.basis_1node})

        # TODO
        if self.numbering_scheme == 'general' or None:
            self._basis[:self.num_basis_xi] = np.kron(edge_basis_1d[0], nodal_basis_1d[0])
            self._basis[-self.num_basis_eta:] = np.kron(nodal_basis_1d[1], edge_basis_1d[1])

        # plt.matshow(self.basis[:self.num_basis_xi])
        # plt.colorbar()
        # # plt.set_cmap('YlGnBu')
        # plt.set_cmap('viridis')
        # plt.matshow(nodal_basis_1d[0])
        # plt.colorbar()
        # plt.matshow(edge_basis_1d[0])
        # plt.colorbar()
        # plt.show()

    # %% BASIS XI AND ETA
    @property
    def basis_xi(self):
        """Return the basis related to the dx component of the 1-form."""
        return self.basis[:self.num_basis_xi]

    @property
    def basis_eta(self):
        """Return the basis related to the dy component of the 1-form."""
        return self.basis[-self.num_basis_eta:]

    # The incidence matrix maker
    @property
    def E21(self):
        """
        #SUMMARY:
        """
        if self._E21 is not None:
            return self._E21

        if self.numbering_scheme is "general":
            px, py = self.p
            total_vol = px * py
            total_edges = px * (py + 1) + py * (px + 1)
            E21 = np.zeros((total_vol, total_edges))
            for i in range(px):
                for j in range(py):
                    volij = i * py + j
                    edge_bottom = i * (py + 1) + j
                    edge_top = i * (py + 1) + j + 1
                    edge_left = (py + 1) * px + i * py + j
                    edge_right = (py + 1) * px + (i + 1) * py + j
                    E21[volij, edge_bottom] = -1
                    E21[volij, edge_top] = +1
                    E21[volij, edge_left] = +1
                    E21[volij, edge_right] = -1

        elif self.numbering_scheme == 'symmetric2':
            px, py = self.p
            total_vol = px * py
            total_edges = px * (py + 1) + py * (px + 1)
            E21 = np.zeros((total_vol, total_edges))
            for i in range(px):
                    for j in range(py):
                        volij = i * py + j

                        edge_bottom = i * py + j
                        edge_top = (i + 1) * py + j
                        edge_left = (py + 1) * px + j * py + i
                        edge_right = (py + 1) * px + (j + 1) * py + i

                        E21[volij, edge_bottom] = -1
                        E21[volij, edge_top] = +1
                        E21[volij, edge_left] = +1
                        E21[volij, edge_right] = -1

        elif self.numbering_scheme == 'symmetric1':
            px, py = self.p
            total_vol = px * py
            total_edges = px * (py + 1) + py * (px + 1)
            E21 = np.zeros((total_vol, total_edges))
            for i in range(px):
                for j in range(py):
                    volij = i * py + j

                    edge_bottom = j * py + i
                    edge_top = (j + 1) * py + i
                    edge_left = (py + 1) * px + i * py + j
                    edge_right = (py + 1) * px + (i + 1) * py + j

                    E21[volij, edge_bottom] = -1
                    E21[volij, edge_top] = +1
                    E21[volij, edge_left] = +1
                    E21[volij, edge_right] = -1
        else:
            raise Exception("Numbering scheme either not specified or not implemented with d operator")

        if self._is_inner is True:
            self._E21 = - E21
        else:
            self._E21 = E21

        return self._E21

    # %% WEIGHTED METRIC TENSOR RELATED
    def weighted_metric_tensor(self, xi, eta, K):
        """Calculate the metric tensor weighted with the constitutive law."""
        assert K.__class__.__name__ in ('AnisoDifTen',)

        K.eval_tensor(xi, eta)

        if self.is_inner:
            k_11, k_12, k_22 = K.tensor
        else:
            k_11, k_12, k_22 = K.inverse

        dx_deta = self.mesh.dx_deta(xi, eta)
        dx_dxi = self.mesh.dx_dxi(xi, eta)
        dy_deta = self.mesh.dy_deta(xi, eta)
        dy_dxi = self.mesh.dy_dxi(xi, eta)
        g = dx_dxi * dy_deta - dx_deta * dy_dxi
        g_11 = (dx_deta ** 2 * k_11 +
                2 * dy_deta * dx_deta * k_12 +
                dy_deta ** 2 * k_22) / g

        g_12 = -(dx_dxi * dx_deta * k_11 +
                 (dy_dxi * dx_deta + dx_dxi
                  * dy_deta) * k_12 + dy_dxi * dy_deta * k_22) / g

        g_22 = (dx_dxi ** 2 *
                k_11 + 2 * dy_dxi * dx_dxi * k_12 + dy_dxi ** 2 * k_22) / g
        """IMPORTENT: g_11, g_12, g_22 actually are g_11*g, g_12*g, g_22*g"""
        return g_11, g_12, g_22

    # %% THE INNER PRODUCT
    def inner(self, other, K=None):
        """
        #SUMMARY: Compute inner product.
        #OUTPUTS: [0] The Inner Matrix
        #             self .basis -> 0 axis
        #             other.basis -> 1 axis
        """
        assert self.mesh is other.mesh, "<FORM> <GL> : Mesh of the forms do not match"
        _, _, quad_weights_2d = self._do_same_quad_as(other)

        # calculate metric components
        if K is not None:
            # metric tensor weighted by constitutive laws
            assert K.mesh is self.mesh, "<FORM> <GL> : K.mesh is not self.mesh"
            g_11, g_12, g_22 = self.weighted_metric_tensor(
                self.xi.ravel('F'), self.eta.ravel('F'), K)
        else:
            # usual metric terms
            g_11, g_12, g_22 = self.mesh.metric_tensor(self.xi.ravel('F'), self.eta.ravel('F'))

        M_1 = np.zeros((self.num_basis, other.num_basis, self.mesh.num_elements))

        M_1[:self.num_basis_xi, :other.num_basis_xi] = np.tensordot(
            self.basis_xi, other.basis_xi[:, :, np.newaxis] * (g_11 * quad_weights_2d), axes=(1, 1))

        # special one
        # edge_inner = np.tensordot(
        #     self.basis_1edge, self.basis_1edge[:, :, np.newaxis] * (g_11 * quad_weights_2d), axes=(1, 1))
        # sio.savemat("edge_inner", mdict={"edge_inner": edge_inner})
        # nodal_inner = np.tensordot(
        #     self.basis_1node, self.basis_1node[:, :, np.newaxis] * (g_11 * quad_weights_2d), axes=(1, 1))
        # sio.savemat("nodal_inner", mdict={"nodal_inner": nodal_inner})
        # nodal_inner = np.tensordot(
        #     self.basis_1node, self.basis_1node * (quad_weights_1d), axes=(1, 1))
        # sio.savemat("nodal_inner", mdict={"Nodal_inner": nodal_inner})

        M_1[:self.num_basis_xi, -other.num_basis_eta:] = np.tensordot(
            self.basis_xi, other.basis_eta[:, :, np.newaxis] * (g_12 * quad_weights_2d), axes=(1, 1))

        M_1[-self.num_basis_eta:, :other.num_basis_xi] = np.tensordot(
            self.basis_eta, other.basis_xi[:, :, np.newaxis] * (g_12 * quad_weights_2d), axes=(1, 1))
        #            M_1[:self.num_basis_xi, -other.num_basis_eta:, :], (1, 0, 2))

        M_1[-self.num_basis_eta:, -other.num_basis_eta:] = np.tensordot(
            self.basis_eta, other.basis_eta[:, :, np.newaxis] * (g_22 * quad_weights_2d), axes=(1, 1))

        return M_1

    # %% THE MASS MATRIX
    def M(self, K=None):
        """
        #SUMMARY: Compute the Mass Matrix: M
        #OUTPUTS: [0] Local M
                  [1] Assembled Mass M
        """
        M = self.inner(self, K)

        # # extra lines to show local mass matrix
        # M_print = np.matrix(M)
        # print('\n\n===mass_local===\n\n', M_print)
        # plt.matshow(M_print)
        # plt.colorbar()
        # plt.clim(-1, 1)
        # plt.set_cmap('YlGnBu')
        #
        # # extra line to show global mass matrix
        # M_assemble = assemble(M, self.dof_map, self.dof_map).todense()
        # print('\n\n===mass_global===\n\n', M_assemble)
        # plt.matshow(M_assemble)
        # plt.colorbar()
        # plt.clim(-1, 1)
        # plt.set_cmap('YlGnBu')
        # plt.show()

        return M, assemble(M, self.dof_map, self.dof_map)

    def invM(self, K=None):
        """
        #SUMMARY: Compute the inversed Mass Matrix: M^{-1}
        #OUTPUTS: [0] Local M^{-1}
        """
        return inv(np.rollaxis(self.M(K)[0], 2, 0))

    # %% THE HODGE MATRIX
    def H(self, _1form, update_self_cochain=True):
        """
        #SUMMARY: Compute the Hodge Matrix H: self = H * _2form
        #OUTPUTS: [0] H
                  [1] Assembled H
        """
        assert self.is_inner is not _1form.is_inner, \
            "<FORM> <GL> : Hodge needs to connect two differently oriented forms"
        if _1form.__class__.__name__ in ('ExtGauss1form',):
            W = _1form.wedged(self)
            H = np.tensordot(self.invM(), W, axes=(2, 0))
            H = np.rollaxis(H, 0, 3)
            if self.is_inner is True:
                H_a = -assemble(H, self.dof_map, _1form.dof_map_internal)
            else:
                H_a = assemble(H, self.dof_map, _1form.dof_map_internal)

            if _1form.cochain_internal is not None and update_self_cochain is True:
                self.cochain = H_a.dot(_1form.cochain_internal)
                if _1form.u is not None and _1form.v is not None:
                    self.func = (_1form.u, _1form.v)
            return H, H_a

    # %% THE WEDGE PRODUCT
    def wedged(self, other):
        """
        #SUMMARY: Integrate the wedge product of two basis
                  Remember, Wedge product is metric free, so W is local
        #OUTPUTS: [0] The Wedge Matrix: W
        #             other.basis -> 0 axis
        #             self .basis -> 1 axis
        """
        assert self.mesh is other.mesh, \
            "<FORM> <GL> : Mesh of the forms do not match"
        assert self.k + other.k == self.mesh.dim, \
            '<FORM> <GL> : k-form wedge l-form, k+l should be equal to n'
        _, quad_weights, _ = self._do_same_quad_as(other)

        quad_weights_2d = np.kron(quad_weights[0], quad_weights[1]
                                  ).reshape(1, np.size(quad_weights[0]) * np.size(quad_weights[1]))

        W = np.zeros((other.num_basis, self.num_basis))

        W[:other.num_basis_xi, -self.num_basis_eta:] = \
            np.tensordot(other.basis_xi, self.basis_eta * quad_weights_2d, axes=(1, 1))

        W[-other.num_basis_eta:, :self.num_basis_xi] = \
            -np.tensordot(other.basis_eta, self.basis_xi * quad_weights_2d, axes=(1, 1))

        return W

    # %% DISCRETIZATION OR REDUCTION
    def discretize(self, func, quad=None):
        """
        #SUMMARY: Project a function onto a finite dimensional space of 1-forms.
        """
        self.func = func

        if quad is None:
            quad = (('gauss', 'gauss'), (self.p[0] + 2, self.p[1] + 2))
        quad_nodes, quad_weights, _ = self._evaluate_quad_grid(quad)

        _, (p_x, p_y) = quad
        if _[0] == 'gauss':
            p_x -= 1

        if _[1] == 'gauss':
            p_y -= 1

        xi_ref, eta_ref = np.meshgrid(quad_nodes[0], quad_nodes[1])

        edges_size = [self._edge_nodes[i][1:] - self._edge_nodes[i][:-1] for i in range(2)]

        magic_factor = 0.5
        cell_nodes = [(0.5 * (edges_size[i][np.newaxis, :]) *
                       (quad_nodes[i][:, np.newaxis] + 1) + self._edge_nodes[i][:-1]).ravel('F') for i in range(2)]

        quad_eta_for_dx = np.tile(self._edge_nodes[1], (p_x + 1, self.p[0]))
        quad_xi_for_dx = np.repeat(cell_nodes[0].reshape(
            p_x + 1, self.p[0], order='F'), self.p[1] + 1, axis=1)

        quad_xi_for_dy = np.repeat(
            self._edge_nodes[0], (p_y + 1) * self.p[1]).reshape(p_y + 1, (self.p[0] + 1) * self.p[1], order='F')
        quad_eta_for_dy = np.tile(cell_nodes[1].reshape(
            p_y + 1, self.p[1], order='F'), (1, self.p[0] + 1))

        x_dx, y_dx = self.mesh.mapping(quad_xi_for_dx, quad_eta_for_dx)

        cochain_local_xi = np.tensordot(quad_weights[0],
                                        (self.func_in_form[0](x_dx, y_dx) * self.mesh.dx_dxi(quad_xi_for_dx, quad_eta_for_dx)
                                        + self.func_in_form[1](x_dx, y_dx) * self.mesh.dy_dxi(quad_xi_for_dx, quad_eta_for_dx)),
                                        axes=(0, 0)
                                        ) * np.repeat(edges_size[0] * magic_factor, self.p[1] + 1).reshape(self.p[0] * (self.p[1] + 1), 1)

        x_dy, y_dy = self.mesh.mapping(quad_xi_for_dy, quad_eta_for_dy)

        cochain_local_eta = np.tensordot(quad_weights[1],
                                         (self.func_in_form[0](x_dy, y_dy) * self.mesh.dx_deta(quad_xi_for_dy, quad_eta_for_dy)
                                         + self.func_in_form[1](x_dy, y_dy) * self.mesh.dy_deta(quad_xi_for_dy, quad_eta_for_dy)),
                                         axes=(0, 0)
                                         ) * np.tile(edges_size[1] * magic_factor, (self.p[0] + 1, 1)).reshape(self.p[1] * (self.p[0] + 1), 1)

        self.cochain_local = np.vstack((cochain_local_xi, cochain_local_eta))

        print(' <FORM> <{}> : Red; Er={:.10f}'.format(self.__class__.__name__, self.L2_error()[0]))

    # %% RECONSTRUCTION
    def reconstruct(self, xi=None, eta=None):
        """Reconstruct the form on the computational domain."""

        assert self.cochain is not None, "<FORM> <GL> : no cochain to reconstruct"

        xi, eta = self._check_xi_eta(xi, eta, factor=1000)

        self.evaluate_basis(domain=(xi, eta))
        self._x, self._y = self.mesh.mapping(self.xi, self.eta)

        xi, eta = self.xi.ravel('F'), self.eta.ravel('F')
        cochain_xi, cochain_eta = self.split_cochain(self.cochain_local)
        g = self.mesh.g(xi, eta)

        self._reconstructed_dx = 1 / g * (self.mesh.dy_deta(xi, eta) * np.tensordot(self.basis_xi, cochain_xi, axes=(0, 0)) -
                                          self.mesh.dy_dxi(xi, eta) * np.tensordot(self.basis_eta, cochain_eta, axes=(0, 0)))

        self._reconstructed_dy = 1 / g * (- self.mesh.dx_deta(xi, eta) * np.tensordot(self.basis_xi, cochain_xi, axes=(0, 0)) +
                                          self.mesh.dx_dxi(xi, eta) * np.tensordot(self.basis_eta, cochain_eta, axes=(0, 0)))

    # %% reconstructed
    @property
    def reconstructed_dx(self):
        if self._reconstructed_dx is not None:
            return self._reconstructed_dx
        else:
            raise Exception("<FORM> <GL> : reconstructed_dx is None, first do reconstruct")

    @property
    def reconstructed_dy(self):
        if self._reconstructed_dy is not None:
            return self._reconstructed_dy
        else:
            raise Exception("<FORM> <GL> : reconstructed_dy is None, first do reconstruct")

    @property
    def reconstructed_all(self):
        return (self.reconstructed_dx, self.reconstructed_dy)

    # %% PLOT SELF.MESH
    def plot_mesh(self, regions=None, elements=None, plot_density=10):
        if self.separated_dof is True:
            self.mesh.plot_mesh(regions=regions, elements=elements, plot_density=plot_density,
                                internal_mesh_type=('lobatto_separated', self.p))
        else:
            self.mesh.plot_mesh(regions=regions, elements=elements, plot_density=plot_density,
                                internal_mesh_type=('lobatto', self.p))


# %% THE MAIN TEST PART
if __name__ == '__main__':
    from mesh_crazy import CrazyMesh

    p = (5, 5)
    n = (1, 1)
    c = 0.15
    domain = ((-1, 1), (-1, 1))
    mesh = CrazyMesh(elements_layout=n, curvature=c, bounds_domain=domain)


    def u(x, y): return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


    def v(x, y): return np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)

    f1 = Lobatto1form(mesh, p, separated_dof=True, numbering_scheme='symmetric')
    f1.discretize((u, v))
    # f1.plot_self()
    f1.plot_mesh(plot_density=4)

    # _, _, f2 = f1.coboundary
    # f2.plot_self()
