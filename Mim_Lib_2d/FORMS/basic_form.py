# -*- coding: utf-8 -*-
"""
Serve as a parent for all forms

@author: Yi Zhang. Created on Oct 10 2017 in KL6008 from PDX to AMS
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import numpy as np
import warnings
import functionals
import copy

from numpy.linalg import inv
from assemble import assemble
from numbering import numbering
from basic import Basic

import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'bwr'
plt.rc('text', usetex=False)
font = {'fontname': 'Times New Roman',
        'color': 'k',
        'weight': 'normal',
        'size': 12}


# %% CLASS BODY
class BasicForm(Basic):
    # %%
    def __init__(self, mesh, p, numbering_scheme=None, name=None, info=None):
        # to call this __init__, use: super().__init__(*args) in children
        super().__init__(mesh, p, numbering_scheme, name, info)

        self._num_basis = None
        self._quad_nodes = [0, 0]
        self._quad_weights = [0, 0]

        # OPTIONALS:-----------------------------------------------------------
        self._form = None
        self._basis = None
        self._xi = None
        self._eta = None

        self._x = None
        self._y = None

        self._cochain = None
        self._cochain_local = None
        self._num_dof = None
        self._dof_map = None

        self._M = None
        self._M_a = None

    # %% FUNC RELATED
    @property
    def func(self):
        return self._func

    @property
    def func_in_form(self):
        return self._func

    @func.setter
    def func(self, func):
        assert callable(func)
        self._func = func

    # %% DOF MAP RELATED
    @property
    def num_dof(self):
        """Return the number of degrees of freedom."""
        if self._num_dof is None:
            self._num_dof = np.max(self.dof_map) + 1
        return self._num_dof

    @property
    def dof_map(self):
        if self._dof_map is not None: return self._dof_map
        global_numbering = numbering(self, numbering_scheme=self.numbering_scheme)
        self._dof_map = global_numbering
        return self._dof_map

    # %% COCHAIN RELATED
    @property
    def cochain(self):
        if self._cochain is None:
            print(" <FORM> : Empty Cochain Warning")
        return self._cochain

    @cochain.setter
    def cochain(self, cochain):
        try:
            assert np.shape(cochain) == (self.num_dof,)
            self._cochain = cochain
            self._cochain_local = self.cochain[np.transpose(self.dof_map)]
        except AssertionError:
            raise AssertionError(
                "The dofs of the cochain do not match the dofs of the function space.shape cochain {0}, number of degrees of freedom : {1}"
                    .format(np.shape(cochain), self.num_dof))

    # %% COCHAIN LOCAL RELATED
    @property
    def cochain_local(self):
        """Map the cochain elements into local dof with the dof map."""
        if self._cochain_local is None:
            self._cochain_local = self.cochain[np.transpose(self.dof_map)]
        return self._cochain_local

    @cochain_local.setter
    def cochain_local(self, cochain):
        assert np.shape(cochain) == (self.num_basis, self.mesh.num_elements), \
            'cochain shape wrong, {} != {}'.format(np.shape(cochain), (self.num_basis, self.mesh.num_elements))

        self._cochain_local = cochain
        self._cochain_to_global

    @property
    def _cochain_to_global(self):
        """Map the local dofs of the cochain into the global cochain."""
        self._cochain = np.zeros(self.num_dof)
        dof_map = np.transpose(self.dof_map)
        # reorder degrees of freedom
        for i, row in enumerate(self.cochain_local):
            for j, dof_value in enumerate(row):
                self._cochain[dof_map[i, j]] = dof_value

    # %% QUAD GRID DEFAULT 
    @property
    def quad_grid(self):
        return self._quad_type, self._quad_order

    # %% PRIVATE EVALUATE QUAD GRID
    def _evaluate_quad_grid(self, quad=None):
        """
        #SUMMARY: Evaluate quad grid, but no update self quad_grid
        #OUTPUTS: [0] quad_nodes
                  [1] quad_weights
        """
        if quad is None:
            quad_nodes = self._quad_nodes
            quad_weights = self._quad_weights
        else:
            quad_nodes = [0, 0]
            quad_weights = [0, 0]
            quad_nodes[0], quad_weights[0] = getattr(functionals, quad[0][0] + '_quad')(quad[1][0])
            quad_nodes[1], quad_weights[1] = getattr(functionals, quad[0][1] + '_quad')(quad[1][1])

        quad_weights_2d = np.kron(quad_weights[0], quad_weights[1]).reshape(
            np.size(quad_weights[0]) * np.size(quad_weights[1]), 1)
        return quad_nodes, quad_weights, quad_weights_2d

    # %% Determine common quad
    def _do_same_quad_as(self, other):
        """
        When self and other's quad are different, make them the same
        """
        if self.quad_grid == other.quad_grid:
            quad_nodes, quad_weights, quad_weights_2d = self._evaluate_quad_grid()
        else:
            p0 = np.max([self.quad_grid[1][0], other.quad_grid[1][0]])
            p1 = np.max([self.quad_grid[1][1], other.quad_grid[1][1]])
            quad_nodes, quad_weights, quad_weights_2d = self._evaluate_quad_grid(
                (('gauss', 'gauss'), (p0, p1)))

        self.evaluate_basis(quad_nodes)
        other.evaluate_basis(quad_nodes)

        return quad_nodes, quad_weights, quad_weights_2d

    # %% EVALUATE BASIS (MUST BE OVER WRITTEN)
    def evaluate_basis(self, domain=None):
        """
        #SUMMARY: Update self.xi, eta and basis.
        """
        pass

    @property
    def evaluate_basis_domain(self):
        return self._evaluate_basis_domain

    # %% self.xi, .eta, .basis
    @property
    def basis(self):
        return self._basis

    @property
    def xi(self):
        return self._xi

    @property
    def eta(self):
        return self._eta

    # %% INNER PRODUCT
    def inner(self, other):
        """
        #SUMMARY: Compute inner product.
        #OUTPUTS: [0] The Inner Matrix
        """
        pass

    # %% MASS MATRIX
    @property
    def M(self):
        """
        #SUMMARY: Compute the Mass Matrix: M
        #OUTPUTS: [0] Local M
                  [1] Assembled Mass M
        """
        if self._M is not None and self._M_a is not None:
            return self._M, self._M_a

        self._M = self.inner(self)
        self._M_a = assemble(self._M, self.dof_map, self.dof_map)
        return self._M, self._M_a

    @property
    def invM(self):
        """
        #SUMMARY: Compute the inversed local Mass Matrix: M^{-1}
        #OUTPUTS: [0] Local M^{-1}
        """
        return inv(np.rollaxis(self.M[0], 2, 0))

    # %% WEDGE PRODUCT
    def wedged(self, other):
        """
        #SUMMARY: Integrate the wedge product of two basis
        #OUTPUTS: [0] The Wedge Matrix: W
        #             other.basis -> 0 axis
        #             self .basis -> 1 axis
        """
        pass

    # %% DISCRETIZTION / REDUCTION
    def discretize(self, func):
        """
        #SUMMARY: Project a function onto a finite dimensional space of 0-forms.
        """
        pass

    # %% check xi, eta
    def _check_xi_eta(self, xi, eta, factor=10000, factor1=0.999):
        """
        USE this to chech xi and eta
        """
        if xi is None and eta is None:
            xi = eta = np.linspace(
                -1, 1, np.int(np.ceil(np.sqrt(factor / self.mesh.num_elements)) + 1))

        if self.mesh.anti_corner_singularity is True:
            xi = factor1 * xi
            eta = factor1 * eta

        return xi, eta

    # %% RECONSTRUCTION
    def reconstruct(self, xi=None, eta=None):
        """
        #SUMMARY: Reconstruct the 0-form on the physical domain.
        # UPDATE: xi; eta; basis; reconstructed;
        """
        warnings.warn("No construct for form:{}".format(self.what_form))

    # %% RECONSTRUCTED
    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def reconstructed_all(self):
        return self._reconstructed

    # %% PLOT SELF
    def plot_self(self, xi=None, eta=None, do_plot=True, do_return=False, num_levels=20, plot_type='quiver'):
        """
        #SUMMARY: plot self on the computational domain.
        #OUTPUTS: the plot data
        """
        assert self.what_form in title_dict.keys(), " <FORM> : {} can not be ploted".format(self.what_form)
        self.reconstruct(xi, eta)

        # %% plot 0-forms
        if self.k == 0 and self.is_form is True:
            return _plot_0form(self, do_plot, do_return, num_levels)

        # %% plot 1-forms
        elif self.k == 1 and self.is_form is True:
            return _plot_1form(self, do_plot, do_return, num_levels, plot_type)

        # %% plot 2-forms
        elif self.k == 2 and self.is_form is True:
            return _plot_2form(self, do_plot, do_return, num_levels)

    # %% L2_ERROR CALCULATOR
    def L2_error(self, func=None, quad=None):
        """
        #SUMMARY: Compute the L2_error
        #OUTPUTS: [0] global_error
                  [1] local_error
        """
        if func is not None: self.func = func
        if quad is None: quad = (('gauss', 'gauss'), (self.p[0] + 3, self.p[1] + 3))
        quad_nodes, quad_weights, quad_weights_2d = self._evaluate_quad_grid(quad)
        self.reconstruct(quad_nodes[0], quad_nodes[1])
        pts_per_element = np.size(quad_weights_2d)
        x, y = self.x, self.y
        local_error = 0
        if np.shape(self.func_in_form) == ():
            func_eval = self.func_in_form(x, y).reshape(
                pts_per_element, self.mesh.num_elements, order='F')
            local_error += (self.reconstructed_all - func_eval) ** 2
        else:
            assert np.shape(self.func_in_form)[0] == np.shape(self.reconstructed_all)[0], " <FORM> : WRONG"
            I = np.shape(self.func_in_form)[0]
            for i in range(I):
                func_eval = self.func_in_form[i](x, y).reshape(
                    pts_per_element, self.mesh.num_elements, order='F')
                local_error += (self.reconstructed_all[i] - func_eval) ** 2

        g = self.mesh.g(self.xi, self.eta).reshape(
            pts_per_element, self.mesh.num_elements, order='F')
        # integrate to get the l_2 norm
        global_error = local_error * g * quad_weights_2d

        return np.sum(global_error) ** 0.5, np.sum(local_error) ** 0.5

    # %% OPERATORS +
    def __sub__(self, other):
        assert self.__class__.__name__ == other.__class__.__name__, ' <FORM> : a-b, a and b should be the same class, a is {}, b is{}'.format(
            self.__class__.__name, other.__class__.__name)
        assert self.mesh is other.mesh, ' <FORM> : a-b, a,b should of the same mesh, a.mesh={}, b.mesh={}'.format(self.mesh, other.mesh)
        assert self.p == other.p, ' <FORM> : a-b, a,b should of the same polynomial order p, a.p={}, b.p={}'.format(self.p, other.p)
        assert self.is_inner is other.is_inner, ' <FORM> : a-b, a,b should of the same is_inner, a.is_inner={}, b.is_inner={}'.format(
            self.is_inner, other.is_inner)
        assert self.orientation is other.orientation, '<FORM> : a-b, a,b should of the same orientation, a.orientation={}, b.orientation={}'.format(
            self.orientation, other.orientation)
        assert self.separated_dof is other.separated_dof, ' <FORM> : a-b, a,b should of the same orientation, a.separated_dof={}, b.separated_dof={}'.format(
            self.separated_dof, other.separated_dof)

        assert self.cochain is not None, " <FORM> : can not do a-b, because a's cochain is None"
        assert other.cochain is not None, " <FORM> : can not do a-b, because b's cochain is None"

        result = copy.copy(self)  # get a copy of self
        result.cochain = self.cochain - other.cochain
        return result

    # %% OPERATORS -
    def __add__(self, other):
        assert self.__class__.__name__ == other.__class__.__name__, ' <FORM> : a+b, a and b should be the same class, a is {}, b is{}'.format(
            self.__class__.__name, other.__class__.__name)
        assert self.mesh is other.mesh, ' <FORM> : a+b, a,b should of the same mesh, a.mesh={}, b.mesh={}'.format(self.mesh, other.mesh)
        assert self.p == other.p, ' <FORM> : a+b, a,b should of the same polynomial order p, a.p={}, b.p={}'.format(self.p, other.p)
        assert self.is_inner is other.is_inner, ' <FORM> : a+b, a,b should of the same is_inner, a.is_inner={}, b.is_inner={}'.format(
            self.is_inner, other.is_inner)
        assert self.orientation is other.orientation, ' <FORM> : a+b, a,b should of the same orientation, a.orientation={}, b.orientation={}'.format(
            self.orientation, other.orientation)
        assert self.separated_dof is other.separated_dof, ' <FORM> : a+b, a,b should of the same orientation, a.separated_dof={}, b.separated_dof={}'.format(
            self.separated_dof, other.separated_dof)

        assert self.cochain is not None, " <FORM> : can not do a+b, because a's cochain is None"
        assert other.cochain is not None, " <FORM> : can not do a+b, because b's cochain is None"

        result = copy.copy(self)  # get a copy of self
        result.cochain = self.cochain + other.cochain
        return result

    # %% PLOT SELF.MESH
    def plot_mesh(self, regions=None, elements=None, plot_density=10):
        self.mesh.plot_mesh(regions=regions, elements=elements, plot_density=plot_density)

        # %% PLOT FUNCTIONS ------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------


title_dict = {"0-lobatto": " Lobatto 0-form",
              "1-lobatto": " Lobatto 1-form",
              "2-lobatto": " Lobatto 2-form",

              "0-gauss": " Gauss 0-form",
              "1-gauss": " Gauss 1-form",
              "2-gauss": " Gauss 2-form",

              "0-ext_gauss": " ext_Gauss 0-form",
              "1-ext_gauss": " ext_Gauss 1-form",
              "2-ext_gauss": " ext_Gauss 2-form"}


# %%---------------------------------------------------------------------------
def _plot_0form(_0form, do_plot=True, do_return=False, num_levels=20):
    x, y = _0form.x, _0form.y
    num_pts_y, num_pts_x = np.shape(_0form.xi)
    reconstructed = _0form.reconstructed.reshape((num_pts_y, num_pts_x, _0form.mesh.num_elements), order='F')

    if do_plot is True:
        plt.figure()
        levels = np.linspace(np.min(reconstructed), np.max(reconstructed), num_levels)
        for i in range(_0form.mesh.num_elements):
            plt.contourf(x[..., i], y[..., i], reconstructed[..., i], levels=levels)
        plt.title(r'' + _0form.orientation + title_dict[_0form.what_form], fontdict=font)
        plt.colorbar()
        plt.axis("equal")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.show()

    if do_return is True:
        return x, y, reconstructed


# %%---------------------------------------------------------------------------
def _plot_1form(_1form, do_plot=True, do_return=False, num_levels=20, plot_type="quiver"):
    x, y = _1form.x, _1form.y
    num_pts_y, num_pts_x = np.shape(_1form.xi)
    reconstructed_dx = _1form.reconstructed_dx.reshape((num_pts_y, num_pts_x, _1form.mesh.num_elements), order='F')
    reconstructed_dy = _1form.reconstructed_dy.reshape((num_pts_y, num_pts_x, _1form.mesh.num_elements), order='F')
    if do_plot is True:
        # %% QUIVER
        if plot_type == 'quiver':
            plt.figure()
            ax1, ax2, ax3 = np.shape(x)
            x_2d = x.reshape(ax1, ax2 * ax3, order='C')
            y_2d = y.reshape(ax1, ax2 * ax3, order='C')
            reconstructed_dx_2d = reconstructed_dx.reshape(ax1, ax2 * ax3, order='C')
            reconstructed_dy_2d = reconstructed_dy.reshape(ax1, ax2 * ax3, order='C')
            if _1form.is_inner is False:
                M = np.hypot(-reconstructed_dy_2d, reconstructed_dx_2d)
                plt.quiver(x_2d, y_2d, -reconstructed_dy_2d, reconstructed_dx_2d, M)
            else:
                M = np.hypot(reconstructed_dx_2d, reconstructed_dy_2d)
                plt.quiver(x_2d, y_2d, reconstructed_dx_2d, reconstructed_dy_2d, M)
            plt.title(r'vector field $(u,v)$ of ' + _1form.orientation + title_dict[_1form.what_form] + ' $(' + _1form.form + ')$', **font)
            plt.scatter(x_2d, y_2d, color='k', s=1)
            plt.axis("equal");
            plt.xlabel(r"$x$");
            plt.ylabel(r"$y$")
            plt.show()

        # %% STREAMPLOT
        elif plot_type == 'streamplot':
            if _1form.mesh.__class__.__name__ == 'CrazyMesh':
                num_pts_y, num_pts_x = np.shape(_1form.xi)
                num_el_x, num_el_y = _1form.mesh.n_x, _1form.mesh.n_y

                x_4d = x.reshape(num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
                x = np.moveaxis(x_4d, 2, 1).reshape(
                    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

                y_4d = y.reshape(num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
                y = np.rollaxis(y_4d, 2, 1).reshape(
                    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

                recon_4d_dx = _1form.reconstructed_dx.reshape(
                    num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
                reconstructed_dx = np.moveaxis(recon_4d_dx, 2, 1).ravel('F').reshape(
                    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

                recon_4d_dy = _1form.reconstructed_dy.reshape(
                    num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
                reconstructed_dy = np.moveaxis(recon_4d_dy, 2, 1).ravel('F').reshape(
                    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

                plt.figure()
                if _1form.is_inner is False:
                    plt.streamplot(x, y, -reconstructed_dy, reconstructed_dx, color='k')
                else:
                    plt.streamplot(x, y, reconstructed_dx, reconstructed_dy, color='k')
                plt.title(r'streamline of $(u,v)$ of ' + _1form.orientation + title_dict[_1form.what_form] + ' $(' + _1form.form + ')$',
                          **font)
                plt.axis("equal");
                plt.xlabel(r"$x$");
                plt.ylabel(r"$y$")
                plt.show()
            else:
                raise Exception("streamplot only works for CrazyMesh for the time being, self.mesh is {}"
                                .format(_1form.mesh.__class__.__name__))

        # %% CONTOURF
        else:
            plt.figure(figsize=(13, 6))
            plt.subplot(121)
            levels = np.linspace(np.min(reconstructed_dx), np.max(reconstructed_dx), num_levels)
            for i in range(_1form.mesh.num_elements):
                plt.contourf(x[..., i], y[..., i], reconstructed_dx[..., i], levels=levels)
                if _1form.is_inner is False:
                    plt.title(r'$v \mathrm{d}x$ of ' + _1form.orientation + title_dict[_1form.what_form] + ' $(' + _1form.form + ')$',
                              **font)
                else:
                    plt.title(r'$u \mathrm{d}x$ of ' + _1form.orientation + title_dict[_1form.what_form] + ' $(' + _1form.form + ')$',
                              **font)
            plt.colorbar()
            plt.axis("equal")
            plt.xlabel(r"$x$");
            plt.ylabel(r"$y$")

            plt.subplot(122)
            if _1form.is_inner is False:
                levels = np.linspace(np.min(-reconstructed_dy), np.max(-reconstructed_dy), num_levels)
                for i in range(_1form.mesh.num_elements):
                    plt.contourf(x[..., i], y[..., i], -reconstructed_dy[..., i], levels=levels)
                    plt.title(r'$u \mathrm{d}y$ of ' + _1form.orientation + title_dict[_1form.what_form] + ' $(' + _1form.form + ')$',
                              **font)
            else:
                levels = np.linspace(np.min(reconstructed_dy), np.max(reconstructed_dy), num_levels)
                for i in range(_1form.mesh.num_elements):
                    plt.contourf(x[..., i], y[..., i], reconstructed_dy[..., i], levels=levels)
                    plt.title(r'$v \mathrm{d}y$ of ' + _1form.orientation + title_dict[_1form.what_form] + ' $(' + _1form.form + ')$',
                              **font)
            plt.colorbar()
            plt.axis("equal");
            plt.xlabel(r"$x$");
            plt.ylabel(r"$y$")
            plt.show()

    # %% RETURN RECONSTRUCTED DATA
    if do_return is True:
        if _1form.is_inner is False:
            return x, y, -reconstructed_dy, reconstructed_dx
        else:
            return x, y, reconstructed_dx, reconstructed_dy


# %%---------------------------------------------------------------------------
def _plot_2form(_2form, do_plot=True, do_return=False, num_levels=20):
    x, y = _2form.x, _2form.y
    num_pts_y, num_pts_x = np.shape(_2form.xi)
    reconstructed = _2form.reconstructed.reshape((num_pts_y, num_pts_x,
                                                  _2form.mesh.num_elements), order='F')
    if do_plot is True:
        plt.figure()
        levels = np.linspace(np.min(reconstructed), np.max(reconstructed), num_levels)
        for i in range(_2form.mesh.num_elements):
            plt.contourf(x[..., i], y[..., i], reconstructed[..., i], levels=levels)

        plt.title(r'' + _2form.orientation + title_dict[_2form.what_form], fontdict=font)
        plt.colorbar()
        plt.axis("equal")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.show()

    if do_return is True:
        return x, y, reconstructed


# %%---------------------------------------------------------------------------
# --------------------------------DONE------------------------------------------
# ------------------------------------------------------------------------------

# %% THE MAIN TEST PART 
if __name__ == '__main__':
    import meshes_chooser

    p = (12, 11)
    n = (2, 3)
    c = 0.3
    domain = ((-1, 1), (-1, 1))
    mesh = meshes_chooser.mesh_No(0, elements_layout=n, bounds_domain=domain, c=c)

    fb = BasicForm(mesh, p)
