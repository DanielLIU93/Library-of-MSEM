# -*- coding: utf-8 -*-
"""
Bssic trace form, serve as a parent for all extended forms

@author: Yi Zhang （张仪）, Created on Fri Nov  3 13:11:54 2017
    Aerodynamics
    Faculty of Aerospace Engineering
    TU Delft
"""
#from basic_form import BasicForm
from basic import Basic
import numpy as np
import warnings
import importlib
import functionals

# %% THE CLASS BODY
class BasicExtendedFrom(Basic):
    """ """
    # %%
    def __init__(self, internal_form, external_form, the_form, trace_form,
                 mesh, p, numbering_scheme, name, info):
        super().__init__(mesh, p, numbering_scheme, name, info)
        
        self._is_extended_form = True
        
        self._the_form_file, self._the_form_Type = internal_form.split(" :: ")
        self._trace_form_file, self._trace_form_Type = external_form.split(" :: ")
        
        if the_form is None:
            internal = importlib.import_module(self._the_form_file)
            the_form = getattr(internal, self._the_form_Type)(
                    mesh, p, numbering_scheme=numbering_scheme)
        
        if trace_form is None:
            external = importlib.import_module(self._trace_form_file)
            trace_form = getattr(external, self._trace_form_Type)(
                    mesh, p, numbering_scheme=numbering_scheme)
        
        assert the_form.__class__.__name__ == self._the_form_Type
        assert the_form.mesh is self.mesh
        assert the_form.p == self.p
        assert the_form.numbering_scheme == self.numbering_scheme
        self._the_form = the_form

        assert trace_form.__class__.__name__ == self._trace_form_Type
        assert trace_form.mesh is self.mesh
        assert trace_form.p == self.p
        assert trace_form.numbering_scheme == self.numbering_scheme
        self._trace_form = trace_form
            
        assert self._the_form.quad_grid == self._trace_form.quad_grid
        assert self.the_form._form == self.trace_form._form
        
        self._form = self.the_form._form
        self._dof_map_boundary = None
        self._num_basis = self._the_form.num_basis + self._trace_form.num_basis
        
    # %% compose from a form and a trace form
    def compose(self, the_form, trace_form):
        """
        #SUMMARY: use this method to compose an extended_gauss_0form by
                  extended_gauss_0form = { gauss_0form, trace_gauss_0form}
        # INPUTS: [1] the gauss_0form
                  [2] the trace_gauss_0form
        """
        assert the_form.__class__.__name__ == self._the_form_Type
        assert trace_form.__class__.__name__ == self._trace_form_Type
        
        assert self.mesh is the_form.mesh and self.mesh is trace_form.mesh
        assert self.p == the_form.p == trace_form.p

        self._the_form = the_form
        self._trace_form = trace_form
    
    @property
    def the_form_file(self):
        return self._the_form_file
    
    def the_form_type(self):
        return self._the_form_Type
        
    def trace_form_file(self):
        return self._trace_form_file
    
    def trace_form_type(self):
        return self._trace_form_Type

    # %% PRIVATE EVALUATE QUAD GRID
    def _evaluate_quad_grid(self, quad=None):
        """
        #SUMMARY: Evaluate quad grid, but no update self quad_grid
        #OUTPUTS: [0] quad_nodes
                  [1] quad_weights
        """
        if quad is None:
            quad_nodes = self.the_form._quad_nodes
            quad_weights = self.the_form._quad_weights
        else:
            quad_nodes = [0, 0]
            quad_weights = [0, 0]
            quad_nodes[0], quad_weights[0] = getattr(functionals, quad[0][0]+'_quad')(quad[1][0])
            quad_nodes[1], quad_weights[1] = getattr(functionals, quad[0][1]+'_quad')(quad[1][1])

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
                    (('gauss','gauss'), (p0, p1)))

        self .evaluate_basis(quad_nodes)
        other.evaluate_basis(quad_nodes)

        return quad_nodes, quad_weights, quad_weights_2d
    
    # %% func
    @property
    def func(self):
        return self._the_form.func

    @property
    def func_in_form(self):
        return self._the_form._func

    @func.setter
    def func(self, func):
        self._the_form.func = func
    
    # %% FORM & TRACE FORM 
    @property
    def the_form(self):
        return self._the_form

    @property
    def trace_form(self):
        return self._trace_form
   
    # %% num_basis
    @property
    def num_basis_internal(self):
        return self.the_form._num_basis

    @property
    def num_basis_external(self):
        return self.trace_form._num_basis
    
    # %% DOF MAP
    @property
    def num_dof(self):
        return self._the_form.num_dof + self._trace_form.num_dof
    
    @property
    def num_dof_internal(self):
        return self._the_form.num_dof
    
    @property
    def num_dof_external(self):
        return self._trace_form.num_dof
    
    @property
    def dof_map(self):
        """
        #SUMMARY: notice that we number all trace dofs after internal dofs
        """
        return np.hstack((self._the_form.dof_map,
                          self._trace_form.dof_map + self._the_form.num_dof))
    
    @property
    def dof_map_internal(self):
        return self.the_form.dof_map
    
    @property
    def dof_map_external(self):
        return self.trace_form.dof_map
    
    @property
    def dof_map_boundary(self):
        """
        #SUMMARY: This dof_map get the number of dof on boundary
        #OUTPUTS:
            CrazyMesh: a tuple of 4 entries which correspond to the (S, N, W, E) boundaries
        """
        if self._dof_map_boundary is None:
            dof_map_boundary = self._trace_form.dof_map_boundary
            self._dof_map_boundary = ()
            for i, row in enumerate(dof_map_boundary):
                self._dof_map_boundary += (dof_map_boundary[i] + self.num_dof_internal,)
        return self._dof_map_boundary
    
    # %% COCHAIN
    @property
    def cochain(self):
        """
        #SUMMARY: return the cochain
        """
        if self._the_form.cochain is not None and self._trace_form.cochain is not None:
            return np.concatenate((self._the_form.cochain, self._trace_form.cochain))
        else:
            if self._the_form.cochain is None: warnings.warn("Empty_form_Cochain_Warning")
            if self._trace_form.cochain is None: warnings.warn("Empty_trace_form_Cochain_Warning")
            return None

    @cochain.setter
    def cochain(self, cochain):
        """
        #SUMMARY: set the cochain
        """
        self._the_form.cochain = cochain[:self._the_form.num_dof]
        self._trace_form.cochain = cochain[self._the_form.num_dof:]
        
    @property
    def cochain_local(self):
        return np.vstack((self._the_form.cochain_local, self._trace_form.cochain_local))

    @cochain_local.setter
    def cochain_local(self, local_cochain):
        self._the_form.cochain_local = local_cochain[:self._the_form.num_basis, self.mesh.num_elements]
        self._trace_form.cochain_local = local_cochain[self._the_form.num_basis:, self.mesh.num_elements]

    @property
    def cochain_internal(self):
        return self._the_form.cochain

    @cochain_internal.setter
    def cochain_internal(self, cochain_internal):
        self._the_form.cochain = cochain_internal

    @property
    def cochain_external(self):
        return self._trace_form.cochain

    @cochain_external.setter
    def cochain_external(self, cochain_external):
        self._trace_form.cochain = cochain_external
    
    # %% QUAD
    @property
    def quad_grid(self):
        return self._the_form.quad_grid
    
    # %% EVALUATE BASIS RELATED
    def evaluate_basis(self, domain=None):
        self.evaluate_the_form_basis(domain)
        self.evaluate_trace_form_basis(domain)
        
    def evaluate_the_form_basis(self, domain=None):
        self._the_form.evaluate_basis(domain)
        
    def evaluate_trace_form_basis(self, domain=None):
        self._trace_form.evaluate_basis(domain)
    
    @property
    def evaluate_the_form_basis_domain(self):
        return self._the_form._evaluate_basis_domain
    
    @property
    def evaluate_trace_form_basis_domain(self):
        return self._trace_form._evaluate_basis_domain
    
    @property
    def basis(self):
        return self._the_form.basis
    
    @property
    def xi(self):
        return self._the_form.xi

    @property
    def eta(self):
        return self._the_form.eta
    
    @property
    def form_basis(self):
        return self._the_form.basis
    
    @property
    def form_xi(self):
        return self._the_form.xi

    @property
    def form_eta(self):
        return self._the_form.eta
    
    @property
    def trace_basis(self):
        return self._trace_form.basis
    
    @property
    def trace_xi(self):
        return self._trace_form.basis
    
    @property
    def trace_eta(self):
        return self._trace_form.basis
    
    # %% discretize
    def discretize(self, func):
        """
        #SUMMARY: Project a function onto a finite dimensional space of 0-forms.
        """
        self._the_form.discretize(func)
        self._trace_form.discretize(func)
        
    # %% reconstruct
    def reconstruct(self, xi=None, eta=None):
        """
        #SUMMARY: Reconstruct the 0-form on the physical domain.
        # UPDATE: xi; eta; basis; reconstructed;
        """
        self._the_form.reconstruct(xi, eta)

    @property
    def x(self):
        return self._the_form.x

    @property
    def y(self):
        return self._the_form.y
    
    @property
    def reconstructed_all(self):
        return self.the_form.reconstructed_all
        
    # %% INNER PRODUCT
    def inner(self, other):
        """
        #SUMMARY: Compute inner product.
        #OUTPUTS: [0] The Inner Matrix
        #             self .basis -> 0 axis
        #             other.basis -> 1 axis
        """
        return self._the_form.inner(other)

    # %% MASS MATRIX
    @property
    def M(self):
        """
        #SUMMARY: Compute the Mass Matrix: M
        #OUTPUTS: [0] Local M
                  [1] Assembled Mass M
        """
        return self._the_form.M

    @property
    def invM(self):
        """
        #SUMMARY: Compute the inversed Mass Matrix: M^{-1}
        #OUTPUTS: [0] Local M^{-1}
        """
        return self._the_form.invM

    # %% HODGE MATRIX
    def H(self, _2form, update_self_cochain = True):
        """
        #SUMMARY: Compute the Hodge Matrix H: self = H * _2form
        #OUTPUTS: [0] H
                  [1] Assembled H
        """
        return self._the_form.H(_2form, update_self_cochain)

    # %% WEDGE
    def wedged(self, other):
        """
        #SUMMARY: Integrate the wedge product of two basis
        #OUTPUTS: [0] The Wedge Matrix: W
        #             other.basis -> 0 axis
        #             self .basis -> 1 axis
        """
        return self._the_form.wedged(other)
    
    # %% L2_error
    def L2_error(self, func=None, quad=None):
        """
        #SUMMARY: Compute the L2_error
        #OUTPUTS: [0] global_error
                  [1] local_error
        """
        return self._the_form.L2_error(func, quad)
    
    
    # %% L2_error
    def plot_mesh(self, regions=None, elements=None, plot_density=10):
        self.mesh.plot_mesh(regions=regions, elements=elements, plot_density=plot_density,
                            internal_mesh_type=('gauss', self.p))
    