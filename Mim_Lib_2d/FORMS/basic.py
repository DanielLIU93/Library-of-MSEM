# -*- coding: utf-8 -*-
"""
The basic for extended basic form and basic form

@author: Yi Zhang （张仪）, Created on Tue Nov 14 20:12:24 2017
    Aerodynamics
    Faculty of Aerospace Engineering
    TU Delft
    Delft, Netherlands
"""

from abc import ABC
import numpy as np

# %%
class Basic(ABC):  
    # %% INIT 
    def __init__(self, mesh, p, numbering_scheme, name, info):
        assert mesh.__class__.__name__ in ('CrazyMesh', 'BilinearMesh', 
                                           'TransfiniteMesh', 'GildingMesh', 
                                           'MixedMesh')
        self._mesh = mesh
        self._name = name
        self._info = info
        self._numbering_scheme = numbering_scheme
        
        if isinstance(p, float):
            print(" <FORM> : WARNING, p (={}) is a float, first convert it into int".format(p))
            p = int(p)
        if isinstance(p, int):
            assert p >= 1, ' <FORM> : p has to be positive, now p = {}'.format(p)
            self._p = (p, p)
        elif isinstance(p, tuple) or isinstance(p, list) or p.__class__.__name__ == 'ndarray':
            assert np.shape(p) == (2,), " <FORM> : The shape of p (p={}) is wrong".format(p)
            self._p = (int(p[0]), int(p[1]))
        else:
            raise Exception(" <FORM> : I do not accept the p(={}) you feed here".format(p))
        
        #^^^^^^^^^change these when create a new extended form^^^^^^^^^^^^^^^^^
        # None means make no sense for this kind of form ----------------------
        self._is_inner = None
        self._separated_dof = None
        # ---------------------------------------------------------------------
        # in a real form, one and only one of following should be True---------
        self._k = None
        self._is_form = False
        self._is_coform = False
        self._is_trace_form = False
        self._is_cotrace_form = False
        self._is_extended_form = False
        self._what_form = None
        #VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
        
    # %% PROPERTIES 
    @property
    def mesh(self):
        return self._mesh

    @property
    def dim(self):
        return self.mesh.dim

    @property
    def name(self):
        return self._name

    @property
    def info(self):
        return self._info   
    
    @property
    def numbering_scheme(self):
        return self._numbering_scheme
    
    @property
    def p(self):
        return self._p
    
    @property
    def is_inner(self):
        return self._is_inner

    @property
    def separated_dof(self):
        return self._separated_dof

    @property
    def orientation(self):
        if self.is_inner is True:
            return "inner-oriented"
        elif self.is_inner is False:
            return "outer-oriented"
        else:
            return None

    @property
    def k(self):
        return self._k
    
    @property
    def is_form(self):
        return self._is_form

    @property
    def is_coform(self):
        return self._is_coform

    @property
    def is_trace_form(self):
        return self._is_trace_form
    
    @property
    def is_cotrace_form(self):
        return self._is_cotrace_form
    
    @property
    def is_extended_form(self):
        return self._is_extended_form

    @property
    def what_form(self):
        return self._what_form

    @property
    def num_basis(self):
        return self._num_basis
    
    @property
    def form(self):
        return self._form
    