# -*- coding: utf-8 -*-
"""
extended gauss 0-form

@author: Yi Zhang [2017]
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import numpy as np
from basic_extended_form import BasicExtendedFrom

# %% The class body
class ExtGauss0form(BasicExtendedFrom):
    def __init__(self, mesh, p, numbering_scheme=None, name=None, info=None,
                 the_form=None, trace_form=None,
                 is_inner=True):
        the_form_type = "gauss_0form :: Gauss0form"
        trace_form_type = "gauss_tr0form :: TraceGauss0form"
        super().__init__(the_form_type, trace_form_type, 
                         the_form, trace_form,
                         mesh, p, numbering_scheme, name, info)
        #<-------------------------------------------------------------------->
        # Addition inputs and "assert" for the_form and trace_form------------>
        # Because for different extended form, the_form and trace_form may needs
        # different optional __init__ variables
        self.the_form._is_inner = is_inner
        self.trace_form._is_inner = is_inner
        #<-------------------------------------------------------------------->
        self._k = 0
        self._is_form = True
        self._is_inner = is_inner
        self._what_form = '0-ext_gauss'
        #<-------------------------------------------------------------------->
        #standard init done, below are special variabels for this extended form
        
        self._E10 = None

    # %% PROPERTIES

    # %% func

    # %% incidence matrix

    # %% dof map

    # %% cochain
    
    # %% cochain local

    # %% DEFAULT QUAD GRID
    
    # %%
    def plot_self(self):
        self.the_form.plot_self()

# %% Main test part
if __name__ == '__main__':
    from mesh_crazy import CrazyMesh
    p = (16, 15)
    n = (2, 3)
    c = 0.15
    domain = ((-1, 1), (-1, 1))
    mesh = CrazyMesh(elements_layout=n, curvature=c, bounds_domain=domain)

    def p0func(x, y): return np.sin(np.pi*x) * np.sin(np.pi*y)

    f0 = ExtGauss0form(mesh, p, is_inner = False)
    
    f0.discretize(p0func)
    f0.plot_self()

    print('L2_error=',f0.L2_error()[0])



