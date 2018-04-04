# -*- coding: utf-8 -*-
"""
Wrap of coboundary of each form

@author: Yi Zhang. Created on Tue Aug 29 12:56:51 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""

# %%
def d(f, which=None):
    """
    #SUMMARY: The coboundary operator, f2 = d(f1)
    # INPUTS:
        ESSENTAIL:
            [0] f:
    #OUTPUTS: 
        [0] E :: local incidence matrix
        [1] E_a :: assembled incidence matrix; global incidence matrix
        [2] df :: the k+1 form df
    """
    # %%
    if f.__class__.__name__ in ('Lobatto0form', 'Lobatto1form', 
                                'ExtGauss0form', 'ExtGauss1form'):
        return f.coboundary
    
    # %%
    else:
        raise Exception("can not do'd' for form: "+f.what_form)
