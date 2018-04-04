# -*- coding: utf-8 -*-
"""
A wrap of H
    
>>>DOCTEST COMMANDS 
(THE TEST ANSWER)

@author: Yi Zhang. Created on Tue Aug 29 17:02:33 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import meshes_chooser
import numpy as np
from forms import form
from dictionaries import Hodge_pairs

dict_Hodge_pairs = Hodge_pairs

# %%
def H(to_form, from_form):
    """
    #SUMMARY: The Hodge operator, H(f1,f2) means f1= H(f2)
    # INPUTS:
        ESSENTAIL:
            [0] to_form:
            [1] from_form:
    #OUTPUTS: 
        No outputs since both inputs are class instance, changes are done locally
    """
    if to_form.what_form not in dict_Hodge_pairs:
        raise Exception("Error. For H(f, g): f=H(g); form f: ({}) can not be obtained from Hodge of any other form g".format(to_form.what_form))
    assert from_form.what_form in dict_Hodge_pairs[to_form.what_form], \
        'Error. H(f, g): f=H(g). Can not do Hodge from g: ('+ from_form.what_form+') to f: ('+to_form.what_form + ')'
    H, H_a = to_form.H(from_form)
    return H, H_a

# %% MAIN
if __name__ == '__main__':
    p = (6,6)
    n = (3,3)
    mesh = meshes_chooser.mesh_No(1, elements_layout=n)
    def func(x, y): return -8 * np.pi**2 * np.sin(2*np.pi * x) * np.sin(2*np.pi * y)
    
    f0 = form(mesh, '0-lobatto', p, is_inner=True)
    f2 = form(mesh, '2-ext_gauss', p, is_inner=False)
    
    f0.discretize(func)
    H(f2, f0)
    
    f2.plot_self()