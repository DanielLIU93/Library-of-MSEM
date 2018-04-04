# -*- coding: utf-8 -*-
"""
Store some useful dictionaries

>>>DOCTEST COMMANDS
(THE TEST ANSWER)

@author: Yi Zhang. Created on Tue Aug 29 17:12:39 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
# %% FORMS
from gauss_0form import Gauss0form
from gauss_tr0form import TraceGauss0form

from lobatto_1form import Lobatto1form
from lobatto_2form import Lobatto2form

from ext_gauss_0form import ExtGauss0form

forms = {   
            '1-lobatto': Lobatto1form,
            '2-lobatto': Lobatto2form,

            '0-gauss': Gauss0form,
            '0-gauss_tr': TraceGauss0form,

            '0-ext_gauss': ExtGauss0form
            }
