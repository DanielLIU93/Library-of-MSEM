# -*- coding: utf-8 -*-
"""
(SHORT NAME EXPLANATION)
    
>>>DOCTEST COMMANDS 
(THE TEST ANSWER)

@author: Yi Zhang. Created on Thu Sep 21 19:50:41 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
         
#SUMMARY----------------

#INPUTS-----------------
    #ESSENTIAL:
    #OPTIONAL:

#OUTPUTS----------------

#EXAMPLES---------------

#NOTES------------------
"""
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'bwr'
plt.rc('text', usetex=True)
font = {'fontname': 'Times New Roman',
        'color' :'k',
        'weight':'normal',
        'size'  : 14}

import scipy.io

# %% L2-error phi
#MAT = scipy.io.loadmat("Res_170921_hc_p1.mat")
#Res = MAT['Res']
#plt.figure()
#linewidth = 1
#plt.loglog(1/Res[:8,1] , Res[:8,3]   , '-o', color=(1, 0, 0, 1) , linewidth=linewidth, 
#           label=r"$N=1,\ c=0$")
#plt.loglog(1/Res[8:16,1], Res[8:16,3], '-^', color=(1, 0, 0, 0.7), linewidth=linewidth, 
#           label=r"$N=1,\ c=0.15$")
#plt.loglog(1/Res[16:,1] , Res[16:,3] , '-v', color=(1, 0, 0, 0.4), linewidth=linewidth, 
#           label=r"$N=1,\ c=0.3$")
#
#MAT = scipy.io.loadmat("Res_170921_hc_p2.mat")
#Res = MAT['Res']
#linewidth = 1
#plt.loglog(1/Res[:8,1], Res[:8,3]    , '-o', color=(0, 1, 0, 1), linewidth=linewidth, 
#           label=r"$N=2,\ c=0$")
#plt.loglog(1/Res[8:16,1], Res[8:16,3], '-^', color=(0, 1, 0, 0.7), linewidth=linewidth, 
#           label=r"$N=2,\ c=0.15$")
#plt.loglog(1/Res[16:,1], Res[16:,3]  , '-v', color=(0, 1, 0, 0.4), linewidth=linewidth, 
#           label=r"$N=2,\ c=0.3$")
#
#MAT = scipy.io.loadmat("Res_170921_hc_p3.mat")
#Res = MAT['Res']
#linewidth = 1
#plt.loglog(1/Res[:8,1], Res[:8,3]    , '-o', color=(0, 0, 1, 1  ), linewidth=linewidth, 
#           label=r"$N=3,\ c=0$")
#plt.loglog(1/Res[8:16,1], Res[8:16,3], '-^', color=(0, 0, 1, 0.7), linewidth=linewidth, 
#           label=r"$N=3,\ c=0.15$")
#plt.loglog(1/Res[16:,1], Res[16:,3]  , '-v', color=(0, 0, 1, 0.4), linewidth=linewidth, 
#           label=r"$N=3,\ c=0.3$")
#plt.legend()
#plt.xlabel(r"$h$")
#plt.ylabel(r"$\left\|\tilde{\phi}_h^{(0)}\right\|_{L^2-\mathrm{error}}$")
#plt.show()

# %% H-error phi
MAT = scipy.io.loadmat("Res_170921_hc_p1.mat")
Res = MAT['Res']

plt.figure()
linewidth = 1
plt.loglog(1/Res[:8,1] , Res[:8,5]   , '-o', color=(1, 0, 0, 1) , linewidth=linewidth, 
           label=r"$N=1,\ c=0$")
plt.loglog(1/Res[8:16,1], Res[8:16,5], '-^', color=(1, 0, 0, 0.7), linewidth=linewidth, 
           label=r"$N=1,\ c=0.15$")
plt.loglog(1/Res[16:,1] , Res[16:,5] , '-v', color=(1, 0, 0, 0.4), linewidth=linewidth, 
           label=r"$N=1,\ c=0.3$")

MAT = scipy.io.loadmat("Res_170921_hc_p2.mat")
Res = MAT['Res']
linewidth = 1
plt.loglog(1/Res[:8,1], Res[:8,5]    , '-o', color=(0, 1, 0, 1), linewidth=linewidth, 
           label=r"$N=2,\ c=0$")
plt.loglog(1/Res[8:16,1], Res[8:16,5], '-^', color=(0, 1, 0, 0.7), linewidth=linewidth, 
           label=r"$N=2,\ c=0.15$")
plt.loglog(1/Res[16:,1], Res[16:,5]  , '-v', color=(0, 1, 0, 0.4), linewidth=linewidth, 
           label=r"$N=2,\ c=0.3$")

MAT = scipy.io.loadmat("Res_170921_hc_p3.mat")
Res = MAT['Res']
linewidth = 1
plt.loglog(1/Res[:8,1], Res[:8,5]    , '-o', color=(0, 0, 1, 1  ), linewidth=linewidth, 
           label=r"$N=3,\ c=0$")
plt.loglog(1/Res[8:16,1], Res[8:16,5], '-^', color=(0, 0, 1, 0.7), linewidth=linewidth, 
           label=r"$N=3,\ c=0.15$")
plt.loglog(1/Res[16:,1], Res[16:,5]  , '-v', color=(0, 0, 1, 0.4), linewidth=linewidth, 
           label=r"$N=3,\ c=0.3$")
plt.legend()
plt.xlabel(r"$h$")
plt.ylabel(r"$\left\|\tilde{\phi}_h^{(0)}\right\|_{H-\mathrm{error}}$")
plt.show()

# %% H-error u
#MAT = scipy.io.loadmat("Res_170921_hc_p1.mat")
#Res = MAT['Res']
#plt.figure()
#linewidth = 1
#plt.loglog(1/Res[:8,1] , Res[:8,8]   , '-o', color=(1, 0, 0, 1) , linewidth=linewidth, 
#           label=r"$N=1,\ c=0$")
#plt.loglog(1/Res[8:16,1], Res[8:16,8], '-^', color=(1, 0, 0, 0.7), linewidth=linewidth, 
#           label=r"$N=1,\ c=0.15$")
#plt.loglog(1/Res[16:,1] , Res[16:,8] , '-v', color=(1, 0, 0, 0.4), linewidth=linewidth, 
#           label=r"$N=1,\ c=0.3$")
#
#MAT = scipy.io.loadmat("Res_170921_hc_p2.mat")
#Res = MAT['Res']
#linewidth = 1
#plt.loglog(1/Res[:8,1], Res[:8,8]    , '-o', color=(0, 1, 0, 1), linewidth=linewidth, 
#           label=r"$N=2,\ c=0$")
#plt.loglog(1/Res[8:16,1], Res[8:16,8], '-^', color=(0, 1, 0, 0.7), linewidth=linewidth, 
#           label=r"$N=2,\ c=0.15$")
#plt.loglog(1/Res[16:,1], Res[16:,8]  , '-v', color=(0, 1, 0, 0.4), linewidth=linewidth, 
#           label=r"$N=2,\ c=0.3$")
#
#MAT = scipy.io.loadmat("Res_170921_hc_p3.mat")
#Res = MAT['Res']
#linewidth = 1
#plt.loglog(1/Res[:8,1], Res[:8,8]    , '-o', color=(0, 0, 1, 1  ), linewidth=linewidth, 
#           label=r"$N=3,\ c=0$")
#plt.loglog(1/Res[8:16,1], Res[8:16,8], '-^', color=(0, 0, 1, 0.7), linewidth=linewidth, 
#           label=r"$N=3,\ c=0.15$")
#plt.loglog(1/Res[16:,1], Res[16:,8]  , '-v', color=(0, 0, 1, 0.4), linewidth=linewidth, 
#           label=r"$N=3,\ c=0.3$")
#plt.legend()
#plt.xlabel(r"$h$")
#plt.ylabel(r"$\left\|u_h^{(n-1)}\right\|_{H-\mathrm{error}}$")
#plt.show()

# %% L2-error u
#MAT = scipy.io.loadmat("Res_170921_hc_p1.mat")
#Res = MAT['Res']
#plt.figure()
#linewidth = 1
#plt.loglog(1/Res[:8,1] , Res[:8,6]   , '-o', color=(1, 0, 0, 1) , linewidth=linewidth, 
#           label=r"$N=1,\ c=0$")
#plt.loglog(1/Res[8:16,1], Res[8:16,6], '-^', color=(1, 0, 0, 0.7), linewidth=linewidth, 
#           label=r"$N=1,\ c=0.15$")
#plt.loglog(1/Res[16:,1] , Res[16:,6] , '-v', color=(1, 0, 0, 0.4), linewidth=linewidth, 
#           label=r"$N=1,\ c=0.3$")
#
#MAT = scipy.io.loadmat("Res_170921_hc_p2.mat")
#Res = MAT['Res']
#linewidth = 1
#plt.loglog(1/Res[:8,1], Res[:8,6]    , '-o', color=(0, 1, 0, 1), linewidth=linewidth, 
#           label=r"$N=2,\ c=0$")
#plt.loglog(1/Res[8:16,1], Res[8:16,6], '-^', color=(0, 1, 0, 0.7), linewidth=linewidth, 
#           label=r"$N=2,\ c=0.15$")
#plt.loglog(1/Res[16:,1], Res[16:,6]  , '-v', color=(0, 1, 0, 0.4), linewidth=linewidth, 
#           label=r"$N=2,\ c=0.3$")
#
#MAT = scipy.io.loadmat("Res_170921_hc_p3.mat")
#Res = MAT['Res']
#linewidth = 1
#plt.loglog(1/Res[:8,1], Res[:8,6]    , '-o', color=(0, 0, 1, 1  ), linewidth=linewidth, 
#           label=r"$N=3,\ c=0$")
#plt.loglog(1/Res[8:16,1], Res[8:16,6], '-^', color=(0, 0, 1, 0.7), linewidth=linewidth, 
#           label=r"$N=3,\ c=0.15$")
#plt.loglog(1/Res[16:,1], Res[16:,6]  , '-v', color=(0, 0, 1, 0.4), linewidth=linewidth, 
#           label=r"$N=3,\ c=0.3$")
#plt.legend()
#plt.xlabel(r"$h$")
#plt.ylabel(r"$\left\|u_h^{(n-1)}\right\|_{L^2-\mathrm{error}}$")
#plt.show()

# %% t
MAT = scipy.io.loadmat("Res_170921_hc_p1.mat")
Res = MAT['Res']
plt.figure()
linewidth = 1
plt.loglog(1/Res[:8,1] , Res[:8,-2]   , '-o', color=(1, 0, 0, 1) , linewidth=linewidth, 
           label=r"$N=1,\ c=0$")
plt.loglog(1/Res[8:16,1], Res[8:16,-2], '-^', color=(1, 0, 0, 0.7), linewidth=linewidth, 
           label=r"$N=1,\ c=0.15$")
plt.loglog(1/Res[16:,1] , Res[16:,-2] , '-v', color=(1, 0, 0, 0.4), linewidth=linewidth, 
           label=r"$N=1,\ c=0.3$")

MAT = scipy.io.loadmat("Res_170921_hc_p2.mat")
Res = MAT['Res']
linewidth = 1
plt.loglog(1/Res[:8,1], Res[:8,-2]    , '-o', color=(0, 1, 0, 1), linewidth=linewidth, 
           label=r"$N=2,\ c=0$")
plt.loglog(1/Res[8:16,1], Res[8:16,-2], '-^', color=(0, 1, 0, 0.7), linewidth=linewidth, 
           label=r"$N=2,\ c=0.15$")
plt.loglog(1/Res[16:,1], Res[16:,-2]  , '-v', color=(0, 1, 0, 0.4), linewidth=linewidth, 
           label=r"$N=2,\ c=0.3$")

MAT = scipy.io.loadmat("Res_170921_hc_p3.mat")
Res = MAT['Res']
linewidth = 1
plt.loglog(1/Res[:8,1], Res[:8,-2]    , '-o', color=(0, 0, 1, 1  ), linewidth=linewidth, 
           label=r"$N=3,\ c=0$")
plt.loglog(1/Res[8:16,1], Res[8:16,-2], '-^', color=(0, 0, 1, 0.7), linewidth=linewidth, 
           label=r"$N=3,\ c=0.15$")
plt.loglog(1/Res[16:,1], Res[16:,-2]  , '-v', color=(0, 0, 1, 0.4), linewidth=linewidth, 
           label=r"$N=3,\ c=0.3$")
plt.legend()
plt.xlabel(r"$h$")
plt.ylabel(r"$\left\|\mathrm{d}u_h^{(n-1)}-f_h^{(n)}\right\|_{L^2-\mathrm{error}}$")
plt.show()