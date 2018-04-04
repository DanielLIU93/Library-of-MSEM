# -*- coding: utf-8 -*-
"""
(SHORT NAME EXPLANATION)
    
>>>DOCTEST COMMANDS 
(THE TEST ANSWER)

@author: Yi Zhang. Created on Thu Sep 21 21:17:53 2017
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
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'bwr'
plt.rc('text', usetex=True)
font = {'fontname': 'Times New Roman',
        'color' :'k',
        'weight':'normal',
        'size'  : 14}
# %% L2-phi
plt.figure()
linewidth = 1

DATA=np.load('Res_170921_pc_N1.npy')
Res=DATA[0]
plt.semilogy(Res[:12,0] , Res[:12,3]   , '-o', color=(1, 0, 0, 1) , linewidth=linewidth, 
           label=r"$K=1,\ c=0$")
plt.semilogy(Res[12:24,0], Res[12:24,3], '-^', color=(1, 0, 0, 0.7), linewidth=linewidth, 
           label=r"$K=1,\ c=0.15$")
plt.semilogy(Res[24:,0] , Res[24:,3] , '-v', color=(1, 0, 0, 0.4), linewidth=linewidth, 
           label=r"$K=1,\ c=0.3$")


DATA=np.load('Res_170921_pc_N3.npy')
Res=DATA[0]
plt.semilogy(Res[:12,0] , Res[:12,3]   , '-o', color=(0, 0, 1, 1) , linewidth=linewidth, 
           label=r"$K=3,\ c=0$")
plt.semilogy(Res[12:24,0], Res[12:24,3], '-^', color=(0, 0, 1, 0.7), linewidth=linewidth, 
           label=r"$K=3,\ c=0.15$")
plt.semilogy(Res[24:,0] , Res[24:,3] , '-v', color=(0, 0, 1, 0.4), linewidth=linewidth, 
           label=r"$K=3,\ c=0.3$")

plt.legend()
plt.xlabel(r"$N$")
plt.ylabel(r"$\left\|\tilde{\phi}_h^{(0)}\right\|_{L^2-\mathrm{error}}$")
plt.show()

# %% H-phi
plt.figure()
linewidth = 1

DATA=np.load('Res_170921_pc_N1.npy')
Res=DATA[0]
plt.semilogy(Res[:12,0] , Res[:12,5]   , '-o', color=(1, 0, 0, 1) , linewidth=linewidth, 
           label=r"$K=1,\ c=0$")
plt.semilogy(Res[12:24,0], Res[12:24,5], '-^', color=(1, 0, 0, 0.7), linewidth=linewidth, 
           label=r"$K=1,\ c=0.15$")
plt.semilogy(Res[24:,0] , Res[24:,5] , '-v', color=(1, 0, 0, 0.4), linewidth=linewidth, 
           label=r"$K=1,\ c=0.3$")


DATA=np.load('Res_170921_pc_N3.npy')
Res=DATA[0]
plt.semilogy(Res[:12,0] , Res[:12,5]   , '-o', color=(0, 0, 1, 1) , linewidth=linewidth, 
           label=r"$K=3,\ c=0$")
plt.semilogy(Res[12:24,0], Res[12:24,5], '-^', color=(0, 0, 1, 0.7), linewidth=linewidth, 
           label=r"$K=3,\ c=0.15$")
plt.semilogy(Res[24:,0] , Res[24:,5] , '-v', color=(0, 0, 1, 0.4), linewidth=linewidth, 
           label=r"$K=3,\ c=0.3$")

plt.legend()
plt.xlabel(r"$N$")
plt.ylabel(r"$\left\|\tilde{\phi}_h^{(0)}\right\|_{H-\mathrm{error}}$")
plt.show()

# %% H-u
plt.figure()
linewidth = 1

DATA=np.load('Res_170921_pc_N1.npy')
Res=DATA[0]
plt.semilogy(Res[:12,0] , Res[:12,8]   , '-o', color=(1, 0, 0, 1) , linewidth=linewidth, 
           label=r"$K=1,\ c=0$")
plt.semilogy(Res[12:24,0], Res[12:24,8], '-^', color=(1, 0, 0, 0.7), linewidth=linewidth, 
           label=r"$K=1,\ c=0.15$")
plt.semilogy(Res[24:,0] , Res[24:,8] , '-v', color=(1, 0, 0, 0.4), linewidth=linewidth, 
           label=r"$K=1,\ c=0.3$")


DATA=np.load('Res_170921_pc_N3.npy')
Res=DATA[0]
plt.semilogy(Res[:12,0] , Res[:12,8]   , '-o', color=(0, 0, 1, 1) , linewidth=linewidth, 
           label=r"$K=3,\ c=0$")
plt.semilogy(Res[12:24,0], Res[12:24,8], '-^', color=(0, 0, 1, 0.7), linewidth=linewidth, 
           label=r"$K=3,\ c=0.15$")
plt.semilogy(Res[24:,0] , Res[24:,8] , '-v', color=(0, 0, 1, 0.4), linewidth=linewidth, 
           label=r"$K=3,\ c=0.3$")

plt.legend()
plt.xlabel(r"$N$")
plt.ylabel(r"$\left\|u_h^{(n-1)}\right\|_{H-\mathrm{error}}$")
plt.show()

# %% L2-du-f
plt.figure()
linewidth = 1

DATA=np.load('Res_170921_pc_N1.npy')
Res=DATA[0]
plt.semilogy(Res[:12,0] , Res[:12,9]   , '-o', color=(1, 0, 0, 1) , linewidth=linewidth, 
           label=r"$K=1,\ c=0$")
plt.semilogy(Res[12:24,0], Res[12:24,9], '-^', color=(1, 0, 0, 0.7), linewidth=linewidth, 
           label=r"$K=1,\ c=0.15$")
plt.semilogy(Res[24:,0] , Res[24:,9] , '-v', color=(1, 0, 0, 0.4), linewidth=linewidth, 
           label=r"$K=1,\ c=0.3$")


DATA=np.load('Res_170921_pc_N3.npy')
Res=DATA[0]
plt.semilogy(Res[:12,0] , Res[:12,9]   , '-o', color=(0, 0, 1, 1) , linewidth=linewidth, 
           label=r"$K=3,\ c=0$")
plt.semilogy(Res[12:24,0], Res[12:24,9], '-^', color=(0, 0, 1, 0.7), linewidth=linewidth, 
           label=r"$K=3,\ c=0.15$")
plt.semilogy(Res[24:,0] , Res[24:,9] , '-v', color=(0, 0, 1, 0.4), linewidth=linewidth, 
           label=r"$K=3,\ c=0.3$")

plt.legend()
plt.xlabel(r"$N$")
plt.ylabel(r"$\left\|\mathrm{d}u_h^{(n-1)}-f_h^{(n)}\right\|_{L^2-\mathrm{error}}$")
plt.show()