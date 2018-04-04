# -*- coding: utf-8 -*-
"""
Introduction

@author: Yi Zhang（张仪）. Created on Tue Jan 30 10:45:43 2018
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import numpy as np
from mesh_crazy import CrazyMesh
import time
import functionals as fn
# import scipy.io as sio

from assemble import assemble
from scipy import sparse
from forms import form
import matplotlib.pyplot as plt

plt.close('all')
plt.rcParams['image.cmap'] = 'YlGnBu'
plt.rc('text', usetex=False)
font = {'fontname': 'Times New Roman',
        'color': 'k',
        'weight': 'normal',
        'size': 14}


# define the exact solution
def p0func(x, y):
    return np.cos(3 * np.pi * x * y)


def ufunc(x, y):
    return -3 * np.pi * y * np.sin(3 * np.pi * x * y)


def vfunc(x, y):
    return -3 * np.pi * x * np.sin(3 * np.pi * x * y)


def f2func(x, y):
    return -9 * np.pi ** 2 * (x ** 2 + y ** 2) * np.cos(3 * np.pi * x * y)

# bounds_domain = ((0, 1), (0, 1))

nx = 2
ny = 2

ele_pnt_x = fn.lobatto_quad(nx)[0]
ele_pnt_y = fn.lobatto_quad(ny)[0]

ele_spa_x = [ele_pnt_x[i + 1] - ele_pnt_x[i] for i in range(0, nx)]
ele_spa_y = [ele_pnt_y[i + 1] - ele_pnt_y[i] for i in range(0, ny)]

Element_Spacing = [ele_spa_x, ele_spa_y]

n = (nx, ny)
p = (2, 2)
c = 0.
# ^^^^^^^^^^^^^^^^^^^^^^^^ standard ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("n, p, c=", n, p, c)

mesh = CrazyMesh(elements_layout=n, curvature=c, bounds_domain=((0, 2), (0, 2)), element_spacing=Element_Spacing)
num_scheme = 'general'
p0 = form(mesh, '0-gauss', p, is_inner=True, numbering_scheme=num_scheme)
u1 = form(mesh, '1-lobatto', p, is_inner=False, separated_dof=False, numbering_scheme=num_scheme)
f2 = form(mesh, '2-lobatto', p, is_inner=False, numbering_scheme=num_scheme)

f2.discretize(f2func)

M = u1.M()[1]
# sio.savemat("Mass_matrix_13_mesh.mat", mdict={'M_13_mesh': M})

E21 = u1.E21
W = f2.wedged(p0)

lhs11 = M
lhs21 = assemble(W.dot(E21), f2, u1)
lhs12 = lhs21.T

rhs1 = np.zeros(shape=u1.num_dof)
rhs2 = assemble(W, f2.dof_map, p0.dof_map).dot(f2.cochain)

lhs = sparse.bmat([[lhs11, lhs12],
                   [lhs21, None]]).tocsc()

rhs = np.concatenate((rhs1, rhs2))

# print("   compute the condition number")
# cn = np.linalg.cond(lhs.toarray())
# print("   condition number=", cn)

size_lhs = np.shape(lhs)[0]
print("solve the sparse system, lhs size:{}".format(np.shape(lhs)))
t1 = time.time()
res = sparse.linalg.spsolve(lhs, rhs)
t2 = time.time()
t = t2 - t1

u1.cochain = res[:u1.num_dof].reshape(u1.num_dof)
p0.cochain = res[u1.num_dof:u1.num_dof + p0.num_dof].reshape(p0.num_dof)

p0.plot_self(plot_type="")

p0_L2_error = p0.L2_error(p0func)[0]
# u1_L2_error = u1.L2_error((ufunc, vfunc))[0]

print('p0_L2_error =', p0_L2_error)
# print('u1_L2_error =', u1_L2_error)

u1.plot_self()
# u1.plot_mesh(plot_density=10)

# du = u1.coboundary
# du_L2_error = du.L2_error(f2func)[0]
# print('du_L2_error =', du_L2_error)
#
# u_Hdiv_error = np.sqrt(u1_L2_error ** 2 + du_L2_error ** 2)
# print('  u_Hdiv_error =', u_Hdiv_error)
#
# dumf = du - f2
# dumf_L2_error = dumf.L2_error(func=lambda x, y: 0 * x * y)[0]
# print('    du-f error =', dumf_L2_error)