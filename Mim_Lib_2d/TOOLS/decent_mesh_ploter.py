# -*- coding: utf-8 -*-
"""
Some scripts to plot the meshes for high quality purposes

@author: Yi Zhang. Created on Tue Sep 19 18:55:24 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import meshes_chooser

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
font = {'fontname': 'Times New Roman',
        'color' :'k',
        'weight':'normal',
        'size'  : 14}

# %% 
grids_dict = {'1': 'dual_Gauss_GaussLobatto',
              '2': 'dual_Gauss_GaussLobatto_multiple_elements'}
grid_No = 2

# %% dual grids, GL and G
if grids_dict[str(grid_No)] == 'dual_Gauss_GaussLobatto':
    p = (3,3)
    n = (1,1)
    c = 0.3
    bounds_domain=((0,1),(0,1))
    mesh = meshes_chooser.mesh_No(0, elements_layout=n, c=c, bounds_domain=bounds_domain)
    xg,yg = mesh.plot_mesh(plot_density=100, internal_mesh_type = ('gauss', p),return_mesh_data=True, show_plot=False)
    xgl,ygl = mesh.plot_mesh(plot_density=100, internal_mesh_type = ('lobatto', p),return_mesh_data=True, show_plot=False)
    
    fig, ax = plt.subplots()
    linewidth = 1.8
    for i in range(n[0]*n[1]):
        plt.plot(xg[:, 4:, i], yg[:, 4:, i], '--k',color = '0.5', linewidth=linewidth*0.5) # internal meshes   
        
        plt.plot(xgl[:, 4:, i], ygl[:, 4:, i], 'k',color = '0.3', linewidth=linewidth*0.75) # internal meshes   
        plt.plot(xgl[:, :4, i], ygl[:, :4, i], 'k', linewidth=linewidth) # ELEMENTS EDGES
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$\eta$')
    plt.axis('equal')
    plt.axis([0, 1, 0, 1])
    plt.show()
    
# %%
elif grids_dict[str(grid_No)] == 'dual_Gauss_GaussLobatto_multiple_elements':
    p = (2,2)
    n = (3,3)
    c = 0.3
    bounds_domain=((0,1),(0,1))
    mesh = meshes_chooser.mesh_No(0, elements_layout=n, c=c, bounds_domain=bounds_domain)
    xg,yg = mesh.plot_mesh(plot_density=100, internal_mesh_type = ('gauss', p),return_mesh_data=True, show_plot=False)
    xgl,ygl = mesh.plot_mesh(plot_density=100, internal_mesh_type = ('lobatto', p),return_mesh_data=True, show_plot=False)
    
    fig, ax = plt.subplots()
    linewidth = 1.8
    for i in range(n[0]*n[1]):
        plt.plot(xg[:, 4:, i], yg[:, 4:, i], '--k',color = '0.5', linewidth=linewidth*0.5) # internal meshes   
        
        plt.plot(xgl[:, 4:, i], ygl[:, 4:, i], 'k',color = '0.3', linewidth=linewidth*0.75) # internal meshes   
        plt.plot(xgl[:, :4, i], ygl[:, :4, i], 'k', linewidth=linewidth) # ELEMENTS EDGES
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.axis('equal')
    plt.show()