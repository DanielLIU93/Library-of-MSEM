# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'bwr'
plt.rc('text', usetex=True)
font = {'fontname': 'Times New Roman',
        'color' :'k',
        'weight':'normal',
        'size'  : 14}

plt.figure()
linewidth = 1

DATA=np.load('SOVER_V1_TIME_COSTS_MEASURE_lp.npy')
Res = DATA[0]
plt.plot(Res[:,1], Res[:,4],'-o',color=(1, 0, 0, 1),linewidth=linewidth,label=r"Hybrid" )

DATA=np.load('SOVER_V2_hp_TIME_COST_MEASURE_hp.npy')
Res = DATA[0]
plt.plot(Res[:,1], Res[:,10],'-s',color=(0, 0, 1, 1),linewidth=linewidth,label=r"Direct" )
axes = plt.gca()
plt.xticks([i for i in range(2,25,2)])
axes.set_xlim([10,24])
plt.xlabel(r"$K$")
plt.ylabel(r"$t$")
plt.legend()
plt.show()