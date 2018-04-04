
%% 
R = 'S'; % Other possible shapes include S,N,C,D,A,H,B
% Generate and display the grid.
n = 7;
G = numgrid(R,n);
spy(G)
title('A finite difference grid')

%% Show a smaller version as sample.
g = numgrid(R,5)

D = delsq(G);
spy(D);
title('The 5-point Laplacian')