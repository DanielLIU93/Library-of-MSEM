clc;
clear;
%%
load('mass_matrix_5.mat');
load('mass_matrix_9.mat');
load('mass_matrix_13.mat');
load('mass_matrix_25.mat');

load('Incidence_matrix_5');
load('Incidence_matrix_9');
load('Incidence_matrix_13');
load('Incidence_matrix_25');

load('B_Wedge_5');
load('B_Wedge_9');
load('B_Wedge_13');
load('B_Wedge_25');

load('Wedge_matrix_5');
load('Wedge_matrix_9');
load('Wedge_matrix_13');
load('Wedge_matrix_25');

M_5_p = full(M_5_p);
M_9_p = full(M_9_p);
M_13_p = full(M_13_p);
M_25_p = full(M_25_p);

B_5_p = (WedgeMat_5_p*E21_5_p);
B_9_p = (WedgeMat_9_p*E21_9_p);
B_13_p = (WedgeMat_13_p*E21_13_p);
B_25_p = (WedgeMat_25_p*E21_25_p);


LHS_5_p = [M_5_p, B_5_p'; B_5_p, zeros(size(WedgeMat_5_p))];
LHS_9_p = [M_9_p, B_9_p'; B_9_p, zeros(size(WedgeMat_9_p))];
LHS_13_p = [M_13_p, B_13_p'; B_13_p, zeros(size(WedgeMat_13_p))];
LHS_25_p = [M_25_p, B_25_p'; B_25_p, zeros(size(WedgeMat_25_p))];


S_5_p = B_5_p*(M_5_p)^(-1)*B_5_p';
S_9_p = B_9_p*(M_9_p)^(-1)*B_9_p';
S_13_p = B_13_p*(M_13_p)^(-1)*B_13_p';
S_25_p = B_25_p*(M_25_p)^(-1)*B_25_p';

%% 
cond_M_p = [cond(M_5_p), cond(M_9_p), cond(M_13_p), cond(M_25_p)];
cond_S_p = [cond(S_5_p), cond(S_9_p), cond(S_13_p), cond(S_25_p)];
cond_W_p = [cond(WedgeMat_5_p), cond(WedgeMat_9_p), cond(WedgeMat_13_p), cond(WedgeMat_25_p)];
cond_LHS_p = [cond(LHS_5_p), cond(LHS_9_p), cond(LHS_13_p), cond(LHS_25_p)];
cond_map_p = [cond(E21_5_p*E21_5_p'), cond(E21_9_p*E21_9_p'), cond(E21_13_p*E21_13_p'), cond(E21_25_p*E21_25_p')];

% cond_MB_5 = cond([eye(size(M_5)),inv(M_5)*B_5' ; zeros(size(B_5)), eye(size(WedgeMat_5))])
% cond_MB_9 = cond([eye(size(M_9)),inv(M_9)*B_9' ; zeros(size(B_9)), eye(size(WedgeMat_9))])
% cond_MB_13 = cond([eye(size(M_13)),inv(M_13)*B_13' ; zeros(size(B_13)), eye(size(WedgeMat_13))])
% cond_MB_25 = cond([eye(size(M_25)),inv(M_25)*B_25' ; zeros(size(B_25)), eye(size(WedgeMat_25))])
% cond_MB = [cond_MB_5, cond_MB_9, cond_MB_13, cond_MB_25]

cond_BD_p = [cond(blkdiag(14*M_5_p,S_5_p)), cond(blkdiag(9*M_9_p,S_9_p)), ...
    cond(blkdiag(7*M_13_p,S_13_p)), cond(blkdiag(6*M_25_p,S_25_p))];

%% construct preconditioner


%%
figure(1)
ax_x = [5,9,13,25];
hold on
plot(ax_x, cond_M_p,'b*-','LineWidth',1.5)
plot(ax_x, cond_S_p,'mo-','LineWidth',1.5)
plot(ax_x, cond_LHS_p,'r^-','LineWidth',1.5)
plot(ax_x, cond_map_p,'k>-','LineWidth',1.5)
plot(ax_x, cond_W_p,'c<-','LineWidth',1.5)
legend('\kappa(M)','\kappa(S)','\kappa(LHS)','\kappa(map)','\kappa(W)','location','northwest')


