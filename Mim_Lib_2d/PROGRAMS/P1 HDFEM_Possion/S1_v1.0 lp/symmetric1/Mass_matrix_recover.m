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

M_5 = full(M_5);
M_9 = full(M_9);
M_13 = full(M_13);
M_25 = full(M_25);

B_5 = (WedgeMat_5'*E21_5);
B_9 = (WedgeMat_9'*E21_9);
B_13 = (WedgeMat_13'*E21_13);
B_25 = (WedgeMat_25'*E21_25);


LHS_5 = [M_5, B_5'; B_5, zeros(size(WedgeMat_5))];
LHS_9 = [M_9, B_9'; B_9, zeros(size(WedgeMat_9))];
LHS_13 = [M_13, B_13'; B_13, zeros(size(WedgeMat_13))];
LHS_25 = [M_25, B_25'; B_25, zeros(size(WedgeMat_25))];


S_5 = B_5*(M_5)^(-1)*B_5';
S_9 = B_9*(M_9)^(-1)*B_9';
S_13 = B_13*(M_13)^(-1)*B_13';
S_25 = B_25*(M_25)^(-1)*B_25';

%% 
cond_M = [cond(M_5), cond(M_9), cond(M_13), cond(M_25)];
cond_S = [cond(S_5), cond(S_9), cond(S_13), cond(S_25)];
cond_W = [cond(WedgeMat_5), cond(WedgeMat_9), cond(WedgeMat_13), cond(WedgeMat_25)];
cond_LHS = [cond(LHS_5), cond(LHS_9), cond(LHS_13), cond(LHS_25)];
cond_map = [cond(E21_5*E21_5'), cond(E21_9*E21_9'), cond(E21_13*E21_13'), cond(E21_25*E21_25')];

% cond_MB_5 = cond([eye(size(M_5)),inv(M_5)*B_5' ; zeros(size(B_5)), eye(size(WedgeMat_5))])
% cond_MB_9 = cond([eye(size(M_9)),inv(M_9)*B_9' ; zeros(size(B_9)), eye(size(WedgeMat_9))])
% cond_MB_13 = cond([eye(size(M_13)),inv(M_13)*B_13' ; zeros(size(B_13)), eye(size(WedgeMat_13))])
% cond_MB_25 = cond([eye(size(M_25)),inv(M_25)*B_25' ; zeros(size(B_25)), eye(size(WedgeMat_25))])
% cond_MB = [cond_MB_5, cond_MB_9, cond_MB_13, cond_MB_25]

cond_BD = [cond(blkdiag(14*M_5,S_5)), cond(blkdiag(9*M_9,S_9)), ...
    cond(blkdiag(7*M_13,S_13)), cond(blkdiag(6*M_25,S_25))];

%% construct preconditioner


%%
figure(1)
ax_x = [5,9,13,25];
hold on
plot(ax_x, cond_M,'b*-','LineWidth',1.5)
plot(ax_x, cond_S,'mo-','LineWidth',1.5)
plot(ax_x, cond_LHS,'r^-','LineWidth',1.5)
plot(ax_x, cond_map,'k>-','LineWidth',1.5)
plot(ax_x, cond_W,'c<-','LineWidth',1.5)
legend('\kappa(M)','\kappa(S)','\kappa(LHS)','\kappa(map)','\kappa(W)','location','northwest')


