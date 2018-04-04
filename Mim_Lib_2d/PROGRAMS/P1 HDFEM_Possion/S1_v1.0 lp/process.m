clear 
clc

%% 
load orthobasis

load B_Wedge_5.mat 
load f2_5.mat
load Incidence_matrix_5
load Mass_matrix_5
load q0_5
load Wedge_matrix_5

load B_Wedge_9.mat 
load f2_9.mat
load Incidence_matrix_9
load Mass_matrix_9
load q0_9
load Wedge_matrix_9

load B_Wedge_13.mat 
load f2_13.mat
load Incidence_matrix_13
load Mass_matrix_13
load q0_13
load Wedge_matrix_13

load B_Wedge_25.mat 
load f2_25.mat
load Incidence_matrix_25
load Mass_matrix_25
load q0_25
load Wedge_matrix_25


%% Preconditioning of Schur Complement
map_5 = E21_5*E21_5';
map_9 = E21_9*E21_9';
map_13 = E21_13*E21_13';
map_25 = E21_25*E21_25';

% Remove Laplacian

[n_5,m_5] = size(E21_5);

S_5 = WedgeMat_5'*E21_5*(M_5)^(-1)*E21_5'*WedgeMat_5;

cond_S_5 = cond(S_5);

Pre_S_5 = inv(map_5);

cond_pre_5 = cond(Pre_S_5*S_5);
%
[n_9,m_9] = size(E21_9);

S_9 = WedgeMat_9'*E21_9*(M_9)^(-1)*E21_9'*WedgeMat_9;

cond_S_9 = cond(S_9);

Pre_S_9 = inv(map_9);

cond_pre_9 = cond(Pre_S_9*S_9);
%
[n_13,m_13] = size(E21_13);

S_13 = WedgeMat_13'*E21_13*(M_13)^(-1)*E21_13'*WedgeMat_13;

cond_S_13 = cond(S_13);

Pre_S_13 = inv(map_13);

cond_pre_13 = cond(Pre_S_13 * S_13);
%
[n_25,m_25] = size(E21_25);

S_25 = WedgeMat_25'*E21_25*(M_25)^(-1)*E21_25'*WedgeMat_25;

cond_S_25 = cond(S_25);

Pre_S_25 = inv(map_25);

cond_pre_25 = cond(Pre_S_25*S_25);

% Total Inverse

% rmv_S_5 = (In_F_5')^(2) *E21_5*(M_5)*E21_5'* (In_F_5)^(2);
rmv_S_5 = E21_5*(M_5_p^(-1))*E21_5';
cond_remove_5 = cond(rmv_S_5\S_5);

rmv_S_9 = E21_9*(M_9_p^(-1))*E21_9';
cond_remove_9 = cond(rmv_S_9\S_9);

rmv_S_13 = E21_13*(M_13_p^(-1))*E21_13';
cond_remove_13 = cond(rmv_S_13\S_13);

rmv_S_25 = E21_25*(M_25_p^(-1))*E21_25';
cond_remove_25 = cond(rmv_S_25\S_25);

%
% rmv_S_5 = (In_F_5')^(2) *E21_5*(M_5)*E21_5'* (In_F_5)^(2);
rmv_S_5 = E21_5*(diag(diag(M_5^(-1))))*E21_5';
cond_remodiag_5 = cond(rmv_S_5\S_5);

In_F_9 = (chol(map_9))^(-1);
rmv_S_9 = E21_9*(diag(diag(M_9^(-1))))*E21_9';
cond_remodiag_9 = cond(rmv_S_9\S_9);

In_F_13 = (chol(map_13))^(-1);
rmv_S_13 = E21_13*(diag(diag(M_13^(-1))))*E21_13';
cond_remodiag_13 = cond(rmv_S_13\S_13);

In_F_25 = (chol(map_25))^(-1);
rmv_S_25 = E21_25*(diag(diag(M_25^(-1))))*E21_25';
cond_remodiag_25 = cond(rmv_S_25\S_25);


%% Plots
scrsz = get(groot,'screensize');
figure('position',[55 125 scrsz(3)/1.05 scrsz(4)/1.7]);
ax1 = subplot(1,2,1);
ax1.FontSize = 25.0;
hold on
xlabel(ax1, 'Polynomial Degree','fontsize',25.0)
ylabel(ax1, 'Condition Number','fontsize',25.0)
plot([5,9,13,25], [cond_pre_5, cond_pre_9, cond_pre_13, cond_pre_25], 'k*-.','linewidth',1.5)
plot([5,9,13,25], [cond_remove_5,cond_remove_9,cond_remove_13,cond_remove_25], 'ro-.','linewidth',1.5)
plot([5,9,13,25], [cond_remodiag_5,cond_remodiag_9,cond_remodiag_13,cond_remodiag_25], 'b^--','linewidth',1.5)
legend('\kappa(map^{-1}\cdot S)','\kappa (G_2^{-1}S)','\kappa (G_1^{-1}S)', 'location','northwest')

ax2 = subplot(1,2,2);
hold on
ax2.FontSize = 25.0;
ylabel(ax2, 'Condition Number','fontsize',25.0)
xlabel(ax2, 'Polynomial Degree','fontsize',25.0)
% plot([5,9,13,25], [cond_pre_5, cond_pre_9, cond_pre_13, cond_pre_25], 'b*-.','linewidth',1.5)
plot([5,9,13,25], [cond_remove_5,cond_remove_9,cond_remove_13,cond_remove_25], 'ro-.','linewidth',1.5)
plot([5,9,13,25], [cond_remodiag_5,cond_remodiag_9,cond_remodiag_13,cond_remodiag_25], 'b^--','linewidth',1.5)
% legend(%'\kappa (S_0^{-1}S), Remove map'
       legend('\kappa (G_2^{-1}S)',...
       '\kappa (G_1^{-1}S)', 'location','northwest')


%% Substructure

% E1_5 = E21_5(:,1:30);
% M_u_5=M_5(1:30,1:30);
% S_u_5 = WedgeMat_5'*E1_5*M_u_5^(-1)*E1_5'*WedgeMat_5;
% cond(S_u_5)
% 
% E2_5 = E21_5(:,31:end);
% M_v_5=M_5(31:end,31:end);
% S_v_5 = WedgeMat_5'*E2_5*M_v_5^(-1)*E2_5'*WedgeMat_5;
% cond(S_v_5)
% 
% cond(S_u_5+S_v_5)
% 

%%
% figure(1)
% ax_x = [5,9,13,25];
% axfig = gca;
% axfig.FontSize =23;
% hold on
% xlabel('Polynomial Degree','fontsize',25.0)
% ylabel('Condition Number','fontsize',25.0)
% plot(ax_x,[cond_S_5,cond_S_9,cond_S_13,cond_S_25], 'k^-','linewidth',1.5)
% plot(ax_x, [cond_pre_5, cond_pre_9, cond_pre_13, cond_pre_25],'b*-','LineWidth',1.5)
% legend('\kappa(S)','\kappa(map^{-1}\cdot S)','location','northwest')




