load 'shape_nodal_inner'
load 'edge_inner.mat'
load 'nodal_inner_5.mat'
% load 'orthobasis.mat'
%% extract nodal inner shape
Mu_5 = M_5(1:30,1:30);
for i = 1:5
    for j = 1:5
        a = 5*(i-1) + 1;
        b = 5*(j-1) + 1;
        edge(i,j) = edge_inner(a,b);
        edge(j,i) = edge_inner(b,a);
    end
end
%%
scrsz = get(groot,'screensize');
figure('position',[55 125 scrsz(3)/1. scrsz(4)/1.8])

subplot(1,3,1)
hold on
grid on
ax = gca;
ax.FontSize = 17.0;
ax.View = [97, 17];
surf(Mu_5)
legend(ax, '(2,2) Block of \bf{M^{(n-1)}}, p=5', 'location', 'NorthOutside')

subplot(1,3,2)
hold on
grid on
bx = gca;
bx.FontSize = 17.0;
bx.View = [97, 17];
surf(nodal_inner)
legend(bx, '\bf{M_{\lambda}^{0}} \otimes ones(size(\bf{M_{\mu}^{(1)}})), p=5','location', 'NorthOutside')

subplot(1,3,3)
hold on
grid on
cx = gca;
cx.FontSize = 17.0;
cx.View = [97, 17];
surf(edge)
legend(cx, '\bf{M_{\mu}^{(1)}}, p=5','location', 'NorthOutside')

hold off


% figure()
% hold on
% ax = gca;
% ax.View = [97, 17];
% mesh(Mu_5)
% 
% figure()
% hold on
% ax = gca;
% ax.View = [97, 17];
% mesh(nodal_inner)


% shape_nodal_inner = diag(edge);

% %% extract edge inner shape
% for i = 1:40
%     for j = 1:40
%         a = 40*(i-1) + 1;
%         b = 40*(j-1) + 1;
%         edge(i,j) = edge_inner(a,b);
%         edge(j,i) = edge_inner(b,a);
%     end
% end
% shape_edge_inner = edge;

%%

for i = 1:26
    a_min(i) = floor(i*41./26 -0.00001);
    step = i*41./26-a_min(i);
    pre_val(i) = Spect_41(a_min(i)) + step*(Spect_41(a_min(i)+1)-Spect_41(a_min(i)));
end

pre_val = 1./(100.*pre_val);
pre_val = 0.5.*(pre_val+fliplr(pre_val));

for i = 1:26
    a_index = 1+(i-1)*25:i*25;
    precond(a_index, a_index) = pre_val(i)*eye(length(a_index));
end

preconditioner = precond * Before_25;

% Preconditioning of mass matrices 
% ortho basis 
aft_pre_5 = M_5_p\M_5 ;
cond_aft_5 = cond(aft_pre_5);

aft_pre_9 = M_9_p\M_9 ;
cond_aft_9 = cond(aft_pre_9);

aft_pre_13 = M_13_p\M_13 ;
cond_aft_13 = cond(aft_pre_13);

aft_pre_25 = M_25_p\M_25 ;
cond_aft_25 = cond(aft_pre_25);

% diagonal scaling
aft_pre_5_diag = diag(diag(M_5))\M_5 ;
cond_aft_5_diag = cond(aft_pre_5_diag);

aft_pre_9_diag = diag(diag(M_9))\M_9 ;
cond_aft_9_diag = cond(aft_pre_9_diag);

aft_pre_13_diag = diag(diag(M_13))\M_13 ;
cond_aft_13_diag = cond(aft_pre_13_diag);

aft_pre_25_diag = diag(diag(M_25))\M_25 ;
cond_aft_25_diag = cond(aft_pre_25_diag);

%%

figure()
hold on
axx = gca;
axx.FontSize = 18.0;
xlabel(axx,'Polynomial Degree','fontsize',25.0)
ylabel(axx,'Condition Number','fontsize',25.0)
plot([5,9,13,25], [cond_aft_5, cond_aft_9, cond_aft_13, cond_aft_25], 'b*-.','linewidth',1.5)
plot([5,9,13,25],[cond(M_5),cond(M_9),cond(M_13),cond(M_25)], 'k^-.','linewidth',1.5)
legend('\kappa (M)','\kappa (M_0^{-1}M)', 'location','northwest')
%%
scrsz = get(groot,'screensize');
figure('position',[55 125 scrsz(3)/1.05 scrsz(4)/1.7]);
ax1 = subplot(1,2,1);
ax1.FontSize = 25.0;
hold on
xlabel(ax1, 'Polynomial Degree','fontsize',25.0)
ylabel(ax1, 'Condition Number','fontsize',25.0)
plot([5,9,13,25], [cond_aft_5, cond_aft_9, cond_aft_13, cond_aft_25], 'b*-.','linewidth',1.5)
plot([5,9,13,25],[cond(M_5),cond(M_9),cond(M_13),cond(M_25)], 'k^-.','linewidth',1.5)
legend('\kappa (M)','\kappa (M_0^{-1}M)', 'location','northwest')

ax2 = subplot(1,2,2);
ax2.FontSize = 25.0;
hold on
ylabel(ax2, 'Condition Number','fontsize',25.0)
xlabel(ax2, 'Polynomial Degree','fontsize',25.0)
plot([5,9,13,25], [cond_aft_5, cond_aft_9, cond_aft_13, cond_aft_25], 'b*-.','linewidth',1.5)
plot([5,9,13,25], [cond_aft_5_diag, cond_aft_9_diag, cond_aft_13_diag, cond_aft_25_diag], 'bo-.','linewidth',1.5)
legend('\kappa (M_0^{-1}M), Orthogonal Basis',...
       '\kappa (M_0^{-1}M), Diagonal Scaling','location','east')



%% Schur Complement

