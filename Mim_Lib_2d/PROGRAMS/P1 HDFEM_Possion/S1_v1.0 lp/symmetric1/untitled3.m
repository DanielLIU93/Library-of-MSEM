% M_eig_13 =[max(eig(M_13)), min(eig(M_13))]
% 
% W_eig_13 =[max(eig(WedgeMat_13)), min(eig(WedgeMat_13))]
% 
% map_eig_13 =[max(eig(E21_13*E21_13')), min(eig(E21_13*E21_13'))]
% 
% S_eig_13 =[max(eig(S_13)), min(eig(S_13))] 

%%
[I,J] = size(M_13);
for i = 1:I
    for j = 1:J
        if M_13(i,j) <= 0.001
            M_13(i,j)= 0;
        end
    end
end
figure(1)
% mesh(M_25)
spy(M_13);
% ax = gca;
% ax.View = [105, 35];

% figure(2)
% mesh(M_5_p)
% ax = gca;
% ax.View = [105, 35];


