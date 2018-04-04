close all
clear
%% >>>>>>>>>>>>> p <<<<<<<<<<<<<<<<
figure
mycolor=hsv(3);  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
linetype={'-o','-s','-v'};
SaveFigureTo = 'COND_NUM';   %%%%%%%%%%%%%%%%%%%%%%%%%%%%
FigureFormat = '.pdf';
Legendnames={};

load('Res_170810_cond_num_n3.mat') %%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_to_plot = Res;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_lines = 1;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_data_per_line = 9; %%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_column_x = 1;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_column_y = 5;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Legends = {'n=3,c=0.1'};  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:num_lines
    plot(     data_to_plot((i-1)*num_data_per_line + 1 : i*num_data_per_line, data_column_x),...
                  data_to_plot((i-1)*num_data_per_line + 1 : i*num_data_per_line, data_column_y),...
                        char(linetype(i)),'color',mycolor(i,:),'linewidth',0.8);

    hold on;
    Legendnames=[   Legendnames;
                    strcat('$$',char(Legends(i)),'$$') ];
end

load('Res_170810_cond_num_n4.mat') %%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_to_plot = Res;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_lines = 1;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_data_per_line = 9; %%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_column_x = 1;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_column_y = 5;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Legends = {'n=4,c=0.1'};  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:num_lines
    plot(     data_to_plot((i-1)*num_data_per_line + 1 : i*num_data_per_line, data_column_x),...
                  data_to_plot((i-1)*num_data_per_line + 1 : i*num_data_per_line, data_column_y),...
                        char(linetype(i+1)),'color',mycolor(i+1,:),'linewidth',0.8);

    hold on;
    Legendnames=[   Legendnames;
                    strcat('$$',char(Legends(i)),'$$') ];
end

load('Res_170810_cond_num_n5.mat') %%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_to_plot = Res;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_lines = 1;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_data_per_line = 9; %%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_column_x = 1;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_column_y = 5;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Legends = {'n=5,c=0.1'};  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:num_lines
    plot(     data_to_plot((i-1)*num_data_per_line + 1 : i*num_data_per_line, data_column_x),...
                  data_to_plot((i-1)*num_data_per_line + 1 : i*num_data_per_line, data_column_y),...
                        char(linetype(i+2)),'color',mycolor(i+2,:),'linewidth',0.8);

    hold on;
    Legendnames=[   Legendnames;
                    strcat('$$',char(Legends(i)),'$$') ];
end

% title('Condition number','Interpreter','latex','FontSize',12);
set(gca,'TickLabelInterpreter','latex','FontSize',12);
xlabel('$$p$$'              ,'Interpreter','latex','FontSize',12);
ylabel('Cond\_Num'    ,'Interpreter','latex','FontSize',12);
handle_legend=legend(char(Legendnames),'location','best');
set(handle_legend,'Interpreter','latex','FontSize',10);
if isempty(SaveFigureTo)
    disp('      figure not saved.......');
else
    switch FigureFormat
        case '.eps'
            saveas(gcf,SaveFigureTo,'epsc') 
        case '.png'
            saveas(gcf,SaveFigureTo,'png') 
        case '.fig'
            saveas(gcf,SaveFigureTo,'fig')
        case '.pdf'
            export_fig(strcat(SaveFigureTo,'.pdf'),'-pdf','-r864','-painters','-transparent');
        otherwise
            error('wrong FigureFormat.......');
    end
end

